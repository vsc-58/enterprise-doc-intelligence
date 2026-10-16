# src/eval/attribution.py
# Module: Attribution support — the validated LLM judge (D43)
# Purpose: Decide, for one answer sentence and the blocks it cites, whether the
#          blocks support every detail of the sentence; plus the agreement and
#          stability statistics that decide whether the judge may be trusted.
# Depends on: langchain-core, langchain-openai, pydantic v2, src.storage.vector_store,
#             src.utils.config, src.utils.logger
#
# The judge sees the claim, the company it is about, and the cited blocks — each
# headed by its company, fiscal year and section. Not the user's question and
# not the rest of the answer: it rules on support, not relevance. The company is
# metadata, not the question: a claim that says "the company" could otherwise be
# "supported" by another company's block, which is exactly the citation error a
# judge must catch.
#
# The judge model (gpt-6-luna) differs from the system model (gpt-4o-mini) to
# reduce self-preference. It is evaluation tooling only.

import hashlib
import json
import re
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.storage.vector_store import RetrievedChunk
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class JudgeLabel(str, Enum):
    """Binary support verdict. "Mostly supported" is a failure in a financial answer."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class JudgeVerdict(BaseModel):
    """
    The judge's structured output.

    Attributes:
        verdict: SUPPORTED only if every detail of the claim is stated in, or
            directly follows from, the excerpts.
        supporting_span: The shortest verbatim span from the excerpts that
            supports the claim; empty when NOT_SUPPORTED.
        unsupported_detail: The detail of the claim the excerpts do not
            support; empty when SUPPORTED.
    """

    verdict: JudgeLabel
    supporting_span: str = Field(description="Shortest verbatim span from the excerpts "
                                             "supporting the claim; empty if NOT_SUPPORTED.")
    unsupported_detail: str = Field(description="The claim detail the excerpts do not "
                                                "support; empty if SUPPORTED.")


JUDGE_SYSTEM = """You check one sentence from an answer about a company's annual report (Form 10-K) \
against the report excerpts it cites.

Return SUPPORTED only if every factual detail in the sentence is stated in, or directly follows from, \
the excerpts: the company, figures, dates and periods, causes, comparisons and qualifiers. \
Paraphrase, summarising and combining details from several excerpts are fine.

Return NOT_SUPPORTED if any detail is missing from the excerpts, contradicts them, is more specific \
than they are, or comes from an excerpt about a different company than the one the sentence is about. \
Use only the excerpts, never outside knowledge.

When SUPPORTED, quote the shortest verbatim span from the excerpts that supports the sentence. \
When NOT_SUPPORTED, name the unsupported detail."""

JUDGE_HUMAN = """The sentence is about: {company}

Sentence:
{claim}

Cited excerpts:
{excerpts}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])

_llm: ChatOpenAI | None = None


def judge_version() -> str:
    """
    Version of everything that can change a verdict: prompt, model, reasoning
    setting, structured-output method and the output schema (whose field
    descriptions are sent to the model).

    Returns:
        32-character hex digest.
    """
    schema = json.dumps(JudgeVerdict.model_json_schema(), sort_keys=True)
    payload = "|".join([JUDGE_SYSTEM, JUDGE_HUMAN, settings.JUDGE_MODEL,
                        str(settings.JUDGE_REASONING_EFFORT), settings.JUDGE_STRUCTURED_METHOD,
                        schema])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _get_llm() -> ChatOpenAI:
    """
    The process-wide judge client, constructed once.

    No temperature is set: reasoning-class models reject non-default values.
    reasoning_effort is passed only when configured.

    Returns:
        The configured ChatOpenAI client.
    """
    global _llm
    if _llm is None:
        kwargs: dict = {"model": settings.JUDGE_MODEL, "api_key": settings.OPENAI_API_KEY,
                        "max_retries": settings.JUDGE_MAX_RETRIES}
        if settings.JUDGE_REASONING_EFFORT:
            kwargs["reasoning_effort"] = settings.JUDGE_REASONING_EFFORT
        _llm = ChatOpenAI(**kwargs)
    return _llm


def render_excerpts(blocks: list[RetrievedChunk]) -> str:
    """
    Number and head the cited blocks as the judge sees them.

    Args:
        blocks: Cited chunks, in citation order.

    Returns:
        "[1] Company · FY2023 · Item 1A\\n<text>" blocks separated by blank lines.
    """
    return "\n\n".join(
        f"[{n}] {b.company_name} · FY{b.fiscal_year} · {b.section}\n{b.text.strip()}"
        for n, b in enumerate(blocks, start=1)
    )


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace and quote styles, for span matching."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


class JudgeResult(BaseModel):
    """
    One judge call's outcome.

    Attributes:
        label: The verdict, or None if the call failed or returned nothing usable.
        supporting_span: As returned.
        unsupported_detail: As returned.
        span_found: For SUPPORTED, whether the quoted span really occurs in the
            excerpts (whitespace- and case-insensitive). A SUPPORTED verdict whose
            "quote" is not in the text is reported, not overridden.
        error: Failure description when label is None.
    """

    model_config = ConfigDict(frozen=True)

    label: JudgeLabel | None
    supporting_span: str = ""
    unsupported_detail: str = ""
    span_found: bool | None = None
    error: str | None = None


def judge_claim(company: str, claim: str, blocks: list[RetrievedChunk]) -> JudgeResult:
    """
    Judge one claim against its cited blocks.

    Batch-safe: never raises. A failed call returns label None with the error,
    so one bad call cannot sink a run of hundreds; the caller counts failures.

    Args:
        company: Display name of the company the claim is about.
        claim: The answer sentence.
        blocks: The chunks it cites.

    Returns:
        The JudgeResult.
    """
    excerpts = render_excerpts(blocks)
    try:
        chain = JUDGE_PROMPT | _get_llm().with_structured_output(
            JudgeVerdict, method=settings.JUDGE_STRUCTURED_METHOD, include_raw=True
        )
        response = chain.invoke({"company": company, "claim": claim, "excerpts": excerpts})
    except Exception as exc:  # noqa: BLE001 — batch item: record and continue
        logger.warning("judge_call_failed", error_type=type(exc).__name__, error=str(exc)[:200])
        return JudgeResult(label=None, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    parsed: JudgeVerdict | None = response.get("parsed")
    if parsed is None:
        logger.warning("judge_parse_failed", error=str(response.get("parsing_error"))[:200])
        return JudgeResult(label=None, error=f"parse: {str(response.get('parsing_error'))[:200]}")
    span_found = None
    if parsed.verdict is JudgeLabel.SUPPORTED:
        span_found = bool(parsed.supporting_span) and _normalise(parsed.supporting_span) in _normalise(excerpts)
    return JudgeResult(label=parsed.verdict, supporting_span=parsed.supporting_span,
                       unsupported_detail=parsed.unsupported_detail, span_found=span_found)


# --- Statistics ------------------------------------------------------------------


def majority(labels: list[JudgeLabel | None]) -> JudgeLabel | None:
    """
    The majority label over runs, ignoring failed calls.

    Args:
        labels: One label per run (None = failed call).

    Returns:
        The majority label; NOT_SUPPORTED on a tie (a tie means doubt, and doubt
        is not support); None if every call failed.
    """
    present = [label for label in labels if label is not None]
    if not present:
        return None
    supported = sum(label is JudgeLabel.SUPPORTED for label in present)
    return JudgeLabel.SUPPORTED if supported > len(present) - supported else JudgeLabel.NOT_SUPPORTED


def flipped(labels: list[JudgeLabel | None]) -> bool:
    """Whether a claim's label changed across runs (failed calls ignored)."""
    return len({label for label in labels if label is not None}) > 1


def cohen_kappa(a: list[JudgeLabel], b: list[JudgeLabel]) -> float | None:
    """
    Cohen's kappa for two binary raters.

    Kappa, not raw agreement: most real claims are supported, so a judge that
    always says SUPPORTED would score high raw agreement and kappa near 0.

    Args:
        a: Labels from rater A.
        b: Labels from rater B, same items, same order.

    Returns:
        Kappa; None if undefined (no items, or both raters constant on the same
        label so chance agreement is 1).

    Raises:
        ValueError: if the lists differ in length.
    """
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x is y for x, y in zip(a, b)) / n
    pa = sum(x is JudgeLabel.SUPPORTED for x in a) / n
    pb = sum(x is JudgeLabel.SUPPORTED for x in b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


# --- Claims and validation items -------------------------------------------------


class JudgedItem(BaseModel):
    """
    Anything the judge rules on: a real claim, a validation item or a seed.

    Attributes:
        id: Stable identifier ("R26|1|c1" for a claim, "T05" or "S3-02" in the
            validation set).
        company: Ticker of the company the claim is about.
        claim: The sentence.
        sources: Cited chunk keys, "TICKER|section|index".
        question_id: Eval question the claim came from (None for seeds).
        kind: "claim" (single-company answer) or "finding" (corpus-wide survey).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    company: str
    claim: str
    sources: list[str]
    question_id: str | None = None
    kind: str | None = None


def split_key(key: str) -> tuple[str, str, int]:
    """
    Parse a chunk key.

    Args:
        key: "TICKER|section|index".

    Returns:
        (ticker, section, index).

    Raises:
        ValueError: if the key is malformed.
    """
    ticker, section, index = key.split("|")
    return ticker, section, int(index)


def _keys(sources: list[dict]) -> list[str]:
    """Chunk keys from SourceRef dumps."""
    return [f"{s['ticker']}|{s['section']}|{s['chunk_index']}" for s in sources]


def claims_from_execution(document: dict, mode: str = "end_to_end", run: int = 1) -> list[JudgedItem]:
    """
    Every judged sentence in one run of an execution results file.

    Single-company answers contribute each cited claim; corpus-wide surveys
    contribute each supported finding's claim. Ids match those in the
    validation set, so a validation claim can be traced to its source.

    Args:
        document: A parsed execution results file.
        mode: "end_to_end" (what users receive) or "oracle".
        run: 1-based run number.

    Returns:
        The claims, in question order.

    Raises:
        KeyError / IndexError: if the file lacks that mode or run.
    """
    out: list[JudgedItem] = []
    for record in document["modes"][mode]["records"][run - 1]:
        for task in record.get("tasks", []):
            for n, claim in enumerate(task.get("claims") or [], start=1):
                out.append(JudgedItem(id=f"{record['id']}|{task['task_id']}|c{n}",
                                      company=task["company"], claim=claim["text"],
                                      sources=_keys(claim["sources"]),
                                      question_id=record["id"], kind="claim"))
            for finding in task.get("findings") or []:
                claim = finding.get("claim")
                if claim:
                    out.append(JudgedItem(id=f"{record['id']}|{task['task_id']}|{finding['ticker']}",
                                          company=finding["ticker"], claim=claim["text"],
                                          sources=_keys(claim["sources"]),
                                          question_id=record["id"], kind="finding"))
    return out


class LabelSheetError(ValueError):
    """Raised when the blind labelling sheet is incomplete or has an invalid label."""


def parse_label_sheet(text: str, expected_ids: list[str]) -> dict[str, str]:
    """
    Read S/N labels from the blind labelling sheet.

    Each item is a "## <id>" heading followed, somewhere before the next
    heading, by a "Label: S" or "Label: N" line (case-insensitive).

    Args:
        text: The sheet's markdown.
        expected_ids: Every id that must be labelled.

    Returns:
        Id -> "S" or "N".

    Raises:
        LabelSheetError: if an item is missing, unlabelled or labelled with
            anything other than S or N.
    """
    labels: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        heading = re.match(r"^##\s+(\S+)\s*$", line)
        if heading:
            current = heading.group(1)
            continue
        found = re.match(r"^\s*Label:\s*(\S*)\s*$", line, flags=re.IGNORECASE)
        if found and current:
            value = found.group(1).upper()
            if value not in ("S", "N"):
                raise LabelSheetError(f"{current}: label must be S or N, got {found.group(1)!r}")
            labels[current] = value
    missing = [i for i in expected_ids if i not in labels]
    if missing:
        raise LabelSheetError(f"unlabelled items: {', '.join(missing)}")
    return {i: labels[i] for i in expected_ids}
