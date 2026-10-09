# src/rag/chain.py
# Module: Narrative chain
# Purpose: Turn retrieved context into a structured, cited NarrativeResult —
#          claims for one company, or one verdict per company for corpus-wide
#          questions. Refusal text and citation validation live here, in Python.
# Depends on: asyncio, langchain-openai, pydantic v2, src.query.schemas,
#             src.rag.figure_guard, src.rag.prompts, src.rag.retriever,
#             src.storage.vector_store, src.utils.config, src.utils.logger,
#             src.utils.text
#
# Three guarantees, each enforced by code rather than by the prompt (D41):
#   1. Every claim cites at least one block that was actually retrieved. Cited
#      numbers are mapped back through CompanyContext.block(); a number that does
#      not exist is counted and dropped, and a claim left with none is dropped.
#   2. No stored metric reaches the model or the user: values are masked in the
#      context, and any that still appear in a claim are redacted.
#   3. The model never writes refusal text. It returns a typed outcome; Python
#      renders one fixed sentence per NarrativeRefusal.

import asyncio
import hashlib
import json
from collections.abc import Sequence
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.query.schemas import (
    CitedClaim,
    CompanyFinding,
    FindingStatus,
    Intent,
    NarrativeRefusal,
    NarrativeResult,
    SourceRef,
)
from src.rag.figure_guard import FigureGuardError, load_guard_values, withhold_figures
from src.rag.prompts import ANSWER_PROMPT, ANSWER_SYSTEM, VERDICT_PROMPT, VERDICT_SYSTEM
from src.rag.retriever import (
    CompanyContext,
    CompanyRef,
    format_context,
    retrieve_corpus_wide,
    retrieve_for_company,
)
from src.storage.vector_store import RetrievalStatus
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

SECTION_NAMES: dict[str, str] = {
    "Item 1": "Business (Item 1)",
    "Item 1A": "Risk Factors (Item 1A)",
    "Item 7": "MD&A (Item 7)",
    "Item 7A": "Market Risk (Item 7A)",
}

FIGURE_WITHHELD_NOTE = (
    "A financial figure was withheld from this passage-based answer; "
    "figures are reported only from the structured data."
)

_llm: ChatOpenAI | None = None


# --- Structured outputs ------------------------------------------------------
# Docstrings and field descriptions are sent to the model (D34): keep them short
# and written for the model.


class AnswerOutcome(str, Enum):
    """Whether the passages answer the question."""

    ANSWERED = "ANSWERED"
    NOT_ADDRESSED = "NOT_ADDRESSED"
    FIGURE_REQUESTED = "FIGURE_REQUESTED"


class DraftClaim(BaseModel):
    """One sentence making one point, with the passages that state it."""

    text: str = Field(description="One sentence.")
    passages: list[int] = Field(description="Numbers of the passages that state this sentence.")


class DraftAnswer(BaseModel):
    """An answer built only from the numbered passages."""

    outcome: AnswerOutcome
    claims: list[DraftClaim] = Field(
        default_factory=list, description="2 to 5 claims when ANSWERED, otherwise empty."
    )


class VerdictOutcome(str, Enum):
    """Whether this company's passages discuss the topic."""

    SUPPORTED = "SUPPORTED"
    NOT_FOUND = "NOT_FOUND"


class DraftVerdict(BaseModel):
    """One company's verdict on the topic."""

    outcome: VerdictOutcome
    claim: str | None = Field(
        default=None, description="One sentence on what the company discloses, if SUPPORTED."
    )
    passages: list[int] = Field(
        default_factory=list, description="Passage numbers stating it, if SUPPORTED."
    )


# --- Helpers -----------------------------------------------------------------


def chain_version() -> str:
    """
    Version identifier for both narrative prompts, their schemas and the model.

    Schemas are hashed with the prompts because with_structured_output sends
    them to the model as part of the call (D34).

    Returns:
        32-character hex digest.
    """
    payload = "|".join(
        [
            ANSWER_SYSTEM,
            VERDICT_SYSTEM,
            settings.OPENAI_MODEL,
            json.dumps(DraftAnswer.model_json_schema(), sort_keys=True),
            json.dumps(DraftVerdict.model_json_schema(), sort_keys=True),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _get_llm() -> ChatOpenAI:
    """
    Return the process-wide chat model, constructing it once.

    api_key passed explicitly: pydantic-settings does not populate os.environ
    (Phase 2 bug 5).

    Returns:
        The configured ChatOpenAI client.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    return _llm


def _section_phrase(sections: frozenset[str]) -> str:
    """
    Name a set of sections for a sentence.

    Args:
        sections: Stored section labels.

    Returns:
        e.g. "MD&A (Item 7) or Market Risk (Item 7A)".
    """
    names = [SECTION_NAMES.get(section, section) for section in sorted(sections)]
    return " or ".join(names) if names else "narrative"


def refusal_message(
    refusal: NarrativeRefusal,
    subject: str,
    sections: frozenset[str] = frozenset(),
) -> str:
    """
    The fixed sentence shown for a narrative refusal.

    One sentence per reason, so the user always sees why, and the refusal metric
    compares enums rather than wording.

    Args:
        refusal: Why there is no answer.
        subject: Display name of the company, or a phrase for the corpus.
        sections: Sections searched, named in SECTION_ABSENT and NOT_ADDRESSED.

    Returns:
        The sentence.
    """
    where = _section_phrase(sections)
    if refusal is NarrativeRefusal.SECTION_ABSENT:
        return f"The filing held for {subject} has no {where} text to answer this from."
    if refusal is NarrativeRefusal.NOT_ADDRESSED:
        return f"The {where} passages retrieved for {subject} do not address this question."
    if refusal is NarrativeRefusal.FIGURE_REQUESTED:
        return (
            f"Financial figures for {subject} come from the structured data — ask for "
            "the metric directly, for example its total revenue or net income."
        )
    if refusal is NarrativeRefusal.UNCITED:
        return f"I couldn't produce an answer about {subject} that is supported by cited passages."
    return f"I couldn't retrieve or read the narrative text for {subject} just now."


def _cite(numbers: Sequence[int], context: CompanyContext) -> tuple[list[SourceRef], int]:
    """
    Map cited block numbers to sources, counting numbers that do not exist.

    Duplicates are collapsed, order of first citation kept.

    Args:
        numbers: Block numbers the model cited.
        context: The context those numbers refer to.

    Returns:
        Sources for valid numbers, and the count of invalid ones.
    """
    sources: list[SourceRef] = []
    seen: set[int] = set()
    invalid = 0
    for number in numbers:
        if number in seen:
            continue
        seen.add(number)
        block = context.block(number)
        if block is None:
            invalid += 1
            continue
        sources.append(block.source_ref())
    return sources, invalid


async def _targets(ticker: str) -> frozenset[float]:
    """
    One company's stored values for the figure guard.

    Args:
        ticker: Company.

    Returns:
        Its stored values; empty if the company has no extraction record.

    Raises:
        FigureGuardError: if the structured store cannot be read.
    """
    values = await asyncio.to_thread(load_guard_values)
    return values.get(ticker, frozenset())


def _masked_context(context: CompanyContext, targets: frozenset[float]) -> tuple[str, int]:
    """
    Render a context for the model with stored figures withheld.

    Args:
        context: An OK context.
        targets: The company's stored values.

    Returns:
        Prompt-ready text and the number of values masked.
    """
    return withhold_figures(format_context(context), targets)


async def _invoke(
    prompt: ChatPromptTemplate,
    schema: type[BaseModel],
    variables: dict[str, str],
) -> BaseModel:
    """
    Run one structured-output call.

    Args:
        prompt: The ChatPromptTemplate.
        schema: Output model.
        variables: Prompt variables.

    Returns:
        The parsed output.

    Raises:
        RuntimeError: if the model output could not be parsed.
    """
    chain = prompt | _get_llm().with_structured_output(schema, include_raw=True)
    response = await chain.ainvoke(variables)
    parsed = response.get("parsed")
    if parsed is None:
        raise RuntimeError(f"unparseable output: {response.get('parsing_error')}")
    return parsed


# --- Single-company ----------------------------------------------------------


async def answer_for_company(
    question: str,
    intent: Intent,
    company: CompanyRef,
) -> NarrativeResult:
    """
    Answer a narrative question about one company, with validated citations.

    Never raises: retrieval, guard-data and LLM faults return FAILED so the
    executor can report this task and keep the rest of the plan.

    Args:
        question: The task's self-contained question.
        intent: Narrative intent, which fixes the sections.
        company: The resolved company.

    Returns:
        A NarrativeResult with claims, or a refusal.
    """
    context = await asyncio.to_thread(retrieve_for_company, question, intent, company)
    searched = context.sections_searched
    if context.status is RetrievalStatus.NO_MATCHING_CHUNKS:
        return NarrativeResult(refusal=NarrativeRefusal.SECTION_ABSENT, sections_searched=searched)
    if context.status is RetrievalStatus.FAILED:
        return NarrativeResult(refusal=NarrativeRefusal.FAILED, sections_searched=searched)

    try:
        targets = await _targets(company.ticker)
        context_text, masked = _masked_context(context, targets)
        draft = await _invoke(
            ANSWER_PROMPT,
            DraftAnswer,
            {
                "company": display_name(company.company_name, company.ticker),
                "fiscal_year": str(company.fiscal_year),
                "question": question,
                "context": context_text,
            },
        )
    except Exception as exc:  # noqa: BLE001 — guard data, parse or SDK error; must not sink the plan
        logger.error(
            "narrative_answer_failed",
            ticker=company.ticker,
            intent=intent.value,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return NarrativeResult(refusal=NarrativeRefusal.FAILED, sections_searched=searched)

    if draft.outcome is AnswerOutcome.FIGURE_REQUESTED:
        return NarrativeResult(
            refusal=NarrativeRefusal.FIGURE_REQUESTED,
            figures_masked=masked,
            sections_searched=searched,
        )
    if draft.outcome is AnswerOutcome.NOT_ADDRESSED or not draft.claims:
        return NarrativeResult(
            refusal=NarrativeRefusal.NOT_ADDRESSED,
            figures_masked=masked,
            sections_searched=searched,
        )

    claims: list[CitedClaim] = []
    invalid_total = 0
    redacted_total = 0
    for draft_claim in draft.claims:
        sources, invalid = _cite(draft_claim.passages, context)
        invalid_total += invalid
        if not sources:
            continue
        text, redacted = withhold_figures(draft_claim.text.strip(), targets)
        redacted_total += redacted
        claims.append(CitedClaim(text=text, sources=sources))

    logger.info(
        "narrative_answered",
        ticker=company.ticker,
        intent=intent.value,
        claims_drafted=len(draft.claims),
        claims_kept=len(claims),
        invalid_citations=invalid_total,
        figures_masked=masked,
        figures_redacted=redacted_total,
    )
    return NarrativeResult(
        claims=claims,
        refusal=None if claims else NarrativeRefusal.UNCITED,
        figures_masked=masked,
        figures_redacted=redacted_total,
        invalid_citations=invalid_total,
        sections_searched=searched,
    )


# --- Corpus-wide -------------------------------------------------------------


class _Verdict(BaseModel):
    """One company's validated verdict plus its guard and citation counters."""

    model_config = ConfigDict(frozen=True)

    finding: CompanyFinding
    masked: int = 0
    redacted: int = 0
    invalid: int = 0


def _finding(context: CompanyContext, status: FindingStatus,
             claim: CitedClaim | None = None) -> CompanyFinding:
    """
    Build a CompanyFinding for a context.

    Args:
        context: The company's retrieval.
        status: Its finding status.
        claim: The cited claim, for SUPPORTED.

    Returns:
        The finding.
    """
    return CompanyFinding(
        ticker=context.company.ticker,
        company_name=context.company.company_name,
        fiscal_year=context.company.fiscal_year,
        status=status,
        claim=claim,
    )


async def _judge(
    question: str,
    context: CompanyContext,
    targets: frozenset[float],
    semaphore: asyncio.Semaphore,
) -> _Verdict:
    """
    Run and validate one company's verdict.

    A SUPPORTED verdict with no valid citation is downgraded to NOT_FOUND: an
    uncited "yes" is not evidence.

    Args:
        question: The corpus-wide question.
        context: An OK context for one company.
        targets: That company's stored values.
        semaphore: Concurrency cap shared across companies.

    Returns:
        The validated verdict.
    """
    context_text, masked = _masked_context(context, targets)
    try:
        async with semaphore:
            draft = await _invoke(
                VERDICT_PROMPT,
                DraftVerdict,
                {
                    "company": display_name(context.company.company_name, context.company.ticker),
                    "fiscal_year": str(context.company.fiscal_year),
                    "question": question,
                    "context": context_text,
                },
            )
    except Exception as exc:  # noqa: BLE001 — one company must not sink the survey
        logger.error(
            "verdict_failed",
            ticker=context.company.ticker,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return _Verdict(finding=_finding(context, FindingStatus.FAILED), masked=masked)

    if draft.outcome is not VerdictOutcome.SUPPORTED or not draft.claim:
        return _Verdict(finding=_finding(context, FindingStatus.NOT_FOUND), masked=masked)

    sources, invalid = _cite(draft.passages, context)
    if not sources:
        logger.warning("verdict_uncited_downgraded", ticker=context.company.ticker)
        return _Verdict(
            finding=_finding(context, FindingStatus.NOT_FOUND), masked=masked, invalid=invalid
        )

    text, redacted = withhold_figures(draft.claim.strip(), targets)
    return _Verdict(
        finding=_finding(context, FindingStatus.SUPPORTED, CitedClaim(text=text, sources=sources)),
        masked=masked,
        redacted=redacted,
        invalid=invalid,
    )


async def survey_companies(
    question: str,
    intent: Intent,
    companies: Sequence[CompanyRef],
) -> NarrativeResult:
    """
    Answer a corpus-wide narrative question with one verdict per company (D39).

    Retrieval runs once in a worker thread; verdict calls run concurrently under
    a semaphore of settings.RAG_VERDICT_CONCURRENCY. Every company appears in the
    findings — searchable or not — so the answer can state its coverage.

    Never raises. Refuses only when nothing was searchable at all.

    Args:
        question: The task's self-contained question.
        intent: Narrative intent.
        companies: Every embedded company in the corpus.

    Returns:
        A NarrativeResult with findings, or a refusal.
    """
    contexts = await asyncio.to_thread(retrieve_corpus_wide, question, intent, companies)
    searchable = [c for c in contexts if c.status is RetrievalStatus.OK]

    if not searchable:
        all_absent = all(c.status is RetrievalStatus.NO_MATCHING_CHUNKS for c in contexts)
        return NarrativeResult(
            refusal=NarrativeRefusal.SECTION_ABSENT if all_absent and contexts
            else NarrativeRefusal.FAILED
        )

    try:
        values = await asyncio.to_thread(load_guard_values)
    except FigureGuardError:
        return NarrativeResult(refusal=NarrativeRefusal.FAILED)

    semaphore = asyncio.Semaphore(settings.RAG_VERDICT_CONCURRENCY)
    judged = await asyncio.gather(
        *(
            _judge(question, context, values.get(context.company.ticker, frozenset()), semaphore)
            for context in searchable
        )
    )
    by_ticker = {verdict.finding.ticker: verdict for verdict in judged}

    findings: list[CompanyFinding] = []
    for context in contexts:
        if context.company.ticker in by_ticker:
            findings.append(by_ticker[context.company.ticker].finding)
        elif context.status is RetrievalStatus.NO_MATCHING_CHUNKS:
            findings.append(_finding(context, FindingStatus.NOT_SEARCHABLE))
        else:
            findings.append(_finding(context, FindingStatus.FAILED))

    result = NarrativeResult(
        findings=findings,
        figures_masked=sum(v.masked for v in judged),
        figures_redacted=sum(v.redacted for v in judged),
        invalid_citations=sum(v.invalid for v in judged),
    )
    logger.info(
        "corpus_surveyed",
        intent=intent.value,
        supported=[f.ticker for f in findings if f.status is FindingStatus.SUPPORTED],
        not_found=sum(f.status is FindingStatus.NOT_FOUND for f in findings),
        not_searchable=[f.ticker for f in findings if f.status is FindingStatus.NOT_SEARCHABLE],
        failed=[f.ticker for f in findings if f.status is FindingStatus.FAILED],
        figures_masked=result.figures_masked,
        figures_redacted=result.figures_redacted,
        invalid_citations=result.invalid_citations,
    )
    return result