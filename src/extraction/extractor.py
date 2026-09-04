"""
src/extraction/extractor.py
Runs a single extraction strategy over a document's sliced sections.

Reads the section artifacts produced by scripts/build_sections.py, assembles the
prompt input (cover + financial statements, identically for every strategy),
binds the correct schema (evidence schema for S3, plain for the rest), invokes
the model, and caches the result on disk.

Caching is keyed by a hash of EVERYTHING that determines the output — strategy
id, prompt template text, model name, and the input text — so a changed prompt
or a re-sliced section produces a new key and forces a fresh call. A cache keyed
on document id alone would silently score new logic against stale outputs.

Transport failures are never cached (a timeout is not a result); validation
failures ARE cached (the model genuinely emitted unusable output, which is a
real experimental outcome).

Dependencies: langchain-openai, langchain-core, pydantic v2, src.extraction.*,
src.utils.*
"""

import hashlib
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.extraction.prompts import EVIDENCE_STRATEGIES, STRATEGIES
from src.extraction.schemas import FilingExtraction, FilingExtractionWithEvidence
from src.utils.config import settings
from src.utils.logger import get_logger

from sqlalchemy import select

from src.eval.grounding import check_evidence_consistency, check_grounding
from src.storage.metadata_store import (
    EVIDENCE_FIELDS,
    Document,
    ExtractionStatus,
    create_extracted_record,
    get_session,
)

logger = get_logger(__name__)

_SECTION_LABELS: dict[str, str] = {
    "edgartools_item8": "Item 8 (Financial Statements) from a 10-K",
    "fallback_fulltext": "the full text of a 10-K filing",
    # Kept so pre-rename cached artifacts still resolve; superseded by
    # fallback_fulltext_oversize (the ceiling was replaced by a budget check).
    "fallback_overcapture": "the full text of a 10-K filing",
    "fallback_fulltext_oversize": "the full text of a 10-K filing",
}


class ExtractionResult(BaseModel):
    """
    One strategy's output for one document, plus the metadata scoring needs.

    Attributes:
        strategy_id: which strategy produced this.
        cik: canonical 10-digit CIK.
        year: fiscal-period-end year.
        extraction: the validated extraction, or None if the model's output
            failed schema validation.
        evidence: S3's per-field source lines, keyed by field name; None for
            non-evidence strategies.
        section_source: provenance of the financial-statements input, carried
            through so scoring can split results by input quality.
        input_tokens: prompt tokens billed.
        output_tokens: completion tokens billed.
        parsing_error: the validation error message, if extraction is None.
        from_cache: whether this result was read from disk rather than called.
    """

    model_config = ConfigDict(str_strip_whitespace=False)

    strategy_id: str
    cik: str
    year: int
    extraction: FilingExtraction | None = None
    evidence: dict[str, dict[str, str | float | None]] | None = None
    section_source: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    parsing_error: str | None = None
    from_cache: bool = Field(default=False, exclude=True)


def load_sections(cik: str, year: int) -> dict[str, dict]:
    """
    Load a document's section artifact from disk.

    Args:
        cik: canonical 10-digit CIK.
        year: fiscal-period-end year.

    Returns:
        The 'sections' mapping (name -> {text, token_count, section_source}).

    Raises:
        FileNotFoundError: if the artifact has not been built.
    """
    path = Path(settings.RAW_DATA_PATH) / f"{cik}_{year}_sections.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Section artifact missing: {path}. Run scripts/build_sections.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["sections"]


def assemble_input(sections: dict[str, dict]) -> tuple[str, str]:
    """
    Concatenate the sections into the single text block the prompts consume.

    Cover and financial statements are joined with labelled separators, in the
    same order for every strategy — the input is a controlled constant, so only
    the prompt varies between arms.

    Args:
        sections: the loaded section mapping.

    Returns:
        (assembled_text, financial_statements_section_source)
    """
    cover = sections["cover"]
    financials = sections["financial_statements"]
    text = (
        "=== COVER PAGE ===\n"
        f"{cover['text']}\n\n"
        "=== FINANCIAL STATEMENTS ===\n"
        f"{financials['text']}"
    )
    return text, financials["section_source"]


def _cache_key(
    strategy_id: str, model: str, prompt_repr: str, section_text: str
) -> str:
    """
    Build the content hash that identifies this exact call.

    Includes every input that determines the output, so any change to the
    prompt, the model, or the sliced text invalidates the cache automatically.

    Args:
        strategy_id: the strategy.
        model: the model name.
        prompt_repr: a stable string form of the prompt template.
        section_text: the assembled input text.

    Returns:
        A hex digest.
    """
    payload = "\x00".join([strategy_id, model, prompt_repr, section_text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _cache_path(strategy_id: str, cik: str, year: int, key: str) -> Path:
    """Return the on-disk path for a cached result."""
    directory = Path(settings.EVAL_DATA_PATH) / "raw_outputs" / strategy_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cik}_{year}_{key}.json"


def _read_cache(path: Path) -> ExtractionResult | None:
    """
    Read a cached ExtractionResult, or None if absent/unreadable.

    A corrupt cache file is logged and treated as a miss rather than raising —
    the correct recovery is to re-call, not to crash the run.
    """
    if not path.exists():
        return None
    try:
        result = ExtractionResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        result.from_cache = True
        return result
    except Exception as exc:
        logger.warning(
            "cache_unreadable",
            path=str(path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None


def _write_cache(path: Path, result: ExtractionResult) -> None:
    """Persist an ExtractionResult to disk; log and continue on failure."""
    try:
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error(
            "cache_write_failed",
            path=str(path),
            error_type=type(exc).__name__,
            error=str(exc),
        )


def _usage(raw: object) -> tuple[int, int]:
    """
    Pull input/output token counts off the raw AIMessage.

    Returns (0, 0) if usage metadata is absent rather than raising — missing
    cost data must not fail an otherwise successful extraction, but it is
    logged so a systematically empty cost column is noticed.
    """
    usage = getattr(raw, "usage_metadata", None)
    if not usage:
        logger.warning("usage_metadata_missing")
        return 0, 0
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def extract_with_strategy(
    strategy_id: str, cik: str, year: int, use_cache: bool = True
) -> ExtractionResult:
    """
    Run one extraction strategy over one document.

    Loads the document's sections, assembles the input, binds the strategy's
    schema, and invokes the model. Returns an ExtractionResult whose
    `extraction` is None when the model's output failed schema validation
    (a real experimental outcome, recorded rather than raised).

    Args:
        strategy_id: one of the keys in prompts.STRATEGIES.
        cik: canonical 10-digit CIK.
        year: fiscal-period-end year.
        use_cache: read from / write to the on-disk cache.

    Returns:
        The ExtractionResult for this (strategy, document) pair.

    Raises:
        KeyError: if strategy_id is not a known strategy.
        FileNotFoundError: if the document's section artifact is missing.
    """
    if strategy_id not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {strategy_id}")

    prompt = STRATEGIES[strategy_id]
    sections = load_sections(cik, year)
    section_text, section_source = assemble_input(sections)

    key = _cache_key(
        strategy_id, settings.OPENAI_MODEL, str(prompt), section_text
    )
    path = _cache_path(strategy_id, cik, year, key)

    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            logger.info(
                "extraction_cache_hit", strategy=strategy_id, cik=cik, year=year
            )
            return cached

    is_evidence = strategy_id in EVIDENCE_STRATEGIES
    schema = FilingExtractionWithEvidence if is_evidence else FilingExtraction

    llm = ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0,)
    chain = prompt | llm.with_structured_output(schema, include_raw=True)

    inputs: dict[str, str] = {"section_text": section_text}
    if "{section_label}" in str(prompt):
        inputs["section_label"] = _SECTION_LABELS.get(
            section_source, "a 10-K filing"
        )

    logger.info(
        "extraction_call",
        strategy=strategy_id,
        cik=cik,
        year=year,
        section_source=section_source,
    )
    response = chain.invoke(inputs)  # transport errors propagate: not cached

    raw = response.get("raw")
    parsed = response.get("parsed")
    parsing_error = response.get("parsing_error")
    input_tokens, output_tokens = _usage(raw)

    extraction: FilingExtraction | None = None
    evidence: dict[str, dict[str, str | float | None]] | None = None

    if parsed is None:
        logger.warning("extraction_validation_failed", strategy=strategy_id, cik=cik, year=year, error=str(parsing_error))
    elif is_evidence:
        try:
            extraction = parsed.to_extraction()
            evidence = {name: ev.model_dump() for name, ev in parsed.evidence_map().items()}
        except Exception as exc:
            logger.error("evidence_projection_failed", strategy=strategy_id, cik=cik, year=year, error=str(exc))
            parsing_error = f"projection failed: {exc}"
    else:
        extraction = parsed

    result = ExtractionResult(
        strategy_id=strategy_id,
        cik=cik,
        year=year,
        extraction=extraction,
        evidence=evidence,
        section_source=section_source,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        parsing_error=str(parsing_error) if parsing_error else None,
    )

    if use_cache:
        _write_cache(path, result)

    return result

# Fields written to extracted_records. business_description and
# primary_risk_factors are DELIBERATELY absent: with only cover + Item 8 in the
# input they have no source, and inspection of the eval-10 showed the model
# returning Item 8 note headings ("Uncertain tax positions") under
# primary_risk_factors — real text, correct location, wrong field. Storing that
# under a column named for Item 1A would misrepresent it. They remain in the
# schema (removing them would change the tool definition sent to the model and
# silently invalidate every cached response) and are dropped at the write.
PERSISTED_FIELDS: tuple[str, ...] = (
    "company_name",
    "fiscal_year_end",
    "auditor_name",
    *EVIDENCE_FIELDS,
)


def prompt_version(strategy_id: str) -> str:
    """
    Return a short, derived identifier for a strategy's prompt template.

    Derived from the template text rather than hand-maintained, so it cannot
    drift from the prompt it names — a manually bumped version is the kind of
    thing that silently does not get bumped. Shares the hash input with the
    response cache, so a record's prompt_version joins to the cached raw output
    that produced it.

    Args:
        strategy_id: one of the keys in prompts.STRATEGIES.

    Returns:
        The first 12 hex chars of the template's SHA-256.

    Raises:
        KeyError: if strategy_id is not a known strategy.
    """
    if strategy_id not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {strategy_id}")
    digest = hashlib.sha256(str(STRATEGIES[strategy_id]).encode("utf-8"))
    return digest.hexdigest()[:12]


def _split_evidence(
    evidence: dict[str, dict] | None,
) -> dict[str, str | None]:
    """
    Flatten an evidence map to field -> cited source line.

    Args:
        evidence: the per-field evidence map, or None.

    Returns:
        Cited line per evidence field; None where the field carried no citation.
    """
    evidence = evidence or {}
    return {
        field: (evidence.get(field) or {}).get("source_line")
        for field in EVIDENCE_FIELDS
    }


def _extract_one(document_id: int, strategy_id: str) -> tuple[str, int, int, bool]:
    """
    Run one document through extraction and persist exactly one record.

    Grounding runs at write time, not at read time, so a figure's lineage verdict
    is stored beside the figure and no consumer has to recompute it. Both checks
    run: check_grounding (is the cited line real?) and check_evidence_consistency
    (does the cited line contain the value?). Neither failing fails the record —
    they mark individual figures unverified, because discarding four verified
    figures over one bad citation loses more than it protects.

    Args:
        document_id: the document to extract.
        strategy_id: the winning strategy to apply.

    Returns:
        (status value, input tokens, output tokens, whether the result was cached).
        Token counts are the tokens the response cost when it was FIRST made — a
        cached result reports the same numbers it reported originally, so the
        caller must use the cached flag to separate spend from replay.

    Raises:
        ValueError: if document_id does not exist.
    """
    with get_session() as session:
        row = session.execute(
            select(
                Document.cik,
                Document.filing_year,
                Document.ticker,
                Document.company_name,
            ).where(Document.id == document_id)
        ).first()
    if row is None:
        raise ValueError(f"no document with id {document_id}")
    cik, year, ticker, company = row

    version = prompt_version(strategy_id)

    try:
        result = extract_with_strategy(strategy_id, cik, year)
    except Exception as exc:
        # The call never completed: transport error, missing section artifact,
        # context-length rejection. Not a statement about the model, so it is
        # recorded as technical and the document stays pending for a re-run.
        logger.error(
            "extraction_technical_failure",
            document_id=document_id,
            ticker=ticker,
            year=year,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        create_extracted_record(
            document_id=document_id,
            extraction_status=ExtractionStatus.TECHNICAL_FAILED,
            extraction_strategy_used=strategy_id,
            prompt_version=version,
            model_name=settings.OPENAI_MODEL,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
        return ExtractionStatus.TECHNICAL_FAILED.value, 0, 0, False

    if result.extraction is None:
        # The call completed and was billed, but the output failed validation.
        # The row is written so the failure is diagnosable rather than lost.
        logger.warning(
            "extraction_validation_failure",
            document_id=document_id,
            ticker=ticker,
            year=year,
            error=result.parsing_error,
        )
        create_extracted_record(
            document_id=document_id,
            extraction_status=ExtractionStatus.EXTRACTION_FAILED,
            extraction_strategy_used=strategy_id,
            prompt_version=version,
            model_name=settings.OPENAI_MODEL,
            section_source=result.section_source,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            failure_reason=result.parsing_error or "schema validation failed",
        )
        return ExtractionStatus.EXTRACTION_FAILED.value, result.input_tokens, result.output_tokens, result.from_cache

    source_text, _ = assemble_input(load_sections(cik, year))
    present = check_grounding(result.evidence, source_text)
    consistent = check_evidence_consistency(result.evidence)

    dumped = result.extraction.model_dump()
    fields = {name: dumped.get(name) for name in PERSISTED_FIELDS if name in dumped}

    create_extracted_record(
        document_id=document_id,
        extraction_status=ExtractionStatus.SUCCESS,
        extraction_strategy_used=strategy_id,
        prompt_version=version,
        model_name=settings.OPENAI_MODEL,
        fields=fields,
        source_lines=_split_evidence(result.evidence),
        evidence_present=present,
        evidence_consistent=consistent,
        evidence_json=(
            json.dumps(result.evidence, indent=2) if result.evidence else None
        ),
        section_source=result.section_source,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )

    ungrounded = [f for f, ok in present.items() if ok is False]
    inconsistent = [f for f, ok in consistent.items() if ok is False]
    if ungrounded or inconsistent:
        logger.warning(
            "extraction_evidence_flags",
            document_id=document_id,
            ticker=ticker,
            year=year,
            ungrounded=ungrounded or None,
            inconsistent=inconsistent or None,
        )

    logger.info(
        "extraction_succeeded",
        document_id=document_id,
        ticker=ticker,
        company=company,
        year=year,
        section_source=result.section_source,
        from_cache=result.from_cache,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    return ExtractionStatus.SUCCESS.value, result.input_tokens, result.output_tokens, result.from_cache


def run_extraction_pipeline(
    document_ids: list[int], strategy_id: str = "S3"
) -> dict[str, object]:
    """
    Extract a batch of documents, writing one record per attempt.

    Per-item error handling: any failure is caught, recorded with the document's
    identity, and the batch continues. Zero silent drops — every id in
    document_ids leaves either a persisted record or a logged failure, and the
    returned counts sum to len(document_ids).

    Token counts include cached results, whose tokens were billed on an earlier
    run; cached_hits reports how many of the totals were not paid for again.

    Args:
        document_ids: documents to process, in order.
        strategy_id: the strategy to apply. Defaults to the Phase 2 winner.

    Returns:
        Summary with processed, succeeded, extraction_failed, technical_failed,
        input_tokens, output_tokens, and cached_hits.

    Raises:
        KeyError: if strategy_id is not a known strategy (raised before any work
            begins, so a typo cannot half-run a batch).
    """
    version = prompt_version(strategy_id)  # fails fast on an unknown strategy
    logger.info(
        "extraction_pipeline_start",
        documents=len(document_ids),
        strategy=strategy_id,
        prompt_version=version,
        model=settings.OPENAI_MODEL,
    )

    counts = {"success": 0, "extraction_failed": 0, "technical_failed": 0}
    billed_input, billed_output = 0, 0
    cached_input, cached_output = 0, 0
    cached_hits = 0

    for document_id in document_ids:
        try:
            status, in_tok, out_tok, cached = _extract_one(document_id, strategy_id)
            counts[status] += 1
            if cached:
                cached_hits += 1
                cached_input += in_tok
                cached_output += out_tok
            else:
                billed_input += in_tok
                billed_output += out_tok
        except Exception as exc:
            # The record write itself failed (dangling FK, disk, DB lock). The
            # document stays pending; the batch continues.
            counts["technical_failed"] += 1
            logger.error(
                "extraction_record_write_failed",
                document_id=document_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    logger.info(
        "extraction_pipeline_summary",
        processed=len(document_ids),
        succeeded=counts["success"],
        extraction_failed=counts["extraction_failed"],
        technical_failed=counts["technical_failed"],
        cached_hits=cached_hits,
        billed_input_tokens=billed_input,
        billed_output_tokens=billed_output,
        replayed_input_tokens=cached_input,
        replayed_output_tokens=cached_output,
        strategy=strategy_id,
        prompt_version=version,
    )
    return {
        "processed": len(document_ids),
        "succeeded": counts["success"],
        "extraction_failed": counts["extraction_failed"],
        "technical_failed": counts["technical_failed"],
        "cached_hits": cached_hits,
        "billed_input_tokens": billed_input,
        "billed_output_tokens": billed_output,
        "replayed_input_tokens": cached_input,
        "replayed_output_tokens": cached_output,
    }