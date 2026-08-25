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

logger = get_logger(__name__)

_SECTION_LABELS: dict[str, str] = {
    "edgartools_item8": "Item 8 (Financial Statements) from a 10-K",
    "fallback_fulltext": "the full text of a 10-K filing",
    "fallback_overcapture": "the full text of a 10-K filing",
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
        logger.warning(
            "extraction_validation_failed",
            strategy=strategy_id,
            cik=cik,
            year=year,
            error=str(parsing_error),
        )
    elif is_evidence:
        extraction = parsed.to_extraction()
        evidence = {
            name: ev.model_dump() for name, ev in parsed.evidence_map().items()
        }
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