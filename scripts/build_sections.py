"""
scripts/build_sections.py
Build the Phase 2 section artifacts for the evaluation set.

For each eval document: read its Document row (for the exact on-disk text path
and CIK), load the raw filing text, resolve the filing object (for Item 8), run
get_sections, and write data/raw/{cik}_{year}_sections.json. Prints a provenance
summary so the fallback distribution is visible at a glance.

The eval set is defined here as (ticker, year) pairs; everything else (CIK,
local_path) is read from SQLite, never re-derived — the acquired row is the
source of truth for what text was saved.

Run: python -m scripts.build_sections
"""

import json
from pathlib import Path

from sqlalchemy import select

from src.ingestion.acquire import ensure_identity, select_10k_for_fiscal_year
from src.ingestion.sections import Section, get_sections
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Eval set: 10 documents, full sector spread, nulls allowed (per-field n/a).
# (ticker, fiscal-period-end year) — resolved to CIK/local_path via SQLite.
EVAL_SET: list[tuple[str, int]] = [
    ("AAPL", 2021),
    ("MSFT", 2022),
    ("GOOGL", 2021),
    ("AMZN", 2023),
    ("WMT", 2023),
    ("JNJ", 2021),
    ("NFLX", 2023),
    ("V", 2022),
    ("ADBE", 2022),
    ("INTC", 2023),
]


def _load_document_rows(
    session, eval_set: list[tuple[str, int]]
) -> dict[tuple[str, int], Document]:
    """
    Load the Document row for each (ticker, year) in the eval set.

    Args:
        session: an open SQLAlchemy session.
        eval_set: the (ticker, year) pairs to load.

    Returns:
        Mapping of (ticker, year) -> Document for every pair that has a row.
        Pairs with no matching row are omitted (the caller reports them).
    """
    rows: dict[tuple[str, int], Document] = {}
    for ticker, year in eval_set:
        doc = session.execute(
            select(Document).where(
                Document.ticker == ticker, Document.filing_year == year
            )
        ).scalar_one_or_none()
        if doc is not None:
            rows[(ticker, year)] = doc
    return rows


def _write_sections(
    cik: str, year: int, sections: dict[str, Section]
) -> Path:
    """
    Write the section dict to data/raw/{cik}_{year}_sections.json.

    Args:
        cik: canonical 10-digit CIK (from the Document row).
        year: fiscal-period-end year.
        sections: the get_sections output.

    Returns:
        The path written.
    """
    payload = {
        "cik": cik,
        "year": year,
        "sections": {name: sec.model_dump() for name, sec in sections.items()},
    }
    out_path = Path(settings.RAW_DATA_PATH) / f"{cik}_{year}_sections.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    """Build and persist section artifacts for the eval set; print a summary."""
    ensure_identity()

    provenance_counts: dict[str, int] = {}
    written, failed, missing_rows = 0, 0, []

    with get_session() as session:
        rows = _load_document_rows(session, EVAL_SET)

    for ticker, year in EVAL_SET:
        doc = rows.get((ticker, year))
        if doc is None:
            missing_rows.append((ticker, year))
            logger.error("eval_doc_no_row", ticker=ticker, year=year)
            continue

        try:
            raw_text = Path(doc.local_path).read_text(encoding="utf-8")
            filing = select_10k_for_fiscal_year(ticker, year)
            sections = get_sections(filing, raw_text)

            fs = sections["financial_statements"]
            cover = sections["cover"]
            provenance_counts[fs.section_source] = (
                provenance_counts.get(fs.section_source, 0) + 1
            )

            out_path = _write_sections(doc.cik, year, sections)
            written += 1
            logger.info(
                "sections_built",
                ticker=ticker,
                year=year,
                cik=doc.cik,
                fs_source=fs.section_source,
                fs_tokens=fs.token_count,
                cover_tokens=cover.token_count,
                path=str(out_path),
            )
        except Exception as exc:  # per-item: one failure never stops the batch
            failed += 1
            logger.error(
                "sections_build_failed",
                ticker=ticker,
                year=year,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    logger.info(
        "build_sections_summary",
        written=written,
        failed=failed,
        missing_rows=missing_rows or None,
        provenance=provenance_counts,
    )


if __name__ == "__main__":
    main()