"""
scripts/build_narrative_sections.py
Persist narrative section slices to disk, once, so chunking is a local operation.

Separates the network step from the chunking step. get_narrative_sections()
resolves filings over the SEC endpoints (via edgartools, cached); re-chunking at
different parameters should not re-hit them. Writes one JSON artifact per
document to data/processed/ (gitignored, regenerable).

Run before scripts/embed_documents.py. Re-run only when the slicing logic or the
corpus changes.

Dependencies: sqlalchemy, src.ingestion.acquire, src.ingestion.narrative_sections,
src.storage.metadata_store, src.utils.config, src.utils.logger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

from src.ingestion.acquire import ensure_identity, select_10k_for_fiscal_year
from src.ingestion.narrative_sections import get_narrative_sections
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def artifact_path(cik: str, year: int) -> Path:
    """Return the on-disk path for one document's narrative section artifact.

    Args:
        cik: Zero-padded 10-digit CIK.
        year: Filing year (period-end convention, per Phase 1 D1).

    Returns:
        Path under settings.PROCESSED_DATA_PATH.
    """
    return Path(settings.PROCESSED_DATA_PATH) / f"{cik}_{year}_narrative.json"


def build_one(doc: Document) -> dict[str, int] | None:
    """Slice and persist narrative sections for one document.

    Args:
        doc: Document row supplying ticker, cik, filing_year, company_name.

    Returns:
        Mapping of item label to token count, or None if the document produced
        no usable sections (logged, never raised — one failure must not stop
        the batch).
    """
    try:
        filing = select_10k_for_fiscal_year(doc.ticker, doc.filing_year)
        if filing is None:
            logger.error(
                "filing_not_resolved", ticker=doc.ticker, year=doc.filing_year
            )
            return None
        sections = get_narrative_sections(filing, doc.company_name, doc.filing_year)
    except Exception as exc:
        logger.error(
            "narrative_slice_failed",
            ticker=doc.ticker,
            year=doc.filing_year,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None

    if not sections:
        logger.error("no_sections_to_write", ticker=doc.ticker, year=doc.filing_year)
        return None

    payload = {
        "cik": doc.cik,
        "ticker": doc.ticker,
        "company_name": doc.company_name,
        "filing_year": doc.filing_year,
        "sections": [section.model_dump() for section in sections],
    }

    path = artifact_path(doc.cik, doc.filing_year)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error(
            "artifact_write_failed",
            ticker=doc.ticker,
            year=doc.filing_year,
            path=str(path),
            error=str(exc),
        )
        return None

    return {section.item_label: section.token_count for section in sections}


def main(limit: int | None = None) -> None:
    """Build narrative section artifacts for every acquired document.

    Args:
        limit: Optional cap on documents processed.
    """
    ensure_identity()

    with get_session() as session:
        documents = list(session.scalars(select(Document).order_by(Document.ticker)))
    if limit:
        documents = documents[:limit]

    logger.info("build_started", document_count=len(documents))

    built, failed, total_tokens = 0, [], 0
    for index, doc in enumerate(documents, start=1):
        result = build_one(doc)
        if result is None:
            failed.append(doc.ticker)
            continue
        built += 1
        total_tokens += sum(result.values())
        if index % 5 == 0:
            logger.info("build_progress", processed=index, total=len(documents))

    logger.info(
        "build_complete",
        built=built,
        failed=len(failed),
        failed_tickers=failed,
        total_tokens=total_tokens,
        output_dir=settings.PROCESSED_DATA_PATH,
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)