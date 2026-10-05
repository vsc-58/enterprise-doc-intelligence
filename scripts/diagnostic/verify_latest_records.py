"""
scripts/verify_latest_records.py — DIAGNOSTIC (print() allowed).

Checks the canonical read after the D31 refactor: one row per document, the
latest SUCCESS attempt, and a statement that stays composable when a caller
attaches its own WHERE and ORDER BY.

Read-only. No LLM, no network.

Usage:
    python -m scripts.verify_latest_records

Dependencies: src.storage.metadata_store, src.utils.logger.
"""

from __future__ import annotations

from sqlalchemy import select

from src.storage.metadata_store import (
    Document,
    ExtractedRecord,
    ExtractionStatus,
    get_session,
    latest_successful_records,
    latest_successful_stmt,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Entry point. Returns the process exit code."""
    pairs = latest_successful_records()
    doc_ids = [doc.id for doc, _ in pairs]

    print(f"\nrows returned        : {len(pairs)}")
    print(f"distinct documents   : {len(set(doc_ids))}")

    with get_session() as session:
        total_attempts = session.execute(
            select(ExtractedRecord.id)
        ).all()
        successes = session.execute(
            select(ExtractedRecord.id).where(
                ExtractedRecord.extraction_status == ExtractionStatus.SUCCESS.value
            )
        ).all()
    print(f"attempts in table    : {len(total_attempts)} ({len(successes)} success)")

    print("\nticker  year  revenue            net income")
    for doc, record in pairs:
        print(
            f"{doc.ticker:<7} {doc.filing_year}  "
            f"{record.total_revenue!s:<18} {record.net_income!s}"
        )

    # Composability: the same statement, filtered and ordered in SQL.
    with get_session() as session:
        filtered = session.execute(
            latest_successful_stmt()
            .where(Document.ticker == "AAPL")
            .order_by(ExtractedRecord.total_revenue.desc())
            .limit(1)
        ).all()
    print(f"\nfiltered (AAPL, limit 1): {len(filtered)} row")
    if filtered:
        doc, record = filtered[0]
        print(f"  {doc.ticker} {doc.filing_year} revenue={record.total_revenue}")

    ok = len(pairs) == len(set(doc_ids)) == 20 and len(filtered) == 1
    print("\nOK" if ok else "\nCHECK FAILED")
    logger.info("latest_records_verified", rows=len(pairs), ok=ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())