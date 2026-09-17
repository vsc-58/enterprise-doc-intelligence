"""
scripts/test_narrative_sections.py — DIAGNOSTIC (print() allowed).

Exercises get_narrative_sections() across the corpus before any chunking.
Read-only: no writes to db/, data/eval/ or Chroma. No API calls.

Dependencies: src.ingestion.narrative_sections, src.ingestion.acquire,
src.storage.metadata_store, src.utils.logger.
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from src.ingestion.acquire import ensure_identity, select_10k_for_fiscal_year
from src.ingestion.narrative_sections import (
    NARRATIVE_ITEMS,
    get_narrative_sections,
)
from src.storage.metadata_store import Document, get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

HEAD_CHARS = 400


def main(limit: int | None = None) -> None:
    """Slice narrative sections for every acquired document and report.

    Args:
        limit: Optional cap on documents processed, for a fast first pass.
    """
    ensure_identity()

    with get_session() as session:
        documents = list(session.scalars(select(Document).order_by(Document.ticker)))
    if limit:
        documents = documents[:limit]

    rows: list[tuple[str, dict[str, int]]] = []
    failed: list[str] = []

    for doc in documents:
        try:
            filing = select_10k_for_fiscal_year(doc.ticker, doc.filing_year)
            if filing is None:
                failed.append(doc.ticker)
                continue
            sections = get_narrative_sections(filing, doc.company_name, doc.filing_year)
        except Exception as exc:
            logger.error(
                "narrative_slice_failed",
                ticker=doc.ticker,
                year=doc.filing_year,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            failed.append(doc.ticker)
            continue

        if not sections:
            failed.append(doc.ticker)
            continue

        rows.append((doc.ticker, {s.item_label: s.token_count for s in sections}))

        print(f"\n{'=' * 78}\n{doc.ticker} {doc.filing_year}")
        for section in sections:
            print(f"  {section.item_label:<8} {section.token_count:>8,} tok")
            print(f"    HEAD: {section.text[:HEAD_CHARS]!r}")

    print(f"\n{'=' * 78}\nCOVERAGE ({len(rows)} documents)")
    header = f"{'ticker':<8}" + "".join(f"{i:>10}" for i in NARRATIVE_ITEMS) + f"{'total':>10}"
    print(header)
    for ticker, found in rows:
        cells = "".join(
            f"{found.get(item, 0):>10,}" if item in found else f"{'-':>10}"
            for item in NARRATIVE_ITEMS
        )
        print(f"{ticker:<8}{cells}{sum(found.values()):>10,}")

    corpus_tokens = sum(sum(found.values()) for _, found in rows)
    print(f"\ncorpus tokens: {corpus_tokens:,}")
    print(f"est. chunks @500 tok: ~{corpus_tokens // 450:,}")
    print(f"est. embedding cost:  ${corpus_tokens * 1.1 / 1_000_000 * 0.02:.3f}")
    for item in NARRATIVE_ITEMS:
        have = sum(1 for _, found in rows if item in found)
        print(f"  {item:<8} present on {have}/{len(rows)}")
    if failed:
        print(f"\nFAILED / EMPTY: {failed}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)