"""
scripts/inspect_chunks.py — DIAGNOSTIC (print() allowed).

Chunks documents from their narrative artifacts and reports size distribution,
boundary quality and metadata. No embedding, no Chroma, no API calls — pure
local inspection before any spend.

Usage:
    python -m scripts.inspect_chunks              # all documents, summary only
    python -m scripts.inspect_chunks BAC          # one ticker, with chunk text

Dependencies: sqlalchemy, src.ingestion.chunker, src.storage.metadata_store,
src.utils.logger.
"""

from __future__ import annotations

import sys
from collections import Counter

import tiktoken
from sqlalchemy import select

from src.ingestion.chunker import chunk_document
from src.storage.metadata_store import Document, get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
SHOW_CHUNKS = 6
HEAD_CHARS = 180
TAIL_CHARS = 120


def inspect_one(ticker: str) -> None:
    """Print consecutive Item 1A chunks for one ticker, head and tail of each.

    Args:
        ticker: Ticker to inspect.
    """
    with get_session() as session:
        doc = session.scalar(select(Document).where(Document.ticker == ticker))
        if doc is None:
            print(f"no document for ticker {ticker}")
            return
        cik, year, company = doc.cik, doc.filing_year, doc.company_name

    chunks = chunk_document(cik, year)
    risk_chunks = [c for c in chunks if c["metadata"]["source_section"] == "Item 1A"]

    print(f"\n{company} {year} — {len(chunks)} chunks, {len(risk_chunks)} in Item 1A")
    for chunk in risk_chunks[:SHOW_CHUNKS]:
        meta = chunk["metadata"]
        tokens = len(_ENCODING.encode(chunk["text"]))
        print(f"\n--- {meta['source_section']} #{meta['chunk_index']}  {tokens} tok  id={chunk['id'][:12]}")
        print(f"HEAD: {chunk['text'][:HEAD_CHARS]!r}")
        print(f"TAIL: {chunk['text'][-TAIL_CHARS:]!r}")


def main(ticker: str | None = None) -> None:
    """Chunk every document (or one) and report the size distribution.

    Args:
        ticker: Optional single ticker to inspect in detail.
    """
    if ticker:
        inspect_one(ticker)
        return

    with get_session() as session:
        documents = list(session.scalars(select(Document).order_by(Document.ticker)))

    total, sections = 0, Counter()
    sizes: list[int] = []
    failed: list[str] = []

    print(f"{'ticker':<8}{'chunks':>8}{'min':>7}{'med':>7}{'max':>7}")
    for doc in documents:
        chunks = chunk_document(doc.cik, doc.filing_year)
        if not chunks:
            failed.append(doc.ticker)
            continue

        doc_sizes = sorted(len(_ENCODING.encode(c["text"])) for c in chunks)
        sizes.extend(doc_sizes)
        total += len(chunks)
        for chunk in chunks:
            sections[chunk["metadata"]["source_section"]] += 1

        print(
            f"{doc.ticker:<8}{len(chunks):>8}{doc_sizes[0]:>7}"
            f"{doc_sizes[len(doc_sizes) // 2]:>7}{doc_sizes[-1]:>7}"
        )

    sizes.sort()
    print(f"\ntotal chunks: {total:,}")
    print(f"tokens embedded: {sum(sizes):,}  (~${sum(sizes) / 1_000_000 * 0.02:.3f})")
    print(f"size min/median/max: {sizes[0]} / {sizes[len(sizes) // 2]} / {sizes[-1]}")
    print(f"under 50 tok: {sum(1 for s in sizes if s < 50)}")
    print(f"over 520 tok: {sum(1 for s in sizes if s > 520)}")
    print(f"\nby section: {dict(sections)}")
    print(f"unique ids: {total} produced")
    if failed:
        print(f"FAILED: {failed}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)