"""
scripts/diagnostic/dump_section_inventory.py — DIAGNOSTIC (print() allowed, per
the scripts/diagnostic convention).

Phase 5C step 1. Answers one question before any eval label is written: for
every document, which narrative sections can retrieval actually reach, and so
which (company, intent) pairs should answer and which should refuse with
SECTION_ABSENT?

Three views, each checked against the next:
  1. Artifacts  — data/processed/{cik}_{year}_narrative.json, the accepted slices.
  2. Store      — the Chroma collection's metadata, i.e. what retrieval sees.
  3. Intents    — sections_for(intent, ticker), i.e. the alias table applied.

The eval labels are derived from view 3 evaluated over view 2. View 1 exists to
prove view 2 is not stale: the artifacts are re-chunked locally (no API call)
and the per-(ticker, section) counts must equal the store's. A mismatch means
the store was built from different artifacts or parameters, and every label
derived from it would be measuring the wrong corpus.

No embedding or LLM calls. Opening the store constructs the embedding client
but never calls it.

Writes data/eval/section_inventory.json — committed: it is the reference the
routing-set section tags are verified against, so it is evidence, not scratch.

Usage:
    python -m scripts.diagnostic.dump_section_inventory

Exit code 1 if any document has no chunks or the store disagrees with its
artifacts.

Dependencies: sqlalchemy, src.ingestion.chunker, src.query.intents,
src.query.schemas, src.storage.metadata_store, src.storage.vector_store,
src.utils.config, src.utils.logger.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from src.ingestion.chunker import chunk_document, chunk_params_hash
from src.query.intents import (
    CORPUS_SECTIONS,
    INTENT_POLICY,
    ITEM_1,
    ITEM_1A,
    ITEM_7,
    ITEM_7A,
    SECTION_ALIASES,
    sections_for,
)
from src.query.schemas import Intent, Tool
from src.storage.metadata_store import Document, get_session
from src.storage.vector_store import VectorStoreError, get_store
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_NAME = "section_inventory.json"
SECTION_ORDER: list[str] = [ITEM_1, ITEM_1A, ITEM_7, ITEM_7A]
RAG_INTENTS: list[Intent] = [i for i, p in INTENT_POLICY.items() if p.tool is Tool.RAG]
ABSENT = "ABSENT"


def load_documents() -> list[tuple[str, str, int]]:
    """Read every tracked document.

    Returns:
        (ticker, cik, filing_year) per document, ordered by ticker.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: if the metadata store cannot be read.
    """
    with get_session() as session:
        rows = session.execute(
            select(Document.ticker, Document.cik, Document.filing_year)
            .order_by(Document.ticker)
        ).all()
    return [(r.ticker, r.cik, r.filing_year) for r in rows]


def store_counts() -> dict[str, Counter[str]]:
    """Count stored chunks per ticker and section label.

    Uses the collection's metadata only; no vectors or text are loaded.

    Returns:
        ticker -> Counter of section label -> chunk count.

    Raises:
        VectorStoreError: if the store cannot be opened or read.
    """
    store = get_store()
    try:
        # store._collection is private langchain-chroma API (known watch item);
        # it is the only way to read metadata without a similarity query.
        result = store._collection.get(include=["metadatas"])
    except Exception as exc:
        logger.error(
            "store_metadata_read_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise VectorStoreError("Could not read collection metadata") from exc

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for meta in result["metadatas"] or []:
        counts[str(meta["source_ticker"])][str(meta["source_section"])] += 1
    return counts


def artifact_counts(cik: str, year: int) -> Counter[str]:
    """Re-chunk one document's artifact locally and count chunks per section.

    Args:
        cik: Zero-padded CIK.
        year: Filing year.

    Returns:
        Counter of section label -> chunk count. Empty if the artifact is
        missing or unreadable (chunk_document logs which).
    """
    return Counter(c["metadata"]["source_section"] for c in chunk_document(cik, year))


def intent_matrix(stored: Counter[str], ticker: str) -> dict[str, dict[str, Any]]:
    """Evaluate every RAG intent for one company against its stored sections.

    Args:
        stored: Section label -> chunk count for this ticker.
        ticker: The company.

    Returns:
        intent value -> {sections: searched labels after aliases,
        eligible_chunks: chunks the filter can reach,
        expected: "answerable" or "section_absent"}.
    """
    matrix: dict[str, dict[str, Any]] = {}
    for intent in RAG_INTENTS:
        searched = sorted(sections_for(intent, ticker))
        eligible = sum(stored.get(s, 0) for s in searched)
        matrix[intent.value] = {
            "sections": searched,
            "eligible_chunks": eligible,
            "expected": "answerable" if eligible else "section_absent",
        }
    return matrix


def main() -> int:
    """Build, print and write the inventory.

    Returns:
        Process exit code: 0 clean, 1 if any check failed.
    """
    try:
        documents = load_documents()
        stored_by_ticker = store_counts()
    except Exception as exc:
        logger.error("inventory_load_failed", error_type=type(exc).__name__, error=str(exc))
        print(f"FAILED to load inputs: {type(exc).__name__}: {exc}")
        return 1

    problems: list[str] = []
    per_doc: dict[str, Any] = {}

    held = {t for t, _, _ in documents}
    for orphan in sorted(set(stored_by_ticker) - held):
        problems.append(f"{orphan}: chunks in store but no Document row")

    # ---- View 1 vs view 2: store per (ticker, section) --------------------------
    header = f"{'ticker':<7}{'year':>6}" + "".join(f"{s:>9}" for s in SECTION_ORDER) + f"{'total':>8}  store==artifact"
    print("\nSTORED CHUNKS PER SECTION (store view; '-' = section not stored)")
    print(header)
    print("-" * len(header))

    for ticker, cik, year in documents:
        stored = stored_by_ticker.get(ticker, Counter())
        rebuilt = artifact_counts(cik, year)
        match = stored == rebuilt

        if not stored:
            problems.append(f"{ticker}: no chunks in store")
        if not match:
            diff = {s: (stored.get(s, 0), rebuilt.get(s, 0))
                    for s in sorted(set(stored) | set(rebuilt))
                    if stored.get(s, 0) != rebuilt.get(s, 0)}
            problems.append(f"{ticker}: store vs artifact (store, artifact) {diff}")
        unexpected = set(stored) - CORPUS_SECTIONS
        if unexpected:
            problems.append(f"{ticker}: unexpected section labels {sorted(unexpected)}")

        cells = "".join(f"{stored.get(s, 0) or '-':>9}" for s in SECTION_ORDER)
        print(f"{ticker:<7}{year:>6}{cells}{sum(stored.values()):>8}  {'yes' if match else 'NO'}")

        per_doc[ticker] = {
            "cik": cik,
            "filing_year": year,
            "stored_sections": {s: stored[s] for s in SECTION_ORDER if stored.get(s)},
            "aliases": {k: sorted(v) for k, v in SECTION_ALIASES.get(ticker, {}).items()},
            "store_matches_artifact": match,
            "intents": intent_matrix(stored, ticker),
        }

    total = sum(sum(c.values()) for c in stored_by_ticker.values())
    by_section = Counter()
    for c in stored_by_ticker.values():
        by_section.update(c)
    print(f"\ntotal chunks in store: {total:,}   by section: "
          + ", ".join(f"{s} {by_section.get(s, 0)}" for s in SECTION_ORDER))

    # ---- View 3: intent x ticker, after aliases ----------------------------------
    short = {i: i.value[:10] for i in RAG_INTENTS}
    header = f"{'ticker':<7}" + "".join(f"{short[i]:>12}" for i in RAG_INTENTS)
    print("\nELIGIBLE CHUNKS PER RAG INTENT, AFTER ALIASES (ABSENT = expect section_absent)")
    print(header)
    print("-" * len(header))
    for ticker, _, _ in documents:
        row = per_doc[ticker]["intents"]
        cells = "".join(
            f"{(row[i.value]['eligible_chunks'] or ABSENT):>12}" for i in RAG_INTENTS
        )
        flag = "  *alias" if ticker in SECTION_ALIASES else ""
        print(f"{ticker:<7}{cells}{flag}")

    print("\nIntent -> canonical sections: " + "; ".join(
        f"{i.value}={sorted(INTENT_POLICY[i].sections)}" for i in RAG_INTENTS))

    # ---- Write ------------------------------------------------------------------
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_params_hash": chunk_params_hash(),
        "total_chunks": total,
        "chunks_by_section": {s: by_section.get(s, 0) for s in SECTION_ORDER},
        "documents": per_doc,
        "problems": problems,
    }
    out = Path(settings.EVAL_DATA_PATH) / OUTPUT_NAME
    try:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.error("inventory_write_failed", path=str(out),
                     error_type=type(exc).__name__, error=str(exc))
        print(f"FAILED to write {out}: {exc}")
        return 1

    print(f"\nwritten: {out}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("checks: every document has chunks; store matches artifacts; labels known.")
    return 0


if __name__ == "__main__":
    sys.exit(main())