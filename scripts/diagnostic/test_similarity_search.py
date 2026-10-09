"""
scripts/diagnostic/test_similarity_search.py — DIAGNOSTIC (print() allowed).

Validates retrieval against the built vector store in three parts:

1. REGRESSION — the six Phase 4 queries, unfiltered, k=3. The query API changed
   in 5B, the store did not, so best distances must reproduce the Phase 4 table
   (0.3168 / 0.3413 / 0.3446 / 0.4046 / 0.4183 / 0.4955) to float noise. A
   difference means the new code path is not searching what the old one did.
2. FILTERS AND STATUSES — first exercise of Chroma's $and / $in syntax on the
   real store. Asserts every returned chunk satisfies its filter, and that a
   filter matching nothing (INTC has no Item 7) returns NO_MATCHING_CHUNKS
   rather than an empty OK.
3. COVERAGE — the measurement behind the per-company design (D39): how many
   distinct companies plain top-k reaches for the headline query as k grows,
   versus per-company retrieval from ONE embedding.

Calls the OpenAI embedding API about ten times, ~10 tokens each: effectively
free. Read-only.

Expected-source tags use tickers, not registrant names (D30). QCOM and META are
tagged at company granularity only (D20, D24).

Usage:
    python -m scripts.diagnostic.test_similarity_search

Dependencies: src.query.resolve, src.storage.vector_store, src.utils.logger.
"""

from __future__ import annotations

from src.query.resolve import load_index
from src.storage.vector_store import (
    RetrievalResult,
    RetrievalStatus,
    collection_count,
    embed_query,
    query,
    query_by_vector,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_K = 3
SNIPPET_CHARS = 160
DISTANCE_TOLERANCE = 1e-3

# (question, expected_ticker | None, expected_section | None, phase-4 best distance)
REGRESSION: list[tuple[str, str | None, str | None, float]] = [
    ("Describe Amazon's core business segments.", "AMZN", None, 0.3168),
    ("What supply chain risks does Apple face?", "AAPL", "Item 1A", 0.3413),
    ("What was Apple's total revenue in 2022?", None, None, 0.3446),
    ("Which companies flagged interest rate risk?", None, "Item 7A", 0.4046),
    ("Why did revenue grow this year?", None, "Item 7", 0.4183),
    ("What is Apple's current stock price?", None, None, 0.4955),
]

# (question, ticker, sections, expected status)
FILTERED: list[tuple[str, str, frozenset[str], RetrievalStatus]] = [
    ("What supply chain risks does Apple face?", "AAPL",
     frozenset({"Item 1A"}), RetrievalStatus.OK),
    ("What is Apple's strategy?", "AAPL",
     frozenset({"Item 1", "Item 7"}), RetrievalStatus.OK),
    ("Why did Intel's revenue change?", "INTC",
     frozenset({"Item 7"}), RetrievalStatus.NO_MATCHING_CHUNKS),
    ("What interest rate exposure does JPMorgan have?", "JPM",
     frozenset({"Item 7A", "Item 7"}), RetrievalStatus.NO_MATCHING_CHUNKS),
]

COVERAGE_QUESTION = "Which companies flagged interest rate risk?"
COVERAGE_SECTIONS = frozenset({"Item 7A", "Item 7"})
COVERAGE_KS = (5, 10, 20, 50)


def show(result: RetrievalResult) -> None:
    """Print a result's status and chunks."""
    print(f"   status={result.status.value}" + (f"  ({result.detail})" if result.detail else ""))
    for rank, chunk in enumerate(result.chunks, start=1):
        snippet = " ".join(chunk.text.split())[:SNIPPET_CHARS]
        print(f"   {rank}. {chunk.ticker} FY{chunk.fiscal_year} {chunk.section} "
              f"#{chunk.chunk_index}  distance={chunk.distance:.4f}\n      {snippet}...")


def regression() -> bool:
    """Part 1. Returns True if every distance reproduces Phase 4."""
    print(f"\n{'=' * 78}\n1. REGRESSION vs Phase 4 (unfiltered, k={TOP_K})")
    ok = True
    for question, ticker, section, expected in REGRESSION:
        print(f"\nQ: {question}")
        result = query(question, k=TOP_K)
        show(result)
        if result.status is not RetrievalStatus.OK:
            print("   FAIL: no results")
            ok = False
            continue
        best = result.chunks[0].distance
        same = abs(best - expected) <= DISTANCE_TOLERANCE
        tickers = {c.ticker for c in result.chunks}
        sections = {c.section for c in result.chunks}
        tag_ok = (ticker is None or ticker in tickers) and (section is None or section in sections)
        print(f"   best={best:.4f} phase4={expected:.4f} {'MATCH' if same else 'DRIFT'}"
              f"  tags {'met' if tag_ok else 'MISSED'}")
        ok = ok and same
    return ok


def filtered() -> bool:
    """Part 2. Returns True if every status and every chunk matches its filter."""
    print(f"\n{'=' * 78}\n2. FILTERS AND STATUSES (k={TOP_K})")
    ok = True
    for question, ticker, sections, expected in FILTERED:
        print(f"\nQ: {question}  [ticker={ticker} sections={sorted(sections)}]")
        result = query(question, k=TOP_K, ticker=ticker, sections=sections)
        show(result)
        leaks = [c for c in result.chunks if c.ticker != ticker or c.section not in sections]
        passed = result.status is expected and not leaks
        print(f"   expected={expected.value} -> {'PASS' if passed else 'FAIL'}"
              + (f"  ({len(leaks)} chunks outside the filter)" if leaks else ""))
        ok = ok and passed
    return ok


def coverage() -> None:
    """Part 3. Distinct companies reached by plain top-k vs per-company retrieval."""
    print(f"\n{'=' * 78}\n3. COVERAGE: {COVERAGE_QUESTION}  sections={sorted(COVERAGE_SECTIONS)}")
    vector = embed_query(COVERAGE_QUESTION)

    for k in COVERAGE_KS:
        result = query_by_vector(vector, k=k, sections=COVERAGE_SECTIONS)
        counts: dict[str, int] = {}
        for chunk in result.chunks:
            counts[chunk.ticker] = counts.get(chunk.ticker, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: -item[1])
        print(f"   plain top-{k:<3} {len(counts):>2} companies  {ranked}")

    # Tickers from the corpus index rather than a hand list, so the diagnostic
    # follows the corpus. Embedded documents only: the rest have no chunks.
    tickers = sorted(e.ticker for e in load_index().entries if e.is_embedded)
    print(f"\n   per-company top-2, one embedding, {len(tickers)} embedded companies:")
    reached = 0
    for ticker in tickers:
        result = query_by_vector(vector, k=2, ticker=ticker, sections=COVERAGE_SECTIONS)
        best = f"{result.chunks[0].distance:.4f}" if result.chunks else "-"
        print(f"   {ticker:<6} {result.status.value:<20} best={best}")
        reached += result.status is RetrievalStatus.OK
    print(f"   per-company reached {reached}/{len(tickers)} "
          "(the rest hold neither section)")


def main() -> int:
    """Entry point. Returns the process exit code."""
    count = collection_count()
    print(f"collection holds {count:,} chunks")
    if count <= 0:
        print("EMPTY OR UNREADABLE COLLECTION — nothing to test")
        return 1

    regression_ok = regression()
    filters_ok = filtered()
    coverage()

    print(f"\n{'=' * 78}\nregression: {'PASS' if regression_ok else 'FAIL'}   "
          f"filters: {'PASS' if filters_ok else 'FAIL'}")
    logger.info("similarity_diagnostic_complete",
                regression_ok=regression_ok, filters_ok=filters_ok)
    return 0 if regression_ok and filters_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())