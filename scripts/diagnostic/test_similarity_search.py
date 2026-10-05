"""
scripts/test_similarity_search.py — DIAGNOSTIC (print() allowed).

Validates retrieval against the built vector store: does a question return
chunks that actually answer it, from the companies and sections it should?

Read-only apart from the embedding call for each query (~10 tokens each,
effectively free). No writes to Chroma or SQLite.

The expected_company / expected_section tags are the seed of Phase 5's
retrieval hit-rate metric (05_evaluation_strategy.md 6.1). Two documents carry a
known label imprecision: QCOM and META have Item 1 contained inside their Item
1A slice, so Item 1 content for those two is labelled Item 1A. Queries touching
them are tagged at company granularity only.

Usage:
    python -m scripts.test_similarity_search

Dependencies: src.storage.vector_store, src.utils.logger.
"""

from __future__ import annotations

from typing import Any

from src.storage.vector_store import collection_count, query
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_K = 3
SNIPPET_CHARS = 200

# (question, expected_company | None, expected_section | None, note)
# None means "no single correct source" — scored by eye, not by tag.
QUERIES: list[tuple[str, str | None, str | None, str]] = [
    (
        "What supply chain risks does Apple face?",
        "Apple Inc.",
        "Item 1A",
        "single-company risk lookup; the baseline case",
    ),
    (
        "Which companies flagged interest rate risk?",
        None,
        "Item 1A",
        "cross-company risk analysis — the architecture's headline demo query",
    ),
    (
        "Describe Amazon's core business segments.",
        "AMAZON COM INC",
        None,
        "business description; Item 1 or Item 7, either is legitimate",
    ),
    (
        "Why did revenue grow this year?",
        None,
        "Item 7",
        "MD&A question; INTC/JPM/XOM have no Item 7 and cannot appear",
    ),
    (
        "What was Apple's total revenue in 2022?",
        None,
        None,
        "OUT OF SCOPE for RAG — the SQL path owns this. Checks what the store "
        "returns when the answer is not in it; Item 8 is excluded by design.",
    ),
    (
        "What is Apple's current stock price?",
        None,
        None,
        "OUT OF SCOPE entirely — nothing in the corpus supports an answer. "
        "Phase 5's chain must refuse; here we only look at the scores.",
    ),
]


def run_query(
    question: str,
    expected_company: str | None,
    expected_section: str | None,
    note: str,
) -> dict[str, Any]:
    """Run one query and print its results.

    Args:
        question: Query text.
        expected_company: Company the top-k should include, or None.
        expected_section: Item label the top-k should include, or None.
        note: Why this query is in the set.

    Returns:
        A dict recording whether each expectation was met.
    """
    print(f"\n{'=' * 78}\nQ: {question}\n   ({note})")

    results = query(question, n_results=TOP_K)
    if not results:
        print("   NO RESULTS")
        return {"question": question, "company_hit": False, "section_hit": False}

    companies = {r["metadata"]["source_company"] for r in results}
    sections = {r["metadata"]["source_section"] for r in results}

    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        snippet = " ".join(result["text"].split())[:SNIPPET_CHARS]
        print(
            f"\n   {rank}. {meta['source_company']} FY{meta['source_year']} "
            f"{meta['source_section']} #{meta['chunk_index']}  "
            f"distance={result['score']:.4f}"
        )
        print(f"      {snippet}...")

    company_hit = expected_company is None or expected_company in companies
    section_hit = expected_section is None or expected_section in sections

    if expected_company and not company_hit:
        print(f"\n   MISS: expected {expected_company}, got {sorted(companies)}")
    if expected_section and not section_hit:
        print(f"   MISS: expected {expected_section}, got {sorted(sections)}")

    return {
        "question": question,
        "company_hit": company_hit,
        "section_hit": section_hit,
        "best_distance": results[0]["score"],
        "worst_distance": results[-1]["score"],
    }


def main() -> None:
    """Run the query set and summarise."""
    count = collection_count()
    print(f"collection holds {count:,} chunks, top-{TOP_K} per query")
    if count <= 0:
        print("EMPTY OR UNREADABLE COLLECTION — nothing to test")
        return

    outcomes = [run_query(*entry) for entry in QUERIES]

    print(f"\n{'=' * 78}\nSUMMARY")
    tagged = [o for o in outcomes if "best_distance" in o]
    hits = sum(1 for o in tagged if o["company_hit"] and o["section_hit"])
    print(f"expectations met: {hits}/{len(tagged)}")
    print(f"\n{'best':>8}{'worst':>8}  question")
    for outcome in tagged:
        print(
            f"{outcome['best_distance']:>8.4f}{outcome['worst_distance']:>8.4f}  "
            f"{outcome['question'][:56]}"
        )


if __name__ == "__main__":
    main()