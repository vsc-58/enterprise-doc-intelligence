"""
scripts/diagnostic/probe_heading_context.py — DIAGNOSTIC (print() allowed).

Phase 5C retrieval follow-up: why do Walmart's liquidity and cash-requirement
chunks rank low for R28 and N29 (dev)? Two hypotheses, tested by changing one
side at a time against the same chunks and the same embedding model:

  H1 heading lost — a chunk deep in "Liquidity and Capital Resources" no longer
     carries that heading, so its embedding lacks the topic word.
     Manipulation: prefix EVERY eligible chunk (relevant or not — prefixing only
     the relevant ones would rig the test) with "<company> · <section> ·
     <subsection heading>", the heading being the last MD&A subsection heading
     seen at or before the chunk, in chunk order.
  H2 vocabulary gap — the question's words differ from the filing's.
     Manipulation: keep the chunks; rewrite the question in the filing's terms.

Conditions: baseline, H1 (prefixed chunks), H2 (rewritten query), H1+H2. For
each, the rank of every gold-relevant chunk and capped recall@5.
Reading: H1 lifts the deep chunks and H2 does not -> heading loss; the reverse
-> vocabulary; both -> both; neither -> something else (e.g. compressed
distances within one company's MD&A).

Dev questions only (R28, N29): held-out questions must not shape a fix. The
rewritten queries were written after reading the relevant chunks, so H2 is an
upper bound on what better wording could do, not a realistic user query.

Embedding calls only: one batch per chunk variant (~35 chunks x 2) plus four
queries. Nothing is written to the store.

Usage:
    python -m scripts.diagnostic.probe_heading_context

Dependencies: numpy, src.eval.retrieval_eval, src.storage.vector_store,
src.utils.config.
"""

from __future__ import annotations

import sys

import numpy as np

from src.eval.retrieval_eval import GoldError, load_gold
from src.storage.vector_store import (
    VectorStoreError,
    build_filter,
    embed_query,
    embed_texts,
    get_store,
)

TICKER = "WMT"
SECTION = "Item 7"
COMPANY = "Walmart Inc."
# Walmart's Item 7 subsection headings, in filing order. Matched as plain text.
HEADINGS = [
    "Overview",
    "Company Performance Metrics",
    "Results of Operations",
    "Liquidity and Capital Resources",
    "Contractual Obligations",
    "Summary of Critical Accounting Estimates",
]
REWRITTEN = {
    "R28": "Walmart sources of liquidity: cash flows from operating activities, "
           "short-term borrowings, working capital deficit and committed lines of credit",
    "N29": "Walmart material cash requirements for inventory purchases and "
           "capital expenditures",
}


def rank_keys(query_vec: list[float], keys: list[int], vectors: np.ndarray) -> list[int]:
    """Chunk indexes ordered by cosine similarity to the query, nearest first."""
    q = np.asarray(query_vec)
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    return [keys[i] for i in np.argsort(-sims)]


def capped_recall(order: list[int], relevant: set[int], k: int = 5) -> float:
    """Relevant in the top k / min(k, R)."""
    return sum(i in relevant for i in order[:k]) / min(k, len(relevant))


def main() -> int:
    """Run the four conditions and print ranks."""
    try:
        gold = {q.id: q for q in load_gold().questions}
        got = get_store()._collection.get(where=build_filter(TICKER, {SECTION}),
                                          include=["documents", "metadatas"])
    except (GoldError, VectorStoreError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    chunks = sorted(((int(m["chunk_index"]), doc) for doc, m in
                     zip(got["documents"], got["metadatas"])), key=lambda c: c[0])
    keys = [i for i, _ in chunks]

    heading = HEADINGS[0]
    prefixed: list[str] = []
    assigned: dict[int, str] = {}
    for index, text in chunks:
        # The heading in force at the START of the chunk is the one carried over.
        assigned[index] = heading
        prefixed.append(f"{COMPANY} · {SECTION} · {heading}\n\n{text}")
        positions = [(text.rfind(h), h) for h in HEADINGS if text.rfind(h) >= 0]
        if positions:
            heading = max(positions)[1]

    try:
        base_vecs = np.asarray(embed_texts([t for _, t in chunks]))
        pref_vecs = np.asarray(embed_texts(prefixed))
    except VectorStoreError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"{len(chunks)} {TICKER} {SECTION} chunks; heading carried into each:")
    print("  " + ", ".join(f"{i}:{assigned[i][:12]}" for i in keys))

    for qid in ("R28", "N29"):
        gq = gold[qid]
        relevant = {int(k.split("|")[2]) for k in gq.relevant(lenient=False)}
        try:
            q_orig = embed_query(gq.question)
            q_new = embed_query(REWRITTEN[qid])
        except VectorStoreError as exc:
            print(f"FAILED: {exc}")
            return 1
        print(f"\n=== {qid}: {gq.question}")
        print(f"    relevant chunks: {sorted(relevant)}")
        print(f"    rewritten query (H2): {REWRITTEN[qid]}")
        print(f"    {'condition':<26}{'recall@5':>9}   ranks of relevant chunks")
        for name, qv, vecs in (("baseline", q_orig, base_vecs),
                               ("H1 heading prefix", q_orig, pref_vecs),
                               ("H2 rewritten query", q_new, base_vecs),
                               ("H1 + H2", q_new, pref_vecs)):
            order = rank_keys(qv, keys, vecs)
            ranks = {i: order.index(i) + 1 for i in relevant}
            print(f"    {name:<26}{capped_recall(order, relevant):>8.0%}   "
                  + ", ".join(f"#{i}→{ranks[i]}" for i in sorted(relevant)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
