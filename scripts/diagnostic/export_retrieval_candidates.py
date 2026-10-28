"""Export the candidate chunk pool for labelling retrieval gold.

Diagnostic (print() allowed). The first step of the retrieval evaluation: it
produces the pool that is labelled, frozen by hash, then scored by
run_retrieval_eval.

For each narrative-set question (plus N20, an audit case), the eligible chunks
are those the system may search: the strict label's intent and company, after
section aliases, and after the section fallback when the intent's own sections are
absent. Every eligible chunk is ranked against the question so recall@k can be
computed for any k later.

Candidates to label are POOLED from two independent sources, so the gold set
is not built only from what the retriever already finds:
  1. the retriever's top 10;
  2. keyword hits — regexes written from the question's meaning, never from
     retrieved text — capped at 25 per question by keyword hit count.
A relevant chunk neither source surfaces stays unlabelled, so measured recall
is an upper bound. That limit is reported with the metric.

Questions marked BROAD ("what are the main risks") make most of their section
relevant; recall is meaningless there, so only the top 10 are exported and
precision@k is reported instead.

Embedding calls only (one per question); no LLM. Writes
settings.PROCESSED_DATA_PATH / retrieval_candidates.json (gitignored: it holds
chunk text). Send that file for labelling.

Usage:
    python -m scripts.diagnostic.export_retrieval_candidates
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.eval.query_eval import EvalSetError, load_eval_set
from src.query.intents import fallback_for, sections_for
from src.query.schemas import Intent
from src.storage.vector_store import (
    RetrievalStatus,
    VectorStoreError,
    build_filter,
    embed_query,
    get_store,
    query_by_vector,
)
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

AUDIT_IDS = {"N20"}
TOP_K_POOL = 10
KEYWORD_CAP = 25
OUTPUT_NAME = "retrieval_candidates.json"

# Written from what each question asks, before looking at any retrieved text.
# None marks a BROAD question: precision@k only.
KEYWORDS: dict[str, list[str] | None] = {
    "R14": None,
    "R16": [r"\bsegments?\b", r"north america", r"\binternational\b", r"\bAWS\b",
            r"amazon web services"],
    "R17": None,
    "R18": [r"\bstrateg", r"\bgrowth\b", r"omni-?channel", r"\binvest(ing|ment)s? in\b"],
    "R19": [r"\bcontent (assets|obligations|spending|investment|amortization)",
            r"original (programming|content)", r"\blicens(ed|ing) content"],
    "R21": [r"\bcompet"],
    "R22": None,
    "R23": None,
    "R24": [r"\bregulat", r"\bFDA\b", r"\bpricing\b", r"\bgovernment\b", r"\breimburse"],
    "R25": [r"legal proceedings", r"\blitigation\b", r"\blawsuits?\b", r"\binvestigations?\b",
            r"\bFTC\b", r"class action"],
    "R26": [r"revenue (increased|grew|growth)", r"driven by", r"\bincreased? \$?\d"],
    "R28": [r"\bliquidity\b", r"capital resources", r"commercial paper", r"credit facilit",
            r"cash flows? (from|provided)"],
    "R30": [r"interest rate", r"foreign (currency|exchange)", r"market risk"],
    "R31": [r"interest rate"],
    "N16": [r"\bcustomers?\b", r"\bmarkets?\b", r"\benterprises?\b", r"creative professionals",
            r"small and medium"],
    "N19": [r"\bstrateg", r"\bgrowth\b", r"\bacquisitions?\b", r"\bexpand"],
    "N20": [r"IDM 2\.0", r"\bstrateg", r"\bfoundry\b", r"\btransformation\b", r"\bturnaround\b"],
    "N21": [r"\bstrateg", r"\bstreaming\b", r"\bmembers(hip)?\b"],
    "N22": [r"\bcompet", r"\bregulat", r"\bprivacy\b", r"\bantitrust\b"],
    "N25": None,
    "N27": [r"\badvertis"],
    "N28": [r"revenues? (increased|grew|decreased)", r"average (monthly )?revenue",
            r"paid memberships?", r"\bpric(e|ing) (increase|change)"],
    "N29": [r"\binventor", r"capital expenditure", r"capital requirement"],
    "N30": [r"\bcurrenc", r"foreign exchange", r"exchange rates?"],
}


def eligible_sections(intent: Intent, ticker: str, held: set[str]) -> tuple[list[str], list[str]]:
    """The sections retrieval searches for this company, applying the section fallback.

    Args:
        intent: The labelled intent.
        ticker: The labelled company.
        held: Section labels this company has in the store.

    Returns:
        (sections searched, sections the fallback replaced — empty if none).
    """
    sections = sections_for(intent, ticker)
    if sections & held:
        return sorted(sections), []
    fallback = fallback_for(intent, ticker)
    if fallback is not None and fallback.sections & held:
        return sorted(fallback.sections), sorted(sections)
    return sorted(sections), []


def chunk_key(ticker: str, section: str, index: int) -> str:
    """Stable identity of a chunk: TICKER|section|index."""
    return f"{ticker}|{section}|{index}"


def main() -> int:
    """Build and write the candidate pools.

    Returns:
        Process exit code: 0 on success, 1 on any setup or store failure.
    """
    try:
        eval_set = load_eval_set()
        store = get_store()
    except (EvalSetError, VectorStoreError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    questions = [q for q in eval_set.questions if q.narrative_set or q.id in AUDIT_IDS]
    out: list[dict] = []
    print(f"{'id':<5}{'intent':<28}{'tkr':<6}{'pool':>6}{'kw hits':>9}{'labelled':>10}  sections")

    for question in questions:
        task = question.strict.tasks[0]
        intent, ticker = Intent(task.intent), task.company
        try:
            held_meta = store._collection.get(where={"source_ticker": ticker},
                                              include=["metadatas"])["metadatas"]
        except Exception as exc:  # noqa: BLE001 — diagnostic batch item
            print(f"{question.id}: store read failed: {exc}")
            return 1
        held = {m["source_section"] for m in held_meta}
        sections, replaced = eligible_sections(intent, ticker, held)

        got = store._collection.get(where=build_filter(ticker, sections),
                                    include=["documents", "metadatas"])
        texts = {
            chunk_key(m["source_ticker"], m["source_section"], int(m["chunk_index"])): doc
            for doc, m in zip(got["documents"], got["metadatas"])
        }
        try:
            vector = embed_query(question.question)
        except VectorStoreError as exc:
            print(f"{question.id}: embedding failed: {exc}")
            return 1
        ranked = query_by_vector(vector, k=max(len(texts), 1), ticker=ticker, sections=sections)
        if ranked.status is not RetrievalStatus.OK:
            print(f"{question.id}: retrieval {ranked.status.value}")
            continue
        ranking = [chunk_key(c.ticker, c.section, c.chunk_index) for c in ranked.chunks]
        distance = {chunk_key(c.ticker, c.section, c.chunk_index): round(c.distance, 4)
                    for c in ranked.chunks}
        rank = {key: i + 1 for i, key in enumerate(ranking)}

        patterns = KEYWORDS.get(question.id)
        keyword_hits: dict[str, int] = {}
        if patterns:
            for key, text in texts.items():
                hits = sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)
                if hits:
                    keyword_hits[key] = hits
        kw_keep = sorted(keyword_hits, key=lambda k: (-keyword_hits[k], rank.get(k, 10**6)))
        kw_keep = kw_keep[:KEYWORD_CAP]

        candidates = list(dict.fromkeys(ranking[:TOP_K_POOL] + kw_keep))
        candidates.sort(key=lambda k: rank.get(k, 10**6))
        out.append({
            "id": question.id,
            "split": question.split,
            "question": question.question,
            "intent": intent.value,
            "ticker": ticker,
            "sections": sections,
            "fallback_replaced": replaced,
            "broad": patterns is None,
            "pool_size": len(texts),
            "ranking": ranking,
            "keyword_patterns": patterns,
            "keyword_hits_total": len(keyword_hits),
            "candidates": [
                {"key": key, "rank": rank.get(key), "distance": distance.get(key),
                 "sources": [s for s, ok in (("top10", rank.get(key, 99) <= TOP_K_POOL),
                                             ("keyword", key in kw_keep)) if ok],
                 "keyword_hits": keyword_hits.get(key, 0),
                 "text": texts[key]}
                for key in candidates
            ],
        })
        print(f"{question.id:<5}{intent.value:<28}{ticker:<6}{len(texts):>6}"
              f"{len(keyword_hits):>9}{len(candidates):>10}  {sections}"
              f"{'  (fallback for ' + str(replaced) + ')' if replaced else ''}")

    path = Path(settings.PROCESSED_DATA_PATH) / OUTPUT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"top_k_pool": TOP_K_POOL, "keyword_cap": KEYWORD_CAP,
                                "questions": out}, indent=1), encoding="utf-8")
    total = sum(len(q["candidates"]) for q in out)
    print(f"\n{len(out)} questions, {total} candidates to label → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())