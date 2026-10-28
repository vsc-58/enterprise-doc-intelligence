"""Retrieval evaluation: rank quality of the narrative path against chunk-level gold labels.

For every question in the retrieval gold, ranks the company's eligible chunks
with the production retriever (retrieve_for_company: same sections, aliases and
section fallback the system uses) and scores the ranking at settings.RETRIEVAL_EVAL_KS:
hit, capped recall, precision, first relevant rank. Strict counts rel chunks;
lenient adds border.

Two queries per question, because production does not embed the user's words:
  original — the question as asked;
  planner  — the self-contained task question the planner wrote, one per run,
             read from a routing results file (--routing), searched under the
             planner's own intent when the labels accept it (R24 routes to
             risk_factors, which searches different sections). A run whose plan
             has no accepted task for the gold company is counted as unmeasured.
The planner score is the primary one: it is what the system actually retrieves
with. The original score isolates the retriever from the planner's rewording.

Chunks that enter any top-k but were never judged are written, with text, to
settings.PROCESSED_DATA_PATH / RETRIEVAL_TO_LABEL_FILE (gitignored) for the next
labelling round. While any remain, precision is a lower bound and the results
are marked provisional. Broad questions have no pool at all, so their first run
exists to produce that file.

Embedding calls only (one per distinct query string, about 100); no LLM.
Writes settings.EVAL_DATA_PATH / RETRIEVAL_RESULTS_FILE plus a markdown summary.

Run:  python -m scripts.run_retrieval_eval --routing data/eval/routing_results_d4bc8e36.json
      python -m scripts.run_retrieval_eval --ids R28,N22        (subset)
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.query_eval import EvalSet, EvalSetError, load_eval_set, wilson_interval
from src.eval.retrieval_eval import (
    GoldError,
    GoldQuestion,
    RankingScore,
    PlannerQuery,
    load_gold,
    mean_or_none,
    planner_queries,
    score_ranking,
    unjudged_in_top,
)
from src.query.resolve import CorpusIndex, load_index
from src.query.schemas import Intent
from src.rag.retriever import CompanyContext, CompanyRef, retrieve_for_company
from src.storage.vector_store import RetrievalStatus
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_key(ticker: str, section: str, index: int) -> str:
    """Stable identity of a chunk: TICKER|section|index (as in the gold)."""
    return f"{ticker}|{section}|{index}"


def company_ref(ticker: str, index: CorpusIndex) -> CompanyRef:
    """The retriever's company identity for a gold ticker.

    Args:
        ticker: Gold company.
        index: Corpus index.

    Returns:
        The CompanyRef.

    Raises:
        KeyError: if the corpus does not hold the ticker — the gold and the
            corpus disagree, and scoring it would be meaningless.
    """
    for entry in index.entries:
        if entry.ticker == ticker:
            return CompanyRef(ticker=entry.ticker, company_name=entry.company_name,
                              fiscal_year=entry.filing_year)
    raise KeyError(f"ticker {ticker} is not in the corpus index")


def accepted_intents(eval_set: EvalSet, question_id: str, ticker: str, gold_intent: str) -> list[str]:
    """Intents the frozen labels accept for this question's company, strict first.

    Args:
        eval_set: The frozen eval set.
        question_id: Question id.
        ticker: Gold company.
        gold_intent: The gold intent, always accepted and always first.

    Returns:
        Accepted intents in label order, without duplicates.
    """
    intents = [gold_intent]
    for question in eval_set.questions:
        if question.id != question_id:
            continue
        for alternative in question.alternatives:
            for task in alternative.tasks:
                if task.company == ticker and task.intent not in intents:
                    intents.append(task.intent)
    return intents


def _score_both(ranking: list[str], gold: GoldQuestion) -> dict[str, RankingScore]:
    """Strict and lenient scores for one ranking.

    Args:
        ranking: Chunk keys, nearest first.
        gold: The question's gold.

    Returns:
        {"strict": ..., "lenient": ...}.
    """
    return {
        mode: score_ranking(ranking, gold.relevant(lenient=mode == "lenient"), gold.judged(),
                            settings.RETRIEVAL_EVAL_KS)
        for mode in ("strict", "lenient")
    }


def _variant_metrics(variants: list[dict[str, Any]], source: str, mode: str) -> dict[str, Any]:
    """Average one query source's scores for one question (planner: over runs).

    Args:
        variants: The question's scored variants.
        source: "original" or "planner".
        mode: "strict" or "lenient".

    Returns:
        Per-k hit, recall and precision means, mean first relevant rank, and the
        number of measured variants. Values are None when nothing was measured.
    """
    scored = [v for v in variants if v["source"] == source and v.get(mode)]
    out: dict[str, Any] = {"measured": len(scored)}
    for k in settings.RETRIEVAL_EVAL_KS:
        cut = [RankingScore.model_validate(v[mode]).at(k) for v in scored]
        out[f"hit@{k}"] = mean_or_none([None if c.hit is None else float(c.hit) for c in cut])
        out[f"recall@{k}"] = mean_or_none([c.recall for c in cut])
        out[f"precision@{k}"] = mean_or_none([c.precision for c in cut])
    ranks = [v[mode]["first_relevant_rank"] for v in scored]
    out["first_rank"] = mean_or_none([float(r) for r in ranks if r is not None])
    return out


def _group(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Aggregate a group of questions for both query sources.

    Each question counts once: planner runs are averaged within the question
    first, so a question is not weighted by how many runs it was measurable in.

    Args:
        records: Per-question records.
        mode: "strict" or "lenient".

    Returns:
        {"n": ..., "original": {...}, "planner": {...}} with 95% Wilson intervals
        on hit@k (n = questions).
    """
    out: dict[str, Any] = {"n": len(records)}
    for source in ("original", "planner"):
        per_q = [r["summary"][mode][source] for r in records]
        per_q = [m for m in per_q if m["measured"]]
        block: dict[str, Any] = {"n_measured": len(per_q)}
        for metric in [f"{m}@{k}" for k in settings.RETRIEVAL_EVAL_KS
                       for m in ("hit", "recall", "precision")] + ["first_rank"]:
            block[metric] = mean_or_none([m[metric] for m in per_q])
        for k in settings.RETRIEVAL_EVAL_KS:
            hits = [m[f"hit@{k}"] for m in per_q if m[f"hit@{k}"] is not None]
            block[f"hit@{k}_ci95"] = wilson_interval(sum(hits), len(hits)) if hits else None
        out[source] = block
    return out


def _paired(records: list[dict[str, Any]], k: int) -> dict[str, list[str]]:
    """Which questions the planner's rewording helped or hurt, at recall@k (strict).

    Args:
        records: Per-question records with both sources measured.
        k: Cut-off.

    Returns:
        {"better": ids, "worse": ids, "same": ids}.
    """
    out: dict[str, list[str]] = {"better": [], "worse": [], "same": []}
    for r in records:
        orig = r["summary"]["strict"]["original"][f"recall@{k}"]
        plan = r["summary"]["strict"]["planner"][f"recall@{k}"]
        if orig is None or plan is None:
            continue
        bucket = "same" if abs(plan - orig) < 1e-9 else ("better" if plan > orig else "worse")
        out[bucket].append(r["id"])
    return out


def _fmt(value: float | None, pct: bool = True) -> str:
    """Render a metric for the markdown summary."""
    if value is None:
        return "–"
    return f"{value:.0%}" if pct else f"{value:.1f}"


def main(routing_path: Path | None, ids: set[str] | None) -> int:
    """Run the retrieval eval and write results.

    Args:
        routing_path: Routing results file for planner queries; None scores the
            original questions only.
        ids: Optional subset of question ids.

    Returns:
        Process exit code: 0 on completion, 1 if setup failed.
    """
    try:
        gold = load_gold()
        eval_set = load_eval_set()
        index = load_index()
        routing: dict[str, dict[str, Any]] = {}
        planner_version: str | None = None
        if routing_path is not None:
            raw = json.loads(routing_path.read_text(encoding="utf-8"))
            routing = {q["id"]: q for q in raw["questions"]}
            planner_version = raw.get("planner_version")
    except (GoldError, EvalSetError, RuntimeError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.error("retrieval_eval_setup_failed", error_type=type(exc).__name__, error=str(exc))
        return 1

    if gold.status != "verified":
        logger.warning("retrieval_gold_not_verified", status=gold.status)

    questions = [q for q in gold.questions if ids is None or q.id in ids]
    max_k = max(settings.RETRIEVAL_EVAL_KS)
    cache: dict[tuple[str, str], CompanyContext] = {}
    to_label: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for gq in questions:
        try:
            company = company_ref(gq.ticker, index)
        except KeyError as exc:
            logger.error("retrieval_eval_question_skipped", id=gq.id, error=str(exc))
            continue
        queries: list[tuple[str, int | None, PlannerQuery | None]] = [
            ("original", None, PlannerQuery(intent=gq.intent, question=gq.question))
        ]
        if gq.id in routing:
            accepted = accepted_intents(eval_set, gq.id, gq.ticker, gq.intent)
            queries += [("planner", run, query) for run, query in
                        enumerate(planner_queries(routing[gq.id], accepted, gq.ticker), start=1)]

        variants: list[dict[str, Any]] = []
        texts: dict[str, str] = {}
        original: CompanyContext | None = None
        for source, run, query in queries:
            variant: dict[str, Any] = {"source": source, "run": run,
                                       "query": query.question if query else None,
                                       "intent": query.intent if query else None}
            if query is None:
                variant["status"] = "no_matching_task"
                variants.append(variant)
                continue
            cache_key = (query.intent, query.question)
            if cache_key not in cache:
                cache[cache_key] = retrieve_for_company(query.question, Intent(query.intent), company,
                                                        k=settings.RETRIEVAL_EVAL_DEPTH)
            context = cache[cache_key]
            original = original or context
            variant["status"] = context.status.value
            variant["sections_searched"] = sorted(context.sections_searched)
            if context.status is not RetrievalStatus.OK:
                variants.append(variant)
                continue
            ranking = [chunk_key(b.chunk.ticker, b.chunk.section, b.chunk.chunk_index)
                       for b in context.blocks]
            texts.update({chunk_key(b.chunk.ticker, b.chunk.section, b.chunk.chunk_index): b.chunk.text
                          for b in context.blocks})
            variant["top"] = ranking[:max_k]
            variant["unjudged"] = unjudged_in_top(ranking, gq.judged(), max_k)
            variant.update({mode: score.model_dump() for mode, score in _score_both(ranking, gq).items()})
            variants.append(variant)
            for key in variant["unjudged"]:
                to_label.setdefault(f"{gq.id}::{key}", {
                    "id": gq.id, "question": gq.question, "key": key, "reason": "unjudged in top-k",
                    "best_rank": ranking.index(key) + 1, "text": texts[key]})
                to_label[f"{gq.id}::{key}"]["best_rank"] = min(
                    to_label[f"{gq.id}::{key}"]["best_rank"], ranking.index(key) + 1)

        for key in gq.recheck:
            if key in texts:
                to_label.setdefault(f"{gq.id}::{key}", {
                    "id": gq.id, "question": gq.question, "key": key,
                    "reason": "judged on partial text", "best_rank": None, "text": texts[key]})

        record = {
            "id": gq.id, "split": gq.split, "question": gq.question, "ticker": gq.ticker,
            "intent": gq.intent, "broad": gq.broad, "audit": gq.audit,
            "relevant": {"strict": len(gq.relevant(False)), "lenient": len(gq.relevant(True))},
            "judged": len(gq.judged()),
            "sections_searched": sorted(original.sections_searched) if original else [],
            "fallback_from": sorted(original.fallback_from) if original else [],
            "planner_intents": sorted({v["intent"] for v in variants
                                       if v["source"] == "planner" and v["intent"]}),
            "notes": gq.notes,
            "planner_unmeasured_runs": sum(v["status"] == "no_matching_task" for v in variants),
            "variants": variants,
            "summary": {mode: {src: _variant_metrics(variants, src, mode)
                               for src in ("original", "planner")}
                        for mode in ("strict", "lenient")},
        }
        records.append(record)
        logger.info("retrieval_eval_question_done", id=gq.id, variants=len(variants),
                    unjudged=sum(len(v.get("unjudged", [])) for v in variants))

    specific = [r for r in records if not r["broad"] and not r["audit"] and r["relevant"]["strict"]]
    broad = [r for r in records if r["broad"]]
    groups = {
        split: [r for r in specific if split == "all" or r["split"] == split]
        for split in ("held", "dev", "all")
    }
    summary = {
        "specific": {mode: {split: _group(rs, mode) for split, rs in groups.items()}
                     for mode in ("strict", "lenient")},
        "broad": {mode: _group(broad, mode) for mode in ("strict", "lenient")},
        "paired_recall@5": _paired(specific, 5) if 5 in settings.RETRIEVAL_EVAL_KS else None,
    }
    audits = [{"id": r["id"], "relevant": r["relevant"]["lenient"], "judged": r["judged"],
               "statuses": sorted({v["status"] for v in r["variants"]})}
              for r in records if r["audit"]]
    provisional = bool(to_label)

    out_dir = Path(settings.EVAL_DATA_PATH)
    results_path = out_dir / settings.RETRIEVAL_RESULTS_FILE
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold_status": gold.status,
        "provisional": provisional,
        "routing_file": str(routing_path) if routing_path else None,
        "planner_version": planner_version,
        "embedding_model": settings.EMBEDDING_MODEL,
        "ks": settings.RETRIEVAL_EVAL_KS,
        "metric_notes": [
            "recall@k is capped: relevant in top k / min(k, R).",
            "Recall is an upper bound: relevant chunks outside the labelled pool are unknown.",
            "Precision counts unjudged chunks as not relevant: a lower bound while any remain.",
            "Planner scores average a question's runs first; each question counts once.",
            "Wilson 95% intervals on hit@k use n = questions.",
        ],
        "summary": summary,
        "audits": audits,
        "unjudged_total": sum(1 for v in to_label.values() if v["reason"] == "unjudged in top-k"),
        "questions": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    label_path = Path(settings.PROCESSED_DATA_PATH) / settings.RETRIEVAL_TO_LABEL_FILE
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(sorted(to_label.values(), key=lambda v: (v["id"], v["key"])),
                                     indent=1), encoding="utf-8")

    _write_markdown(results_path.with_suffix(".md"), results, records)
    logger.info("retrieval_eval_written", results=str(results_path), to_label=str(label_path),
                to_label_count=len(to_label), provisional=provisional)
    return 0


def _write_markdown(path: Path, results: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Write the human-readable summary.

    Args:
        path: Output path.
        results: The results document.
        records: Per-question records.
    """
    ks = results["ks"]
    lines = [
        "# Retrieval eval",
        "",
        f"gold `{results['gold_status']}` · planner `{(results['planner_version'] or '-')[:8]}` · "
        f"embeddings `{results['embedding_model']}` · k = {ks}",
        "",
    ]
    if results["provisional"]:
        lines += [f"**PROVISIONAL** — {results['unjudged_total']} unjudged chunks entered a top "
                  f"{max(ks)}; label them and rerun.", ""]
    lines += [f"- {note}" for note in results["metric_notes"]] + [""]

    for mode in ("strict", "lenient"):
        lines += [f"## Specific questions — {mode}", "",
                  "| split | source | n | " + " | ".join(
                      f"hit@{k} | recall@{k} | prec@{k}" for k in ks) + " | first rank |",
                  "|---|---|---|" + "---|---|---|" * len(ks) + "---|"]
        for split, group in results["summary"]["specific"][mode].items():
            for source in ("original", "planner"):
                g = group[source]
                cells = []
                for k in ks:
                    ci = g.get(f"hit@{k}_ci95")
                    ci_txt = f" [{ci[0]:.0%}–{ci[1]:.0%}]" if ci else ""
                    cells += [_fmt(g[f"hit@{k}"]) + ci_txt, _fmt(g[f"recall@{k}"]),
                              _fmt(g[f"precision@{k}"])]
                lines.append(f"| {split} | {source} | {g['n_measured']} | " + " | ".join(cells)
                             + f" | {_fmt(g['first_rank'], pct=False)} |")
        lines.append("")

    lines += ["## Broad questions — precision", "", "| mode | source | n | " +
              " | ".join(f"prec@{k}" for k in ks) + " |", "|---|---|---|" + "---|" * len(ks)]
    for mode in ("strict", "lenient"):
        for source in ("original", "planner"):
            g = results["summary"]["broad"][mode][source]
            lines.append(f"| {mode} | {source} | {g['n_measured']} | "
                         + " | ".join(_fmt(g[f"precision@{k}"]) for k in ks) + " |")
    lines.append("")

    paired = results["summary"].get("paired_recall@5")
    if paired:
        lines += ["## Planner rewording vs original (strict recall@5)", "",
                  f"- better: {', '.join(paired['better']) or 'none'}",
                  f"- worse: {', '.join(paired['worse']) or 'none'}",
                  f"- same: {len(paired['same'])} questions", ""]

    for audit in results["audits"]:
        verdict = ("no relevant chunk exists in the corpus — a refusal is the correct answer"
                   if audit["relevant"] == 0 else f"{audit['relevant']} relevant chunks exist")
        lines.append(f"**Audit {audit['id']}**: {verdict} (judged {audit['judged']}, "
                     f"retrieval status {', '.join(audit['statuses'])}).")
    lines += ["", "## Per question (strict)", "",
              "| ID | split | R | source | " + " | ".join(f"recall@{k}" for k in ks)
              + f" | prec@{ks[0]} | first rank | unjudged | note |",
              "|---|---|---|---|" + "---|" * len(ks) + "---|---|---|---|"]
    for r in records:
        for source in ("original", "planner"):
            m = r["summary"]["strict"][source]
            if not m["measured"]:
                continue
            unjudged = sum(len(v.get("unjudged", [])) for v in r["variants"] if v["source"] == source)
            note = "broad" if r["broad"] else ("audit" if r["audit"] else "")
            if r["fallback_from"]:
                note = (note + " fallback").strip()
            other = [i for i in r["planner_intents"] if i != r["intent"]]
            if source == "planner" and other:
                note = (note + f" planner intent {'/'.join(other)}").strip()
            if source == "planner" and r["planner_unmeasured_runs"]:
                note = (note + f" {r['planner_unmeasured_runs']} run(s) unmeasured").strip()
            lines.append(
                f"| {r['id']} | {r['split']} | {r['relevant']['strict']} | {source} | "
                + " | ".join(_fmt(m[f"recall@{k}"]) for k in ks)
                + f" | {_fmt(m[f'precision@{ks[0]}'])} | {_fmt(m['first_rank'], pct=False)} "
                f"| {unjudged} | {note} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--routing", type=Path, default=None,
                        help="Routing results JSON whose plans supply the planner queries.")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated question ids.")
    args = parser.parse_args()
    subset = {i.strip() for i in args.ids.split(",")} if args.ids else None
    raise SystemExit(main(args.routing, subset))
