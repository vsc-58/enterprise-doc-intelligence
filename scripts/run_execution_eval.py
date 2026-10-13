"""
scripts/run_execution_eval.py
Phase 5C steps 5–6: what each question actually produces, in two modes.

  oracle      the label's strict plan, executed directly — the downstream
              system (resolver, SQL path, retriever, fallback, chain, guard)
              with routing assumed perfect. Executable questions only.
  end_to_end  the planner's own plan, executed — what a user receives.

Every task is reduced to an outcome class the code already emits (resolver
failure, NarrativeRefusal, D29 quality verdict, D44 fallback) and compared with
the label. A question is correct when every task matches; end to end it must
also have routed correctly. A miss in oracle mode is a downstream failure no
routing fix would cure; a miss only end to end is a routing cost.

Also recorded per run, both modes: the figure-guard counters (masked,
redacted), invalid citations, fallback use, and every claim with its cited
sources — the input for the attribution judge (D43). Any redaction is a
masking miss to investigate.

HELD-OUT IS REPORT-ONLY, as for routing.

Writes settings.EVAL_DATA_PATH / execution_results_<planner8>_<chain8>.json and
a markdown summary beside it.

Run:  python -m scripts.run_execution_eval
      python -m scripts.run_execution_eval --runs 1 --ids R01,R31,H02 --modes oracle

Cost: roughly $1 for the full set, both modes, three runs. Corpus-wide surveys
are most of it (one verdict call per company).

Dependencies: src.eval.execution_eval, src.eval.query_eval, src.query.*,
src.rag.chain, src.utils.config, src.utils.logger.
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.eval.execution_eval import (
    OracleError,
    expected_by_key,
    is_executable,
    load_oracle_params,
    matched_alternative,
    oracle_plan,
    record,
)
from src.eval.query_eval import (
    EvalQuestion,
    EvalSetError,
    load_eval_set,
    observe,
    wilson_interval,
)
from src.query.executor import execute_plan
from src.query.planner import plan_query, planner_version
from src.query.resolve import CorpusIndex, load_index
from src.query.synthesizer import assemble
from src.rag.chain import chain_version
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

ORACLE = "oracle"
END_TO_END = "end_to_end"
MODES = (ORACLE, END_TO_END)


async def _run_question(
    question: EvalQuestion,
    mode: str,
    index: CorpusIndex,
    filters: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Plan (or take the gold plan), execute, assemble and record one question.

    Batch rule: any failure is logged with the question id and recorded as an
    error, never raised.

    Returns:
        A JSON-ready record for this question in this run.
    """
    async with semaphore:
        try:
            routed = True
            if mode == ORACLE:
                plan = oracle_plan(question, filters)
                companies = [t.company for t in question.strict.tasks]
                expected = [t.outcome for t in question.strict.tasks]
            else:
                plan = await asyncio.to_thread(plan_query, question.question)
                observed = observe(plan, index)
                alt = matched_alternative(observed, question)
                routed = alt is not None
                companies = [t.company for t in observed.tasks]
                expected = expected_by_key(observed, alt) if alt else [None] * len(observed.tasks)

            outcomes = await execute_plan(plan, index)
            answer = assemble(plan, outcomes)
            tasks = [
                record(outcome, company, exp)
                for outcome, company, exp in zip(outcomes, companies, expected)
            ]
        except (OracleError, Exception) as exc:  # noqa: BLE001 — batch item; logged and counted
            logger.warning(
                "execution_eval_question_failed",
                question_id=question.id, mode=mode,
                error_type=type(exc).__name__, error=str(exc),
            )
            return {"id": question.id, "error": f"{type(exc).__name__}: {exc}",
                    "routed": False, "correct": False, "tasks": []}

    task_ok = all(t.correct for t in tasks) if tasks else True
    return {
        "id": question.id,
        "plan_type": plan.plan_type.value,
        "routed": routed,
        "correct": routed and task_ok,
        "tasks": [t.model_dump() | {"correct": t.correct} for t in tasks],
        "answer": answer.text,
    }


def _guard_totals(records: list[dict]) -> dict:
    """Sum the guard and citation counters over one run's narrative tasks."""
    narrative = [t for r in records for t in r["tasks"] if t["intent"] not in
                 ("financial_metric", "financial_ranking", "financial_filter")]
    return {
        "narrative_tasks": len(narrative),
        "figures_masked": sum(t["figures_masked"] for t in narrative),
        "figures_redacted": sum(t["figures_redacted"] for t in narrative),
        "invalid_citations": sum(t["invalid_citations"] for t in narrative),
        "fallback_answers": sum(t["outcome"] == "fallback_answered" for t in narrative),
        "fallback_findings": sum(
            1 for t in narrative for f in t["findings"] if f.get("fallback_from")
        ),
        "redacted_at": [f"{r['id']}/{t['task_id']}" for r in records for t in r["tasks"]
                        if t["figures_redacted"]],
    }


def _accuracy_rows(
    label: str,
    groups: dict[str, list[str]],
    correct: dict[str, list[bool]],
    runs: int,
) -> tuple[dict, list[str]]:
    """Question-level accuracy per group, as JSON and markdown rows."""
    out, lines = {}, [f"| {label} | n | mean | 95% CI | per run | varied |", "|---|---|---|---|---|---|"]
    for name, ids in groups.items():
        if not ids:
            continue
        per_run = [sum(correct[i][r] for i in ids) for r in range(runs)]
        mean = sum(per_run) / runs
        lo, hi = wilson_interval(mean, len(ids))
        varied = sum(len(set(correct[i])) > 1 for i in ids)
        out[name] = {"n": len(ids), "per_run": per_run, "mean": mean / len(ids),
                     "ci95": [lo, hi], "varied": varied}
        lines.append(f"| {name} | {len(ids)} | {mean / len(ids):.1%} | {lo:.0%}–{hi:.0%} "
                     f"| {'/'.join(map(str, per_run))} | {varied} |")
    return out, lines


async def _run_mode(
    mode: str,
    questions: list[EvalQuestion],
    index: CorpusIndex,
    filters: dict,
    runs: int,
) -> list[list[dict]]:
    """All runs of one mode; returns records[run][question]."""
    semaphore = asyncio.Semaphore(settings.QUERY_EVAL_CONCURRENCY)
    pool = [q for q in questions if mode == END_TO_END or is_executable(q)]
    all_runs = []
    for run in range(1, runs + 1):
        records = await asyncio.gather(
            *(_run_question(q, mode, index, filters, semaphore) for q in pool)
        )
        logger.info("execution_eval_run_done", mode=mode, run=run,
                    correct=sum(r["correct"] for r in records), n=len(records))
        all_runs.append(list(records))
    return all_runs


async def _run_all(
    modes: list[str],
    questions: list[EvalQuestion],
    index: CorpusIndex,
    filters: dict,
    runs: int,
) -> dict[str, list[list[dict]]]:
    """Every mode, sequentially, inside one event loop."""
    return {mode: await _run_mode(mode, questions, index, filters, runs) for mode in modes}


def main(runs: int, ids: set[str] | None, modes: list[str]) -> int:
    """
    Run the execution eval and write results.

    Returns:
        Process exit code: 0 on completion, 1 if setup failed.
    """
    try:
        eval_set = load_eval_set()
        filters = load_oracle_params()
        index = load_index()
    except (EvalSetError, OracleError, RuntimeError) as exc:
        logger.error("execution_eval_setup_failed", error_type=type(exc).__name__, error=str(exc))
        return 1

    questions = [q for q in eval_set.questions if ids is None or q.id in ids]
    by_id = {q.id: q for q in questions}
    p_version, c_version = planner_version(), chain_version()
    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planner_version": p_version, "chain_version": c_version,
        "model": settings.OPENAI_MODEL, "eval_set_sha256": eval_set.sha256,
        "runs": runs, "subset": sorted(ids) if ids else None, "modes": {},
    }
    md = ["# Execution eval", "",
          f"planner `{p_version[:8]}` · chain `{c_version[:8]}` · {runs} runs · "
          f"eval set `{eval_set.sha256[:12]}`", "",
          "Question correct = every task's outcome matches its label (end to end: and "
          "routed correctly). Intervals use n = questions. Held-out is report-only.", ""]

    # One event loop for every mode: the chat client is cached per process and
    # its async transport must not outlive the loop it was created on.
    by_mode = asyncio.run(_run_all(modes, questions, index, filters, runs))
    for mode in modes:
        all_runs = by_mode[mode]
        qids = [r["id"] for r in all_runs[0]]
        correct = {qid: [run[i]["correct"] for run in all_runs] for i, qid in enumerate(qids)}
        splits = {s: [i for i in qids if by_id[i].split == s] for s in ("held", "dev")}
        splits["all"] = qids
        acc_json, acc_md = _accuracy_rows("split", splits, correct, runs)

        # Task-level outcome accuracy by expected class (labelled tasks only).
        by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        confusion: Counter = Counter()
        for run in all_runs:
            for rec in run:
                for t in rec["tasks"]:
                    if t["expected"] is None:
                        continue
                    by_class[t["expected"]][0] += 1
                    by_class[t["expected"]][1] += t["correct"]
                    if not t["correct"]:
                        confusion[(t["expected"], t["outcome"])] += 1
        class_md = ["| expected outcome | task-runs | matched | rate |", "|---|---|---|---|"]
        for cls, (n, ok) in sorted(by_class.items()):
            class_md.append(f"| {cls} | {n} | {ok} | {ok / n:.1%} |")

        guard = [_guard_totals(run) for run in all_runs]
        miss_md = ["| ID | split | run | routed | task outcomes (expected → got) | error |",
                   "|---|---|---|---|---|---|"]
        for r_no, run in enumerate(all_runs, start=1):
            for rec in run:
                if rec["correct"]:
                    continue
                detail = "; ".join(f"{t['intent']}:{t['company']} {t['expected']}→{t['outcome']}"
                                   for t in rec["tasks"] if not t["correct"])
                miss_md.append(f"| {rec['id']} | {by_id[rec['id']].split} | {r_no} "
                               f"| {'yes' if rec['routed'] else 'no'} | {detail} "
                               f"| {rec.get('error', '')} |")

        results["modes"][mode] = {
            "accuracy": acc_json,
            "outcome_by_expected_class": {k: {"task_runs": n, "matched": ok}
                                          for k, (n, ok) in by_class.items()},
            "confusion": {f"{e}->{g}": c for (e, g), c in confusion.items()},
            "guard_per_run": guard,
            "records": all_runs,
        }
        md += [f"## {mode}", "", *acc_md, "", *class_md, "",
               "Guard per run: " + " · ".join(
                   f"run {i}: masked {g['figures_masked']}, redacted {g['figures_redacted']}, "
                   f"invalid citations {g['invalid_citations']}, fallback answers "
                   f"{g['fallback_answers']}, fallback findings {g['fallback_findings']}"
                   for i, g in enumerate(guard, start=1)),
               "", "Redactions (masking misses): "
               + (", ".join(sorted({x for g in guard for x in g["redacted_at"]})) or "none"),
               "", "### Misses", "", *miss_md, ""]

    suffix = "_smoke" if ids else ""
    out = (Path(settings.EVAL_DATA_PATH)
           / f"{Path(settings.EXECUTION_RESULTS_FILE).stem}_{p_version[:8]}_{c_version[:8]}{suffix}.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    logger.info("execution_eval_written", results=str(out))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--runs", type=int, default=settings.QUERY_EVAL_RUNS)
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated question ids, for a smoke test.")
    parser.add_argument("--modes", type=str, default=",".join(MODES))
    args = parser.parse_args()
    subset = {i.strip() for i in args.ids.split(",")} if args.ids else None
    chosen = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    raise SystemExit(main(args.runs, subset, chosen))
