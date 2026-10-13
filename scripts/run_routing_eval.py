"""
scripts/run_routing_eval.py
Phase 5C step 3: routing accuracy of the query planner over the frozen eval set.

One planner call per question per run, settings.QUERY_EVAL_RUNS runs (D35).
Each plan is reduced to (plan type, unsupported reason, and per task: intent,
resolved company, metric, year, direction) and scored strictly against the
first label and leniently against any listed alternative. Every miss is
classified at the first level that differs and listed with what the planner
produced.

Only the planner and the resolver run: no SQL, no retrieval, no chain. Routing
is scored on its own so that a downstream failure cannot be charged to it, nor
a routing failure hidden by a lucky downstream answer.

HELD-OUT IS REPORT-ONLY. The planner prompt is not revised in response to
held-out results. Revisions are made against dev results only, and any
revision changes planner_version, which is recorded in the results so a
before/after comparison is explicit.

Writes settings.EVAL_DATA_PATH / ROUTING_RESULTS_FILE (committed evidence)
and a markdown summary beside it.

Run:  python -m scripts.run_routing_eval
      python -m scripts.run_routing_eval --runs 1 --ids R01,N10    (smoke test)

Cost: about $0.15 for the full set at three runs (~330 planner calls).

Dependencies: src.eval.query_eval, src.query.planner, src.query.resolve,
src.utils.config, src.utils.logger.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.eval.query_eval import (
    CALL_ERROR_SCORE,
    EvalQuestion,
    EvalSetError,
    GroupSummary,
    ObservedPlan,
    QuestionScore,
    load_eval_set,
    observe,
    score_question,
    summarise,
)
from src.query.planner import plan_query, planner_version
from src.query.resolve import CorpusIndex, load_index
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

FREEZE_COMMIT = "a15727e"
_RETRIES = 1


def _plan_once(question: EvalQuestion, index: CorpusIndex) -> ObservedPlan | None:
    """
    Plan one question, retrying once on a transport failure.

    Batch rule: a failure is logged with the question id and returned as None,
    never raised, so one bad call cannot stop the run.

    Args:
        question: The labelled question.
        index: Corpus index for resolving mentions.

    Returns:
        The observed plan, or None if every attempt failed.
    """
    for attempt in range(_RETRIES + 1):
        try:
            return observe(plan_query(question.question), index)
        except Exception as exc:  # noqa: BLE001 — batch item; logged and counted
            logger.warning(
                "routing_eval_call_failed",
                question_id=question.id,
                attempt=attempt + 1,
                error_type=type(exc).__name__,
                error=str(exc),
            )
    return None


def _run(questions: list[EvalQuestion], index: CorpusIndex) -> list[ObservedPlan | None]:
    """
    Plan every question once, concurrently, preserving input order.

    Args:
        questions: Questions to plan.
        index: Corpus index.

    Returns:
        One observation (or None on failure) per question, in input order.
    """
    with ThreadPoolExecutor(max_workers=settings.QUERY_EVAL_CONCURRENCY) as pool:
        return list(pool.map(lambda q: _plan_once(q, index), questions))


def _group_rows(
    label: str,
    groups: dict[str, list[EvalQuestion]],
    scores: dict[str, list[QuestionScore]],
    signatures: dict[str, list[str]],
) -> tuple[dict[str, dict], list[str]]:
    """
    Summarise named groups and render them as markdown table rows.

    Args:
        label: Column header for the group name.
        groups: Group name -> its questions.
        scores: Per-question scores across runs.
        signatures: Per-question plan signatures across runs.

    Returns:
        JSON-ready summaries keyed by group, and markdown lines.
    """
    out: dict[str, dict] = {}
    lines = [
        f"| {label} | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI "
        "| per run (strict) | plan varied | score varied |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, members in groups.items():
        if not members:
            continue
        summary: GroupSummary = summarise(members, scores, signatures)
        out[name] = summary.model_dump()
        lo, hi = summary.strict_ci95
        llo, lhi = summary.lenient_ci95
        lines.append(
            f"| {name} | {summary.n} | {summary.strict_mean:.1%} | {lo:.0%}–{hi:.0%} "
            f"| {summary.lenient_mean:.1%} | {llo:.0%}–{lhi:.0%} "
            f"| {'/'.join(map(str, summary.strict_per_run))} "
            f"| {summary.plan_varied} | {summary.score_varied} |"
        )
    return out, lines


def main(runs: int, ids: set[str] | None, output: Path) -> int:
    """
    Run the routing eval and write results.

    Args:
        runs: Number of runs.
        ids: Optional subset of question ids (smoke tests); None for all.
        output: Results JSON path.

    Returns:
        Process exit code: 0 on completion, 1 if the eval set or index could not
        be loaded.
    """
    try:
        eval_set = load_eval_set()
        index = load_index()
    except (EvalSetError, RuntimeError) as exc:
        logger.error("routing_eval_setup_failed", error_type=type(exc).__name__, error=str(exc))
        return 1

    questions = [q for q in eval_set.questions if ids is None or q.id in ids]
    version = planner_version()
    logger.info("routing_eval_started", questions=len(questions), runs=runs,
                planner_version=version)

    scores: dict[str, list[QuestionScore]] = {q.id: [] for q in questions}
    signatures: dict[str, list[str]] = {q.id: [] for q in questions}
    observed: dict[str, list[dict | None]] = {q.id: [] for q in questions}

    for run in range(1, runs + 1):
        plans = _run(questions, index)
        for question, plan in zip(questions, plans):
            if plan is None:
                scores[question.id].append(CALL_ERROR_SCORE)
                signatures[question.id].append("call_error")
                observed[question.id].append(None)
                continue
            scores[question.id].append(score_question(question, plan))
            signatures[question.id].append(plan.signature())
            observed[question.id].append(plan.model_dump())
        logger.info(
            "routing_eval_run_done", run=run,
            strict=sum(scores[q.id][-1].strict for q in questions),
            lenient=sum(scores[q.id][-1].lenient for q in questions),
            n=len(questions),
        )

    splits = {s: [q for q in questions if q.split == s] for s in ("held", "dev")}
    splits["all"] = questions
    strata = sorted({q.stratum for q in questions})
    held_strata = {s: [q for q in splits["held"] if q.stratum == s] for s in strata}
    dev_strata = {s: [q for q in splits["dev"] if q.stratum == s] for s in strata}

    split_json, split_md = _group_rows("split", splits, scores, signatures)
    held_json, held_md = _group_rows("held stratum", held_strata, scores, signatures)
    dev_json, dev_md = _group_rows("dev stratum", dev_strata, scores, signatures)

    per_question = []
    miss_lines = ["| ID | split | run | category | resolver? | planner produced | strict label |",
                  "|---|---|---|---|---|---|---|"]
    for question in questions:
        record = {
            "id": question.id, "split": question.split, "stratum": question.stratum,
            "question": question.question,
            "strict_label": question.strict.model_dump(),
            "runs": [
                {"score": s.model_dump(), "signature": sig, "observed": obs}
                for s, sig, obs in zip(scores[question.id], signatures[question.id],
                                       observed[question.id])
            ],
        }
        per_question.append(record)
        for run_no, (score, sig) in enumerate(
            zip(scores[question.id], signatures[question.id]), start=1
        ):
            if not score.lenient:
                miss_lines.append(
                    f"| {question.id} | {question.split} | {run_no} | {score.miss.value} "
                    f"| {'yes' if score.resolver_suspect else ''} | `{sig}` "
                    f"| `{question.strict.plan_type}` |"
                )

    varied = [q.id for q in questions if len(set(signatures[q.id])) > 1]
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "planner_version": version,
        "model": settings.OPENAI_MODEL,
        "eval_set_sha256": eval_set.sha256,
        "freeze_commit": FREEZE_COMMIT,
        "runs": runs,
        "subset": sorted(ids) if ids else None,
        "interval_note": (
            "Wilson 95% intervals use n = number of questions; runs measure model "
            "variability on the same questions and do not narrow the interval (D35)."
        ),
        "by_split": split_json,
        "by_stratum_held": held_json,
        "by_stratum_dev": dev_json,
        "plan_varied_ids": varied,
        "questions": per_question,
    }
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_path = output.with_suffix(".md")
    summary_path.write_text(
        "\n".join(
            [
                "# Routing eval",
                "",
                f"planner_version `{version}` · model `{settings.OPENAI_MODEL}` · "
                f"{runs} runs · eval set `{eval_set.sha256[:12]}` (freeze {FREEZE_COMMIT})",
                "",
                "Held-out is report-only. Intervals: n = questions, not question×runs.",
                "",
                *split_md, "", *held_md, "", *dev_md, "",
                f"Plan varied across runs: {', '.join(varied) or 'none'}",
                "",
                "## Misses (lenient), per run",
                "",
                *miss_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("routing_eval_written", results=str(output), summary=str(summary_path),
                **{f"{k}_strict_mean": round(v["strict_mean"], 3) for k, v in split_json.items()})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--runs", type=int, default=settings.QUERY_EVAL_RUNS)
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated question ids, for a smoke test.")
    args = parser.parse_args()
    subset = {i.strip() for i in args.ids.split(",")} if args.ids else None
    name = settings.ROUTING_RESULTS_FILE if subset is None else "routing_results_smoke.json"
    raise SystemExit(main(args.runs, subset, Path(settings.EVAL_DATA_PATH) / name))
