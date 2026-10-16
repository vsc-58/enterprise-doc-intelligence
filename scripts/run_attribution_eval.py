"""
scripts/run_attribution_eval.py
Phase 5C step 7, part 2: validate the attribution judge, then score attribution (D43).

1. VALIDATE. The judge rules on every validation item settings.QUERY_EVAL_RUNS
   times: 20 calibration claims, 40 test claims and 30 seeded mismatches.
   Criteria, committed before any judge output was seen (D43, amended by D52
   before the first judge run):
     seeded catch rate >= JUDGE_MIN_SEED_CATCH overall and
       >= JUDGE_MIN_SEED_CATCH_PER_TYPE per type;
     Cohen's kappa vs the reference labels over the test claims PLUS the seeds
       (reference NOT_SUPPORTED) >= JUDGE_MIN_KAPPA. The 40 real test claims
       alone are almost all supported, and kappa is unstable at that
       prevalence: one false alarm would drop it below 0.70 (D52);
     label flips across runs <= JUDGE_MAX_FLIP_RATE.
   Calibration results are shown but do not gate: that split exists to revise
   the prompt, and the test split is report-only.
2. SCORE. The judge rules on every claim of one execution run (end_to_end,
   run 1 by default: what users received). The supported rate is the HEADLINE
   only if the judge passed; otherwise it is withheld, and the 40 reference
   labels are reported as a spot-check (D43 fallback).

Reference labels for the test split were written by Claude (a different model
family from the judge), not by a human (D52). Every judge-reference
disagreement is listed with its text so the developer can adjudicate it.

Majority over runs is the per-item label; a tie counts as NOT_SUPPORTED.
Wilson intervals use n = claims; claims cluster within questions, so the true
uncertainty is wider — stated with the number.

Writes settings.EVAL_DATA_PATH / ATTRIBUTION_RESULTS_FILE plus a markdown summary.

Run:  python -m scripts.run_attribution_eval --execution data/eval/execution_results_d4bc8e36_b1d8ee5c.json

Cost: ~1,250 judge calls at three runs (90 validation items + 324 claims), well
under $1 on gpt-6-luna.

Dependencies: src.eval.attribution, src.eval.query_eval, src.query.resolve,
src.storage.vector_store, src.utils.config, src.utils.logger, src.utils.text.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.attribution import (
    JudgedItem,
    JudgeLabel,
    LabelSheetError,
    JudgeResult,
    claims_from_execution,
    cohen_kappa,
    flipped,
    judge_claim,
    judge_version,
    majority,
    parse_label_sheet,
    split_key,
)
from src.eval.query_eval import EvalSetError, load_eval_set, wilson_interval
from src.query.resolve import load_index
from src.storage.vector_store import RetrievedChunk, VectorStoreError, get_chunks_by_key
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

LABEL_CODES = {"S": JudgeLabel.SUPPORTED, "N": JudgeLabel.NOT_SUPPORTED,
               "SUPPORTED": JudgeLabel.SUPPORTED, "NOT_SUPPORTED": JudgeLabel.NOT_SUPPORTED}


def _load_json(path: Path) -> Any:
    """Read a JSON file (errors propagate to the caller's setup handler)."""
    return json.loads(path.read_text(encoding="utf-8"))


def _judge_all(
    items: list[JudgedItem],
    chunks: dict[tuple[str, str, int], RetrievedChunk],
    names: dict[str, str],
    runs: int,
) -> dict[str, list[JudgeResult]]:
    """
    Judge every item `runs` times, concurrently.

    Args:
        items: What to judge.
        chunks: Cited chunk texts, by key.
        names: Ticker -> display name.
        runs: Repetitions per item.

    Returns:
        Item id -> one JudgeResult per run. An item whose cited chunk is missing
        from the store gets failed results, never a guess.
    """
    jobs = [(item, run) for run in range(runs) for item in items]

    def one(job: tuple[JudgedItem, int]) -> tuple[str, JudgeResult]:
        item, _ = job
        blocks = [chunks.get(split_key(k)) for k in item.sources]
        if any(b is None for b in blocks):
            return item.id, JudgeResult(label=None, error="cited chunk missing from store")
        return item.id, judge_claim(names.get(item.company, item.company), item.claim, blocks)

    results: dict[str, list[JudgeResult]] = {item.id: [] for item in items}
    with ThreadPoolExecutor(max_workers=settings.JUDGE_CONCURRENCY) as pool:
        for done, (item_id, result) in enumerate(pool.map(one, jobs), start=1):
            results[item_id].append(result)
            if done % 100 == 0:
                logger.info("judge_progress", done=done, total=len(jobs))
    return results


def _summary(results: list[JudgeResult]) -> dict[str, Any]:
    """Per-item record: majority label, runs, flip, the first run's span/detail."""
    labels = [r.label for r in results]
    first = next((r for r in results if r.label is not None), results[0])
    label = majority(labels)
    return {
        "label": label.value if label else None,
        "runs": [lab.value if lab else None for lab in labels],
        "flipped": flipped(labels),
        "failed_calls": sum(lab is None for lab in labels),
        "supporting_span": first.supporting_span,
        "span_found": first.span_found,
        "unsupported_detail": first.unsupported_detail,
        "error": first.error,
    }


def _agreement(ids: list[str], human: dict[str, JudgeLabel],
               judged: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Kappa, raw agreement and confusion between reference labels and the judge."""
    pairs = [(human[i], JudgeLabel(judged[i]["label"])) for i in ids if judged[i]["label"]]
    confusion = Counter(f"reference {h.value} / judge {j.value}" for h, j in pairs)
    return {
        "n": len(pairs),
        "kappa": cohen_kappa([h for h, _ in pairs], [j for _, j in pairs]),
        "raw_agreement": sum(h is j for h, j in pairs) / len(pairs) if pairs else None,
        "human_supported_rate": sum(h is JudgeLabel.SUPPORTED for h, _ in pairs) / len(pairs) if pairs else None,
        "confusion": dict(confusion),
        "disagreements": [i for i in ids if judged[i]["label"]
                          and JudgeLabel(judged[i]["label"]) is not human[i]],
    }


def main(execution_path: Path, runs: int, run_number: int) -> int:
    """
    Validate the judge and score one execution run.

    Returns:
        Process exit code: 0 on completion (pass or fail), 1 if setup failed.
    """
    eval_dir = Path(settings.EVAL_DATA_PATH)
    try:
        spec = _load_json(eval_dir / settings.ATTRIBUTION_SET_FILE)
        seed_sources = _load_json(eval_dir / settings.ATTRIBUTION_SEEDS_FILE)["sources"]
        labels_doc = _load_json(eval_dir / settings.ATTRIBUTION_LABELS_FILE)
        execution = _load_json(execution_path)
        splits = {q.id: q.split for q in load_eval_set().questions}
        index = load_index()
        claims = claims_from_execution(execution, "end_to_end", run_number)
    except (OSError, json.JSONDecodeError, KeyError, IndexError, EvalSetError, RuntimeError) as exc:
        logger.error("attribution_eval_setup_failed", error_type=type(exc).__name__, error=str(exc))
        return 1

    names = {e.ticker: display_name(e.company_name, e.ticker) for e in index.entries}
    if "test" not in labels_doc:
        # First run after blind labelling: read the sheet, and record the labels
        # in the committed labels file so the evidence does not depend on a
        # gitignored file.
        test_ids = [i["id"] for i in spec["items"] if i["split"] == "test"]
        sheet = Path(settings.PROCESSED_DATA_PATH) / settings.ATTRIBUTION_SHEET_FILE
        try:
            labels_doc["test"] = parse_label_sheet(sheet.read_text(encoding="utf-8"), test_ids)
        except (OSError, LabelSheetError) as exc:
            logger.error("attribution_label_sheet_invalid", error=str(exc))
            return 1
        (eval_dir / settings.ATTRIBUTION_LABELS_FILE).write_text(
            json.dumps(labels_doc, indent=1) + "\n", encoding="utf-8")
        logger.info("attribution_test_labels_recorded", n=len(labels_doc["test"]))
    human = {split: {i: LABEL_CODES[v.strip().upper()] for i, v in labels_doc[split].items()}
             for split in ("test", "calib")}
    excluded: dict[str, str] = labels_doc.get("seed_exclusions", {})

    validation = [JudgedItem(id=i["id"], company=i["company"], claim=i["claim"], sources=i["sources"],
                             question_id=i["question_id"], kind=i["kind"]) for i in spec["items"]]
    validation += [JudgedItem(id=s["id"], company=s["company"], claim=s["claim"],
                              sources=seed_sources[s["id"]]) for s in spec["seeds"]]
    seed_type = {s["id"]: s["type"] for s in spec["seeds"]}

    keys = {split_key(k) for item in validation + claims for k in item.sources}
    try:
        chunks = get_chunks_by_key(keys)
    except VectorStoreError as exc:
        logger.error("attribution_eval_fetch_failed", error=str(exc))
        return 1

    version = judge_version()
    logger.info("attribution_eval_started", judge_version=version, validation=len(validation),
                claims=len(claims), runs=runs)
    judged_val = {k: _summary(v) for k, v in _judge_all(validation, chunks, names, runs).items()}
    for item in validation:
        judged_val[item.id]["claim"] = item.claim

    # --- Validation ----------------------------------------------------------
    test_ids = [i["id"] for i in spec["items"] if i["split"] == "test"]
    calib_ids = [i["id"] for i in spec["items"] if i["split"] == "calib"]
    test = _agreement(test_ids, human["test"], judged_val)
    calib = _agreement(calib_ids, human["calib"], judged_val)
    seed_reference = {s: JudgeLabel.NOT_SUPPORTED for s in seed_type if s not in excluded}
    gated = _agreement(test_ids + list(seed_reference), {**human["test"], **seed_reference},
                       judged_val)

    seeds_used = [s for s in seed_type if s not in excluded]
    caught = {s: judged_val[s]["label"] == JudgeLabel.NOT_SUPPORTED.value for s in seeds_used}
    per_type = {}
    for t in sorted(set(seed_type.values())):
        ids = [s for s in seeds_used if seed_type[s] == t]
        per_type[t] = {"n": len(ids), "caught": sum(caught[s] for s in ids),
                       "rate": sum(caught[s] for s in ids) / len(ids) if ids else None,
                       "missed": [s for s in ids if not caught[s]]}
    overall_catch = sum(caught.values()) / len(caught) if caught else None
    flip_val = sum(v["flipped"] for v in judged_val.values()) / len(judged_val)

    criteria = {
        "seed_catch_overall": {"value": overall_catch, "threshold": settings.JUDGE_MIN_SEED_CATCH,
                               "pass": overall_catch is not None
                               and overall_catch >= settings.JUDGE_MIN_SEED_CATCH},
        "seed_catch_per_type": {"value": {t: v["rate"] for t, v in per_type.items()},
                                "threshold": settings.JUDGE_MIN_SEED_CATCH_PER_TYPE,
                                "pass": all(v["rate"] is not None and v["rate"]
                                            >= settings.JUDGE_MIN_SEED_CATCH_PER_TYPE
                                            for v in per_type.values())},
        "kappa_test_plus_seeds": {"value": gated["kappa"], "n": gated["n"],
                                  "threshold": settings.JUDGE_MIN_KAPPA,
                                  "pass": gated["kappa"] is not None
                                  and gated["kappa"] >= settings.JUDGE_MIN_KAPPA},
        "flip_rate": {"value": flip_val, "threshold": settings.JUDGE_MAX_FLIP_RATE,
                      "pass": flip_val <= settings.JUDGE_MAX_FLIP_RATE},
    }
    passed = all(c["pass"] for c in criteria.values())

    # --- Scoring ---------------------------------------------------------------
    judged_claims = {k: _summary(v) for k, v in _judge_all(claims, chunks, names, runs).items()}
    by_claim = {c.id: c for c in claims}

    def rate(ids: list[str]) -> dict[str, Any]:
        labelled = [i for i in ids if judged_claims[i]["label"]]
        supported = sum(judged_claims[i]["label"] == JudgeLabel.SUPPORTED.value for i in labelled)
        return {"n": len(labelled), "supported": supported,
                "rate": supported / len(labelled) if labelled else None,
                "ci95": wilson_interval(supported, len(labelled)) if labelled else None}

    all_ids = list(judged_claims)
    scoring = {
        "all": rate(all_ids),
        "held": rate([i for i in all_ids if splits.get(by_claim[i].question_id) == "held"]),
        "dev": rate([i for i in all_ids if splits.get(by_claim[i].question_id) == "dev"]),
        "single_company_claims": rate([i for i in all_ids if by_claim[i].kind == "claim"]),
        "corpus_wide_findings": rate([i for i in all_ids if by_claim[i].kind == "finding"]),
        "flip_rate": sum(v["flipped"] for v in judged_claims.values()) / len(judged_claims),
        "failed_calls": sum(v["failed_calls"] for v in judged_claims.values()),
        "supported_span_not_found": sum(v["label"] == JudgeLabel.SUPPORTED.value
                                        and v["span_found"] is False for v in judged_claims.values()),
    }
    not_supported = [{"id": i, "question_id": by_claim[i].question_id, "company": by_claim[i].company,
                      "claim": by_claim[i].claim, "sources": by_claim[i].sources,
                      "detail": judged_claims[i]["unsupported_detail"], "runs": judged_claims[i]["runs"]}
                     for i in all_ids if judged_claims[i]["label"] == JudgeLabel.NOT_SUPPORTED.value]

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_version": version,
        "judge_model": settings.JUDGE_MODEL,
        "reasoning_effort": settings.JUDGE_REASONING_EFFORT,
        "structured_method": settings.JUDGE_STRUCTURED_METHOD,
        "runs": runs,
        "execution_file": execution_path.name,
        "execution_run": run_number,
        "judge_passed": passed,
        "reference_not_supported": sorted(i for i, lab in human["test"].items()
                                          if lab is JudgeLabel.NOT_SUPPORTED),
        "criteria": criteria,
        "validation": {"test": test, "calibration": calib, "seeds": per_type,
                       "seed_exclusions": excluded, "items": judged_val},
        "headline": scoring["all"] if passed else None,
        "spot_check_reference_test": {"n": test["n"], "supported_rate": test["human_supported_rate"]},
        "scoring": scoring,
        "not_supported_claims": not_supported,
        "claims": judged_claims,
    }
    out = eval_dir / settings.ATTRIBUTION_RESULTS_FILE
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(out.with_suffix(".md"), results)
    logger.info("attribution_eval_written", results=str(out), judge_passed=passed,
                supported_rate=scoring["all"]["rate"])
    return 0


def _pct(value: float | None) -> str:
    """Render a rate."""
    return "–" if value is None else f"{value:.1%}"


def _write_markdown(path: Path, r: dict[str, Any]) -> None:
    """Write the human-readable summary."""
    c, s, v = r["criteria"], r["scoring"], r["validation"]
    kappa = c["kappa_test_plus_seeds"]["value"]
    kappa_text = "–" if kappa is None else f"{kappa:.2f}"
    lines = [
        "# Attribution eval (D43)", "",
        f"judge `{r['judge_model']}` · reasoning `{r['reasoning_effort']}` · method "
        f"`{r['structured_method']}` · judge_version `{r['judge_version'][:12]}` · {r['runs']} runs · "
        f"scored: {r['execution_file']} end_to_end run {r['execution_run']}", "",
        f"## Judge validation — {'PASSED' if r['judge_passed'] else 'FAILED'}", "",
        "| criterion | value | threshold | pass |", "|---|---|---|---|",
        f"| seeded catch, overall | {_pct(c['seed_catch_overall']['value'])} | ≥ {c['seed_catch_overall']['threshold']:.0%} | {c['seed_catch_overall']['pass']} |",
        *[f"| seeded catch, {t} | {_pct(val)} | ≥ {c['seed_catch_per_type']['threshold']:.0%} | "
          f"{val is not None and val >= c['seed_catch_per_type']['threshold']} |"
          for t, val in c["seed_catch_per_type"]["value"].items()],
        f"| Cohen's kappa vs reference labels (test + seeds, n={c['kappa_test_plus_seeds']['n']}) | "
        f"{kappa_text} | ≥ {c['kappa_test_plus_seeds']['threshold']:.2f} | {c['kappa_test_plus_seeds']['pass']} |",
        f"| label flips across runs | {_pct(c['flip_rate']['value'])} | ≤ {c['flip_rate']['threshold']:.0%} | {c['flip_rate']['pass']} |",
        "",
        f"Test: raw agreement {_pct(v['test']['raw_agreement'])}, confusion {v['test']['confusion']}, "
        f"disagreements {', '.join(v['test']['disagreements']) or 'none'}.",
        f"Calibration (not gating): kappa {v['calibration']['kappa']}, disagreements "
        f"{', '.join(v['calibration']['disagreements']) or 'none'}.",
        f"Seeds missed: " + "; ".join(f"{t}: {', '.join(x['missed']) or 'none'}" for t, x in v["seeds"].items())
        + (f". Excluded as accidentally supported: {', '.join(v['seed_exclusions'])}." if v["seed_exclusions"] else "."),
        "",
        "### Judge vs reference disagreements on test claims (developer to adjudicate)", "",
        "| id | claim | reference | judge | judge's reason |", "|---|---|---|---|---|",
        *[f"| {i} | {v['items'][i]['claim'].replace('|', '/')} | "
          f"{'N' if i in r['reference_not_supported'] else 'S'} | {v['items'][i]['label']} | "
          f"{(v['items'][i]['unsupported_detail'] or v['items'][i]['supporting_span'] or '').replace('|', '/')} |"
          for i in v["test"]["disagreements"]],
        "",
        "## Attribution support", "",
    ]
    if r["judge_passed"]:
        h = r["headline"]
        lines.append(f"**Headline: {_pct(h['rate'])} of {h['n']} claims supported** "
                     f"[{h['ci95'][0]:.0%}–{h['ci95'][1]:.0%}] (Wilson, n = claims; claims cluster "
                     "within questions, so the true interval is wider).")
    else:
        sc = r["spot_check_reference_test"]
        lines.append(f"**Judge failed validation: score withheld.** Spot-check (reference labels): "
                     f"{_pct(sc['supported_rate'])} of {sc['n']} test claims supported.")
    lines += ["", "| group | n | supported | rate |", "|---|---|---|---|"]
    for name in ("all", "held", "dev", "single_company_claims", "corpus_wide_findings"):
        g = s[name]
        lines.append(f"| {name} | {g['n']} | {g['supported']} | {_pct(g['rate'])} |")
    lines += ["", f"Flip rate on scored claims {_pct(s['flip_rate'])}; failed calls {s['failed_calls']}; "
              f"SUPPORTED verdicts whose quoted span is not in the excerpts: {s['supported_span_not_found']}.",
              "", "## Claims judged NOT_SUPPORTED", "", "| id | claim | judge: unsupported detail | runs |",
              "|---|---|---|---|"]
    for x in r["not_supported_claims"]:
        claim = x["claim"].replace("|", "/")
        detail = (x["detail"] or "").replace("|", "/")
        lines.append(f"| {x['id'].replace('|', '/')} | {claim} | {detail} | {'/'.join(str(y)[:1] for y in x['runs'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--execution", type=Path, required=True,
                        help="Execution results JSON whose end_to_end claims are scored.")
    parser.add_argument("--runs", type=int, default=settings.QUERY_EVAL_RUNS)
    parser.add_argument("--run-number", type=int, default=1,
                        help="Which end_to_end run of the execution file to score (1-based).")
    args = parser.parse_args()
    raise SystemExit(main(args.execution, args.runs, args.run_number))
