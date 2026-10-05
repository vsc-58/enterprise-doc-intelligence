"""
scripts/verify_numbers_parity.py — DIAGNOSTIC (print() allowed).

Regression gate for the D-numbers refactor: proves that moving the numeric
token parser and scale multipliers out of src/eval/grounding.py and into
src/utils/numbers.py changed no verdict.

check_evidence_consistency is the only function the move touches, so it is
what this compares. check_grounding uses _flatten_for_grounding, which stays
in grounding.py untouched, and would need the raw filing text (gitignored)
to re-run — it is smoke-imported here, not re-scored.

Usage:
    python -m scripts.verify_numbers_parity --baseline   # BEFORE the refactor
    python -m scripts.verify_numbers_parity              # AFTER; exits 1 on drift

Dependencies: src.eval.grounding, src.utils.config, src.utils.logger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.grounding import check_evidence_consistency, check_grounding
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_OUTPUTS_DIR = Path("data/eval/raw_outputs")
BASELINE_PATH = Path("data/eval/grounding_parity_baseline.json")


def collect_verdicts() -> dict[str, dict[str, bool | None]]:
    """
    Recompute evidence-consistency verdicts over every cached raw output.

    Only S3 and S4 carry evidence maps; the other strategies are skipped
    rather than recorded as empty, so a missing strategy is visible as a
    missing key rather than as a silent pass.

    Returns:
        Key "<strategy>/<cik>_<year>" -> field -> True | False | None.

    Raises:
        FileNotFoundError: if the cached-output directory is absent.
    """
    if not RAW_OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"no cached outputs at {RAW_OUTPUTS_DIR}")

    verdicts: dict[str, dict[str, bool | None]] = {}
    for path in sorted(RAW_OUTPUTS_DIR.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))            
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("raw_output_unreadable", path=str(path), error=str(exc))
            continue

        evidence = payload.get("evidence")
        if not evidence:
            continue

        key = f"{payload['strategy_id']}/{payload['cik']}_{payload['year']}"
        verdicts[key] = check_evidence_consistency(evidence)

    return verdicts


def summarise(verdicts: dict[str, dict[str, bool | None]]) -> dict[str, int]:
    """
    Count verdicts by outcome across every document and field.

    Args:
        verdicts: the map returned by collect_verdicts.

    Returns:
        Counts keyed "true", "false", "none".
    """
    counts = {"true": 0, "false": 0, "none": 0}
    for fields in verdicts.values():
        for value in fields.values():
            counts["true" if value is True else "false" if value is False else "none"] += 1
    return counts


def write_baseline(verdicts: dict[str, dict[str, bool | None]]) -> None:
    """
    Persist the pre-refactor verdicts.

    Args:
        verdicts: the map returned by collect_verdicts.

    Raises:
        OSError: if the baseline cannot be written.
    """
    BASELINE_PATH.write_text(
        json.dumps(verdicts, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("baseline_written", path=str(BASELINE_PATH), documents=len(verdicts))


def compare(verdicts: dict[str, dict[str, bool | None]]) -> int:
    """
    Compare current verdicts against the stored baseline.

    Args:
        verdicts: the map returned by collect_verdicts.

    Returns:
        0 when every verdict matches, 1 otherwise.

    Raises:
        FileNotFoundError: if no baseline has been written.
    """
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"no baseline at {BASELINE_PATH} — run with --baseline before refactoring"
        )

    baseline: dict[str, dict[str, bool | None]] = json.loads(
        BASELINE_PATH.read_text(encoding="utf-8")
    )

    drift: list[str] = []
    for key in sorted(set(baseline) | set(verdicts)):
        before, after = baseline.get(key), verdicts.get(key)
        if before is None or after is None:
            drift.append(f"{key}: present in {'baseline' if after is None else 'current'} only")
            continue
        for field in sorted(set(before) | set(after)):
            if before.get(field) != after.get(field):
                drift.append(f"{key}.{field}: {before.get(field)} -> {after.get(field)}")

    print(f"\ndocuments compared : {len(verdicts)}")
    print(f"tolerance in use   : {settings.EVIDENCE_MATCH_REL_TOLERANCE}")
    print(f"verdict counts     : {summarise(verdicts)}")

    if drift:
        print(f"\nDRIFT — {len(drift)} verdict(s) changed:")
        for line in drift:
            print(f"  {line}")
        logger.error("parity_failed", drift_count=len(drift))
        return 1

    print("\nPARITY OK — every verdict identical to baseline.")
    logger.info("parity_passed", documents=len(verdicts))
    return 0


def main() -> int:
    """Entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="write the baseline")
    args = parser.parse_args()

    assert check_grounding(None, "") == {}, "check_grounding smoke failed"

    verdicts = collect_verdicts()
    if args.baseline:
        write_baseline(verdicts)
        print(f"baseline written: {len(verdicts)} documents, {summarise(verdicts)}")
        return 0
    return compare(verdicts)


if __name__ == "__main__":
    raise SystemExit(main())