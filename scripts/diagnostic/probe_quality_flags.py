"""
scripts/probe_quality_flags.py — DIAGNOSTIC (print() allowed).

Validates the D29 quality flag before anything is built on it: prints every
field of every stored record with its implied multiplier and verdict, then
asserts the controls.

Positive controls (known wrong, from Phase 3 held-out): BAC total_assets and
total_liabilities, GS operating_cash_flow.
Negative controls (known correct): AAPL, MSFT, GOOGL, and NFLX — Netflix prints
in thousands throughout, so it catches any rule that assumes millions globally.

Also cross-checks each recomputed evidence match against the stored
*_evidence_consistent column; a divergence means the database and the evidence
columns have drifted apart.

Read-only. No LLM, no network, no answer key.

Usage:
    python -m scripts.probe_quality_flags

Dependencies: src.storage.metadata_store, src.storage.quality, src.utils.logger.
"""

from __future__ import annotations

from collections import Counter

from src.storage.metadata_store import (
    EVIDENCE_FIELDS,
    get_session,
    latest_successful_stmt,
)
from src.storage.quality import QualityVerdict, assess_record
from src.utils.logger import get_logger

logger = get_logger(__name__)

POSITIVE_CONTROLS = {
    ("BAC", "total_assets"),
    ("BAC", "total_liabilities"),
    ("GS", "operating_cash_flow"),
}
NEGATIVE_CONTROLS = ("AAPL", "MSFT", "GOOGL", "NFLX")


def main() -> int:
    """Entry point. Returns the process exit code."""
    with get_session() as session:
        pairs = session.execute(latest_successful_stmt()).all()

    counts: Counter[str] = Counter()
    disputed: set[tuple[str, str]] = set()
    drift: list[str] = []
    failures: list[str] = []

    for doc, record in sorted(pairs, key=lambda p: p[0].ticker):
        flags = assess_record(record)
        print(f"\n{doc.ticker} {doc.filing_year}")
        for field in EVIDENCE_FIELDS:
            flag = flags[field]
            counts[flag.verdict.value] += 1
            if flag.verdict is QualityVerdict.DISPUTED:
                disputed.add((doc.ticker, field))

            stored = getattr(record, f"{field}_evidence_consistent")
            recomputed = None if getattr(record, field) is None else (
                flag.implied_multiplier is not None
            )
            if stored != recomputed:
                drift.append(f"{doc.ticker}.{field}: stored={stored} recomputed={recomputed}")

            multiplier = (
                "-" if flag.implied_multiplier is None else f"{flag.implied_multiplier:,.0f}"
            )
            print(
                f"  {field:<20} {str(getattr(record, field)):<18} "
                f"x{multiplier:<10} {flag.verdict.value:<10} {flag.reason.value}"
            )

    print(f"\nverdict counts: {dict(counts)}")

    missing = POSITIVE_CONTROLS - disputed
    if missing:
        failures.append(f"positive controls not flagged: {sorted(missing)}")

    clean_tickers = {t for t, _ in disputed}
    for ticker in NEGATIVE_CONTROLS:
        if ticker in clean_tickers:
            failures.append(
                f"negative control {ticker} has disputed fields: "
                f"{sorted(f for t, f in disputed if t == ticker)}"
            )

    if drift:
        failures.append(f"{len(drift)} evidence-column divergence(s): {drift[:5]}")

    print(f"\ndisputed fields ({len(disputed)}): {sorted(disputed)}")
    if failures:
        print("\nCONTROLS FAILED:")
        for line in failures:
            print(f"  {line}")
        logger.error("quality_probe_failed", failures=len(failures))
        return 1

    print("\nCONTROLS OK")
    logger.info("quality_probe_passed", disputed=len(disputed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())