"""
scripts/diagnostic/probe_judge.py — DIAGNOSTIC (print() allowed).

Probe the attribution judge model before building on it (D43): does
structured output work with the configured model, reasoning setting and
method, and does it give the obvious verdict on two hand-made cases?

If the configured method fails, the other method is tried and reported, so the
fix is a one-line settings change (JUDGE_STRUCTURED_METHOD in .env).

Two judge calls per method tried. No store access.

Usage:
    python -m scripts.diagnostic.probe_judge

Dependencies: src.eval.attribution, src.storage.vector_store, src.utils.config.
"""

from __future__ import annotations

import sys
import time

from src.eval import attribution
from src.eval.attribution import JudgeLabel, judge_claim, judge_version
from src.storage.vector_store import RetrievedChunk
from src.utils.config import settings

BLOCK = RetrievedChunk(
    text=("We have $15.0 billion of various undrawn committed lines of credit in the U.S. "
          "that provide additional liquidity, if needed."),
    ticker="WMT", company_name="Walmart Inc.", cik="0000104169", fiscal_year=2023,
    section="Item 7", chunk_index=25, distance=0.0,
)
CASES = [
    ("Walmart has $15.0 billion of undrawn committed U.S. credit lines for additional liquidity.",
     JudgeLabel.SUPPORTED),
    ("Walmart has $15.0 billion of undrawn committed U.S. credit lines that expire in 2026.",
     JudgeLabel.NOT_SUPPORTED),
]


def run(method: str) -> bool:
    """Run both cases with one structured-output method; True if both are right."""
    settings.JUDGE_STRUCTURED_METHOD = method
    attribution._llm = None
    print(f"\nmodel={settings.JUDGE_MODEL} reasoning_effort={settings.JUDGE_REASONING_EFFORT} "
          f"method={method} judge_version={judge_version()[:12]}")
    ok = True
    for claim, expected in CASES:
        start = time.perf_counter()
        result = judge_claim("Walmart Inc.", claim, [BLOCK])
        elapsed = time.perf_counter() - start
        right = result.label is expected
        ok &= right
        print(f"  [{'OK' if right else 'WRONG'}] {elapsed:.1f}s expected={expected.value} "
              f"got={result.label.value if result.label else None}")
        print(f"      span={result.supporting_span!r} span_found={result.span_found}")
        print(f"      detail={result.unsupported_detail!r} error={result.error}")
    return ok


def main() -> int:
    """Probe the configured method, then the alternative if it fails."""
    configured = settings.JUDGE_STRUCTURED_METHOD
    if run(configured):
        print(f"\nPASS with configured method '{configured}'.")
        return 0
    other = "json_schema" if configured == "function_calling" else "function_calling"
    if run(other):
        print(f"\nConfigured method failed; '{other}' works. Set JUDGE_STRUCTURED_METHOD={other} in .env.")
        return 0
    print("\nFAILED with both methods: send this output.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
