# scripts/acquire_filings.py
# Runnable entrypoint: acquire the Phase 1 target corpus.
# Run from the project root with the venv active:
#   python scripts/acquire_filings.py

import sys
from src.ingestion.pipeline import acquire_corpus
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Phase 1 targets: 20 companies, years cycled across 2021-2023 for variety,
# giving 20 (ticker, year) pairs = 20 Document rows (matches the Phase 1
# VERIFY count). To grow toward the 50-100 corpus later, switch to the full
# cross-product (every ticker × every year) — see note below.
_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "JPM", "GS", "BAC", "WMT", "TGT",
    "XOM", "PFE", "JNJ", "TSLA", "NFLX",
    "CRM", "ADBE", "INTC", "QCOM", "V",
)
_YEARS: tuple[int, ...] = (2021, 2022, 2023)

TARGETS: list[tuple[str, int]] = [
    (ticker, _YEARS[i % len(_YEARS)]) for i, ticker in enumerate(_TICKERS)
]

# Larger corpus later — full cross-product (20 × 3 = 60 rows):
# TARGETS = [(t, y) for t in _TICKERS for y in _YEARS]
def main() -> None:
    """Run corpus acquisition and log the final summary.

    Optional first CLI arg caps the number of targets, for incremental
    testing: `python scripts/acquire_filings.py 2` runs the first 2 targets;
    with no arg, runs the full list.
    """
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    targets = TARGETS[:limit] if limit else TARGETS

    logger.info("acquire_filings_start", targets=len(targets), capped=bool(limit))
    summary = acquire_corpus(targets)
    logger.info(
        "acquire_filings_done",
        total=summary["total"],
        acquired=summary["acquired"],
        skipped=summary["skipped"],
        failed=summary["failed"],
    )
    if summary["failures"]:
        logger.warning("acquire_filings_failures", failures=summary["failures"])


if __name__ == "__main__":
    main()