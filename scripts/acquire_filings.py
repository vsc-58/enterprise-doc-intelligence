"""Acquire the corpus: one 10-K per company for the 20 target companies.

For each (ticker, fiscal year) target: fetch the filing's text with edgartools,
capture its XBRL figures as the evaluation answer key, and record the filing in
SQLite. Already-acquired filings are skipped, so a re-run only fetches what is
missing.

Side effects: calls SEC EDGAR (settings.EDGAR_IDENTITY must name you, as the
SEC requires), writes data/raw/{cik}_{year}.txt, merges records into
data/eval/ground_truth.json and adds rows to the documents table. No OpenAI
calls.

Usage (from the repo root):
    python -m scripts.acquire_filings        # all 20 targets
    python -m scripts.acquire_filings 2      # only the first 2, for a quick test
"""

import sys
from src.ingestion.pipeline import acquire_corpus
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 20 companies, fiscal years cycled across 2021-2023 for variety: 20 (ticker,
# year) pairs, so 20 Document rows. To grow the corpus, switch to the full
# cross-product (every ticker x every year); see the note below.
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

    An optional first command-line argument caps the number of targets, for
    incremental testing: `python -m scripts.acquire_filings 2` runs the first
    2 targets; with no argument, the full list.
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