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

import argparse

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


def _positive_int(text: str) -> int:
    """Parse a target cap, rejecting zero and negatives.

    A cap of 0 must not mean "no cap": that would start the full 20-filing
    acquisition when the user asked for none.

    Args:
        text: The command-line value.

    Returns:
        The cap, at least 1.

    Raises:
        argparse.ArgumentTypeError: if the value is not a whole number of at
            least 1 (argparse turns this into a usage error and exit code 2).
    """
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("the cap must be at least 1")
    return value


def select_targets(argv: list[str] | None = None) -> list[tuple[str, int]]:
    """Read the optional cap from the command line and return the targets to acquire.

    Args:
        argv: Command-line arguments without the program name; None reads
            sys.argv.

    Returns:
        The first N targets when a cap is given, otherwise all of them.

    Raises:
        SystemExit: on an invalid cap (exit code 2, with a usage message).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "limit", nargs="?", type=_positive_int, default=None,
        help="Acquire only the first N targets, for a quick test.",
    )
    limit = parser.parse_args(argv).limit
    return TARGETS if limit is None else TARGETS[:limit]


def main() -> None:
    """Run corpus acquisition and log the final summary.

    An optional command-line argument caps the number of targets, for
    incremental testing: `python -m scripts.acquire_filings 2` runs the first
    2 targets; with no argument, the full list. A cap below 1 is rejected.
    """
    targets = select_targets()
    logger.info(
        "acquire_filings_start", targets=len(targets), capped=len(targets) < len(TARGETS)
    )
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