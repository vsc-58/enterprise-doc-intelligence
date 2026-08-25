# src/ingestion/pipeline.py
# Module: Acquisition orchestration
# Purpose: Drive the corpus acquisition loop — for each (ticker, year): skip
#          if already present, else acquire filing text, capture XBRL ground
#          truth, and persist a tracking row only when BOTH succeed.
# Depends on: src.ingestion.acquire, src.eval.ground_truth,
#             src.storage.metadata_store, src.utils.logger

from __future__ import annotations

from pathlib import Path

from src.eval.ground_truth import capture_ground_truth, write_ground_truth
from src.ingestion.acquire import acquire_filing
from src.storage.metadata_store import create_document, document_exists, init_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _cleanup_partial(local_path: str | None) -> None:
    """
    Delete an orphaned raw-text file left by a half-completed acquisition.

    When text was saved but the document did not fully succeed (ground-truth
    capture failed, or the row write raised), the .txt would otherwise linger
    with no Document row, contradicting the implicit-completeness model where
    no committed row means no artifacts. Cleanup failure is logged, never
    raised — it must not crash the batch.

    Args:
        local_path: Path to the raw-text file, or None.
    """
    if not local_path:
        return
    try:
        path = Path(local_path)
        if path.exists():
            path.unlink()
            logger.info("acquire_corpus_partial_cleaned", path=local_path)
    except OSError as exc:
        logger.warning(
            "acquire_corpus_cleanup_failed", path=local_path, error=str(exc)
        )


def acquire_corpus(targets: list[tuple[str, int]]) -> dict:
    """
    Acquire a corpus of 10-K filings, tracking each in SQLite and capturing
    XBRL ground truth.

    Per (ticker, year):
      1. Skip (INFO) if a Document row already exists — no network work.
      2. Acquire filing text; on failure, count failed, continue.
      3. Capture XBRL ground truth; on failure, delete the orphan text,
         count failed, continue.
      4. Both succeeded: write the ground-truth record (durable), THEN
         persist the Document row. The row is written last and is the skip
         key, so a committed row implies its ground truth is already on disk;
         a crash or Ctrl-C between documents leaves both stores in sync and
         the next run resumes cleanly.

    One document's failure never stops the batch: per-item errors are caught,
    logged with ticker and year, and counted.

    Args:
        targets: List of (ticker, year) pairs, year in period-end convention.

    Returns:
        Summary dict: total, acquired, skipped, failed, and a failures list
        of {ticker, year, stage} for diagnosis.
    """
    init_db()

    summary: dict = {
        "total": len(targets),
        "acquired": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
    }

    for ticker, year in targets:
        # Heartbeat: how many targets have been fully handled so far.
        done = summary["acquired"] + summary["skipped"] + summary["failed"]
        if done and done % 5 == 0:
            logger.info(
                "acquire_corpus_progress", processed=done, total=summary["total"]
            )

        acquired: dict | None = None
        try:
            if document_exists(ticker, year):
                logger.info("acquire_corpus_skip", ticker=ticker, year=year)
                summary["skipped"] += 1
                continue

            acquired = acquire_filing(ticker, year)
            if acquired is None:
                logger.warning(
                    "acquire_corpus_acquire_failed", ticker=ticker, year=year
                )
                summary["failed"] += 1
                summary["failures"].append(
                    {"ticker": ticker, "year": year, "stage": "acquire"}
                )
                continue

            truth = capture_ground_truth(ticker, year)
            if truth is None:
                _cleanup_partial(acquired["local_path"])
                logger.warning(
                    "acquire_corpus_ground_truth_failed",
                    ticker=ticker,
                    year=year,
                )
                summary["failed"] += 1
                summary["failures"].append(
                    {"ticker": ticker, "year": year, "stage": "ground_truth"}
                )
                continue

            # Truth before row: a committed row must always imply its ground
            # truth is already persisted, so skip-on-row-existence is safe.
            write_ground_truth([truth])
            doc_id = create_document(
                company_name=acquired["company_name"],
                ticker=ticker,
                cik=acquired["cik"],
                filing_year=acquired["filing_year"],
                accession_number=acquired["accession_number"],
                local_path=acquired["local_path"],
                filing_date=acquired["filing_date"], ## added for filing_date extraction 
            )

            summary["acquired"] += 1
            logger.info(
                "acquire_corpus_acquired",
                ticker=ticker,
                year=year,
                cik=acquired["cik"],
                doc_id=doc_id,
            )

        except Exception as exc:
            # Batch safety: any unexpected error for one document is contained.
            # If text was written, remove the orphan (e.g. the row write raised
            # after ground truth was persisted — the truth entry is harmless,
            # gets overwritten identically on the self-healing re-run).
            if acquired is not None:
                _cleanup_partial(acquired.get("local_path"))
            logger.error(
                "acquire_corpus_item_error",
                ticker=ticker,
                year=year,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            summary["failed"] += 1
            summary["failures"].append(
                {"ticker": ticker, "year": year, "stage": "unexpected"}
            )
            continue

    logger.info(
        "acquire_corpus_complete",
        total=summary["total"],
        acquired=summary["acquired"],
        skipped=summary["skipped"],
        failed=summary["failed"],
    )
    return summary