"""
scripts/run_extraction.py
Run the winning extraction strategy across every un-extracted document.

Selects documents where Document.is_extracted is False — the single unambiguous
signal, since is_extracted is set only by a SUCCESS write. A document with two
failed attempts is still pending and gets picked up again; a successful one is
never re-billed.

Cost is reported split: tokens billed on this run vs tokens replayed from the
Phase 2 response cache. The eval-10 hit that cache (their prompt, model and
sliced input are unchanged), so a correct run shows ~10 cached hits and pays
only for the held-out half.

Run: python -m scripts.run_extraction [--strategy S3] [--limit N] [--dry-run]

Dependencies: src.extraction.extractor, src.storage.metadata_store,
src.utils.config, src.utils.logger.
"""

import argparse

from sqlalchemy import select

from src.extraction.extractor import run_extraction_pipeline
from src.storage.metadata_store import Document, get_session, init_db
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# OpenAI list prices per 1M tokens for the estimate below. Module-level facts,
# not settings — they belong to the vendor, not to this deployment, and a stale
# value here misreports cost rather than breaking behaviour.
_INPUT_USD_PER_MTOK = 0.15
_OUTPUT_USD_PER_MTOK = 0.60


def _pending_documents() -> list[tuple[int, str, int]]:
    """
    Return (id, ticker, filing_year) for every document not yet extracted.

    Ordered by id so a resumed run processes in the same sequence as the first,
    which makes a partial run's progress readable against the previous log.

    Returns:
        Pending documents, oldest first.
    """
    with get_session() as session:
        rows = session.execute(
            select(Document.id, Document.ticker, Document.filing_year)
            .where(Document.is_extracted.is_(False))
            .order_by(Document.id)
        ).all()
    return [(row[0], row[1], row[2]) for row in rows]


def _estimate_usd(input_tokens: int, output_tokens: int) -> float:
    """
    Convert token counts to an approximate USD cost.

    Args:
        input_tokens: prompt tokens billed.
        output_tokens: completion tokens billed.

    Returns:
        Estimated spend in USD.
    """
    return (
        input_tokens / 1_000_000 * _INPUT_USD_PER_MTOK
        + output_tokens / 1_000_000 * _OUTPUT_USD_PER_MTOK
    )


def main() -> None:
    """
    Extract all pending documents and report the outcome and cost.

    --dry-run lists what would be processed and exits without calling the model
    or writing any record: the last checkpoint before spending.
    """
    parser = argparse.ArgumentParser(
        description="Run the extraction pipeline over pending documents."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="S3",
        help="Strategy id to apply. Defaults to the Phase 2 winner.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N documents (incremental testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending documents and exit without extracting.",
    )
    args = parser.parse_args()

    init_db()
    pending = _pending_documents()

    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        logger.info("extraction_nothing_pending")
        return

    logger.info(
        "extraction_run_start",
        pending=len(pending),
        strategy=args.strategy,
        model=settings.OPENAI_MODEL,
        documents=[f"{ticker}_{year}" for _, ticker, year in pending],
    )

    if args.dry_run:
        logger.info("extraction_dry_run_complete", would_process=len(pending))
        return

    summary = run_extraction_pipeline(
        [doc_id for doc_id, _, _ in pending], strategy_id=args.strategy
    )

    billed_usd = _estimate_usd(
        int(summary["billed_input_tokens"]), int(summary["billed_output_tokens"])
    )
    replayed_usd = _estimate_usd(
        int(summary["replayed_input_tokens"]),
        int(summary["replayed_output_tokens"]),
    )

    logger.info(
        "extraction_run_complete",
        processed=summary["processed"],
        succeeded=summary["succeeded"],
        extraction_failed=summary["extraction_failed"],
        technical_failed=summary["technical_failed"],
        cached_hits=summary["cached_hits"],
        billed_tokens=summary["billed_input_tokens"]
        + summary["billed_output_tokens"],
        estimated_usd=round(billed_usd, 4),
        saved_by_cache_usd=round(replayed_usd, 4),
    )


if __name__ == "__main__":
    main()