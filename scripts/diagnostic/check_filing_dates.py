"""Log ticker, fiscal year and filing date for every filing, and list any missing date.

Read-only check of the documents table. No network, no API calls.

Usage:
    python -m scripts.diagnostic.check_filing_dates

Diagnostic script: print() is used deliberately for readable output;
project code logs through structlog.
"""

from sqlalchemy import select

from src.storage.metadata_store import Document, get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Print ticker, filing_year, filing_date for every Document row."""
    with get_session() as session:
        docs = session.execute(select(Document)).scalars().all()
        for d in sorted(docs, key=lambda x: x.ticker):
            logger.info(
                "filing_date_check",
                ticker=d.ticker,
                filing_year=d.filing_year,
                filing_date=d.filing_date,
            )
        missing = [d.ticker for d in docs if not d.filing_date]
        logger.info("summary", total=len(docs), missing=missing or None)


if __name__ == "__main__":
    main()