"""scripts/check_filing_dates.py — throwaway verification.

# DIAGNOSTIC — throwaway investigation script, not project code.
# Uses print() deliberately for readable tabular output; project code uses logger."""

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