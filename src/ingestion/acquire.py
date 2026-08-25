# src/ingestion/acquire.py
# Module: Filing acquisition (edgartools wrapper)
# Purpose: Locate a company's 10-K for a target fiscal year, save its clean
#          text to disk, and return tracking metadata. TEXT side only — XBRL
#          ground truth is captured separately in src/eval/ground_truth.py.
# Depends on: edgartools, src.utils.config, src.utils.logger

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from edgar import Company, set_identity

from src.utils.config import settings
from src.utils.logger import get_logger
from datetime import date, datetime ## added for filing_date extraction

if TYPE_CHECKING:
    # Imported for type-checking only; keeps runtime import surface minimal.
    # If Pylance flags this path, the class is exported from `edgar`.
    from edgar import Filing

logger = get_logger(__name__)

# Module-level guard so identity is set once per process without making the
# import itself perform that side effect.
_identity_set: bool = False

## added for filing_date extraction
def to_iso_date(value: object) -> str | None:
    """
    Normalise edgartools' filing_date to an ISO 'YYYY-MM-DD' string.

    edgartools' return type varies by code path (date | datetime | str |
    pandas Timestamp); this coerces every shape to one stored form.

    Args:
        value: The raw filing_date from a filing object.

    Returns:
        ISO date string 'YYYY-MM-DD', or None if empty/unparseable.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date().isoformat()
    except ValueError:
        logger.warning("filing_date_unparseable", raw=text)
        return None

def ensure_identity() -> None:
    """Set the SEC identity for edgartools exactly once per process. (Renamed
    from _ensure_identity — now part of acquire.py's shared public surface,
    used by both acquisition and ground-truth capture.)"""
    global _identity_set
    if not _identity_set:
        set_identity(settings.EDGAR_IDENTITY)
        _identity_set = True
        logger.debug("edgar_identity_set")


def normalize_cik(cik: int | str) -> str:
    """Normalise a CIK to EDGAR's canonical zero-padded 10-digit string.
    (Renamed from _normalize_cik — shared so the raw-text filename, the SQLite
    cik column, and the ground_truth.json key all use the identical form.)"""
    return f"{int(cik):010d}"


def _period_end_year(filing: Filing) -> int | None:
    """
    Return the calendar year in which a filing's fiscal period ends.

    Reads filing.period_of_report (the fiscal-period-end date, e.g.
    "2023-09-30" for Apple's FY2023). This is the selection discriminator,
    because edgartools' `year` filter keys on FILING (calendar) year, which
    diverges from fiscal year for any company that files after its
    fiscal-year-end calendar year (every December-year filer). Selecting on
    period-of-report avoids that silent off-by-one.

    Args:
        filing: An edgartools Filing/EntityFiling object.

    Returns:
        The 4-digit period-end year, or None if it cannot be determined.
    """
    period = getattr(filing, "period_of_report", None)
    if period is None:
        return None
    try:
        # Robust whether period_of_report is a date object or an ISO string.
        return int(str(period)[:4])
    except (ValueError, TypeError):
        return None


def _extract_accession(filing: Filing) -> str | None:
    """
    Read the accession number, tolerant of edgartools' attribute naming.

    The accession identifier is exposed as `accession_number` in some
    edgartools versions and `accession_no` in others. This reads whichever
    is present rather than betting on one name and raising AttributeError.

    Args:
        filing: An edgartools Filing/EntityFiling object.

    Returns:
        The accession number string, or None if neither attribute is set.
    """
    return getattr(filing, "accession_number", None) or getattr(
        filing, "accession_no", None
    )


def select_10k_for_fiscal_year(ticker: str, year: int) -> Filing | None:
    """
    Locate a company's original 10-K whose fiscal period ENDS in `year`.

    Shared by acquisition (filing text) and ground-truth capture (XBRL) so
    both operate on the IDENTICAL filing — same period, same accession. That
    shared identity is what keeps the {cik}_{year} raw-text file and the
    {cik}_{year} ground-truth key aligned; if the two selected different
    filings, the Phase 2 evaluator would score extraction against the wrong
    answer key. Selection is by period_of_report (fiscal-period-end year),
    NOT edgartools' `year` filter, which keys on filing/calendar year.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        year: Calendar year in which the target fiscal period ends.

    Returns:
        The matching Filing, or None if no original 10-K matches that year.
    """
    ensure_identity()
    company = Company(ticker)
    filings = company.get_filings(form="10-K", amendments=False)
    for filing in filings:
        if _period_end_year(filing) == year:
            return filing
    return None


def acquire_filing(ticker: str, year: int) -> dict | None:
    """Acquire a company's 10-K for a target fiscal year and save its text.

    IMPORTANT — what `year` means: the calendar year in which the fiscal
    period ENDS, not the year the 10-K was filed. Apple's FY2023 ends
    2023-09-30; a December-year company's FY2023 ends 2023-12-31 but files
    in early 2024. Selection is by filing.period_of_report, NOT edgartools'
    `year` filter (which keys on filing year and would return the wrong
    year's 10-K for every December-year filer). For January-fiscal-year
    retailers (e.g. Walmart, Target) the period-end year is one greater than
    the company's own "fiscal year" label; this function keys on period-end
    year uniformly and the caller chooses targets accordingly.

    Single-item operation, but invoked inside a batch: it never raises.
    Any failure is logged with ticker and year and returns None, so one bad
    filing cannot interrupt the corpus acquisition loop.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        year: Calendar year in which the target fiscal period ends.

    Returns:
        A dict with keys company_name, cik (zero-padded 10-digit string),
        accession_number, filing_year (the period-end year), and local_path;
        or None if no matching 10-K is found or acquisition fails."""
    logger.info("acquire_filing_start", ticker=ticker, year=year)
    try:
        matched_filing = select_10k_for_fiscal_year(ticker, year)
        if matched_filing is None:
            logger.warning("acquire_filing_not_found", ticker=ticker, year=year)
            return None

        cik = normalize_cik(matched_filing.cik)
        accession_number = _extract_accession(matched_filing)
        filing_date = to_iso_date(matched_filing.filing_date) ## added for filing_date extraction

        text = matched_filing.text()
        if not text or not text.strip():
            logger.error(
                "acquire_filing_empty_text",
                ticker=ticker, year=year, cik=cik,
                accession_number=accession_number,
            )
            return None

        raw_dir = Path(settings.RAW_DATA_PATH)
        raw_dir.mkdir(parents=True, exist_ok=True)
        local_path = raw_dir / f"{cik}_{year}.txt"
        local_path.write_text(text, encoding="utf-8")

        company_name = getattr(matched_filing, "company", None) or ticker
        logger.info(
            "acquire_filing_success",
            ticker=ticker, year=year, cik=cik,
            accession_number=accession_number,
            chars=len(text), local_path=str(local_path),
        )
        return {
            "company_name": company_name,
            "cik": cik,
            "accession_number": accession_number,
            "filing_year": year,
            "local_path": str(local_path),
            "filing_date": filing_date, ## added for filing_date extraction
        }
    except Exception as exc:
        logger.error(
            "acquire_filing_error",
            ticker=ticker, year=year,
            error_type=type(exc).__name__, error=str(exc),
        )
        return None