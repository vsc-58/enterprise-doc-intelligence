"""
scripts/smoke_extract.py — throwaway diagnostic, not project code.

Runs ONE extraction (one strategy, one document) and prints the extracted values
side by side with the XBRL ground truth, so the four things the scorer cannot
tell you are visible before spending on the full bake-off:

  1. units      — is revenue 394328000000 (correct) or 394328 (scaled, wrong)?
  2. column     — does net_income match the current year or the prior year?
  3. cover      — did fiscal_year_end / company_name come back at all?
  4. usage      — does include_raw actually return token counts on this version?

Run: python -m scripts.smoke_extract S1 AAPL 2021
     python -m scripts.smoke_extract S1 NFLX 2023      (the fallback document)

# DIAGNOSTIC — throwaway investigation script, not project code.
# Uses print() deliberately for readable tabular output; project code uses logger.
"""

import json
import sys
from pathlib import Path

from sqlalchemy import select

from src.extraction.extractor import extract_with_strategy
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

NUMERIC_FIELDS = [
    "total_revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "operating_cash_flow",
]
STRING_FIELDS = ["company_name", "fiscal_year_end"]


def _resolve(ticker: str, year: int) -> str:
    """
    Look up the canonical CIK for a ticker/year from the Document row.

    Args:
        ticker: the company ticker.
        year: fiscal-period-end year.

    Returns:
        The canonical 10-digit CIK.

    Raises:
        LookupError: if no Document row exists for that pair.
    """
    with get_session() as session:
        doc = session.execute(
            select(Document).where(
                Document.ticker == ticker, Document.filing_year == year
            )
        ).scalar_one_or_none()
    if doc is None:
        raise LookupError(f"No Document row for {ticker} {year}")
    return doc.cik


def _load_truth(cik: str, year: int) -> dict:
    """
    Load this document's ground-truth record.

    Args:
        cik: canonical 10-digit CIK.
        year: fiscal-period-end year.

    Returns:
        The ground-truth record, or an empty dict if the key is absent.
    """
    path = Path(settings.EVAL_DATA_PATH) / "ground_truth.json"
    truth = json.loads(path.read_text(encoding="utf-8"))
    return truth.get(f"{cik}_{year}", {})


def _ratio(extracted: float | None, actual: float | None) -> str:
    """
    Return extracted/truth as a readable ratio — the fastest unit-error tell.

    ~1.0 means correct. ~0.000001 means the model returned a scaled ("in
    millions") figure. ~1000000 means it over-multiplied.
    """
    if extracted is None or actual in (None, 0):
        return "-"
    return f"{extracted / actual:,.6f}"


def main() -> None:
    """Run one extraction and print extracted vs ground truth side by side."""
    if len(sys.argv) != 4:
        logger.error("usage", expected="python -m scripts.smoke_extract <STRATEGY> <TICKER> <YEAR>")
        sys.exit(1)

    strategy_id, ticker, year = sys.argv[1], sys.argv[2], int(sys.argv[3])

    cik = _resolve(ticker, year)
    truth = _load_truth(cik, year)
    result = extract_with_strategy(strategy_id, cik, year, use_cache=False)

    logger.info(
        "smoke_meta",
        strategy=strategy_id,
        ticker=ticker,
        year=year,
        cik=cik,
        section_source=result.section_source,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        from_cache=result.from_cache,
        parsing_error=result.parsing_error,
    )

    if result.extraction is None:
        logger.error("smoke_no_extraction", parsing_error=result.parsing_error)
        sys.exit(1)

    ex = result.extraction

    print(f"\n{'field':<24}{'extracted':>22}{'ground truth':>22}{'ratio':>14}")
    print("-" * 82)

    for field in NUMERIC_FIELDS:
        got = getattr(ex, field)
        want = truth.get(field)
        got_s = f"{got:,.0f}" if got is not None else "None"
        want_s = f"{want:,.0f}" if want is not None else "None"
        print(f"{field:<24}{got_s:>22}{want_s:>22}{_ratio(got, want):>14}")

    print("-" * 82)
    for field in STRING_FIELDS:
        got = getattr(ex, field, None)
        want = truth.get(field)
        print(f"{field:<24}{str(got):>22}{str(want):>22}{'':>14}")

    print("-" * 82)
    print(f"{'auditor_name':<24}{str(ex.auditor_name):>22}")
    print(f"{'business_description':<24}{('set' if ex.business_description else 'None'):>22}")
    print(f"{'primary_risk_factors':<24}{len(ex.primary_risk_factors):>22}")

    if result.evidence:
        print("\nevidence (S3):")
        for field, ev in result.evidence.items():
            line = (ev.get("source_line") or "")[:70]
            print(f"  {field:<22} {ev.get('value')}  <- {line!r}")
    print()


if __name__ == "__main__":
    main()