# src/eval/ground_truth.py
# Module: XBRL ground-truth capture (EVALUATION SIDE ONLY)
# Purpose: Read five financial fields STRICTLY from a specific 10-K's XBRL —
#          by exact concept, undimensioned, for the filing's own period — and
#          write them to data/eval/ground_truth.json as the Phase 2 answer key.
# Depends on: edgartools, src.ingestion.acquire (shared selector),
#             src.utils.config, src.utils.logger
#
# BOUNDARY RULE (non-negotiable): these XBRL values are the EVALUATION answer
# key ONLY. They are written to data/eval/ground_truth.json and must NEVER be
# written into the extraction store (extracted_records in SQLite). The fields
# the system PRODUCES come from the LLM extraction pipeline — the project's
# contribution. XBRL as output would collapse the thesis.
#
# WHY STRICT DATAFRAME READS, not the get_*() accessors:
#   get_total_liabilities() and friends substitute a *resembling* concept when
#   the exact one is untagged — e.g. Amazon has no us-gaap:Liabilities, so the
#   accessor returned LiabilitiesAndStockholdersEquity (= total assets), a
#   silent wrong value scored as "captured". We instead match the exact XBRL
#   concept, take only the undimensioned consolidated row, and store null when
#   the concept is genuinely absent. Missing-means-null, never substituted.

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.ingestion.acquire import normalize_cik, select_10k_for_fiscal_year
from src.utils.config import settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from edgar import Filing

logger = get_logger(__name__)

# Each ground-truth field mapped to its statement and an ORDERED list of
# candidate XBRL concepts. The list holds TRUE SYNONYMS only — the same
# economic line under an old vs new tag — tried newest/most-specific first;
# the first concept present in the filing wins. A list is NOT a license to
# include resembling-but-different concepts (that would rebuild the fuzzy-
# substitution bug by hand). If none match, the field is null.
#
# Deliberately NOT included: us-gaap_RevenuesNetOfInterestExpense (banks'
# "total net revenue"). Bank net revenue is a different concept from an
# industrial's total revenue — it nets out interest expense — so scoring LLM
# extraction against it would measure concept-matching, not extraction.
# Banks (JPM, GS, BAC) therefore get null total_revenue, excluded from
# scoring. Honest gap over a not-quite-equivalent value.
_FIELD_SPECS: dict[str, dict[str, object]] = {
    "total_revenue": {
        "statement": "income_statement",
        "concepts": [
            "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap_Revenues",
        ],
    },
    "net_income": {
        "statement": "income_statement",
        "concepts": [
            "us-gaap_NetIncomeLoss",
            "us-gaap_ProfitLoss",
        ],
    },
    "total_assets": {
        "statement": "balance_sheet",
        "concepts": ["us-gaap_Assets"],
    },
    "total_liabilities": {
        "statement": "balance_sheet",
        "concepts": ["us-gaap_Liabilities"],
    },
    "operating_cash_flow": {
        "statement": "cash_flow_statement",
        "concepts": ["us-gaap_NetCashProvidedByUsedInOperatingActivities"],
    },
}


def _period_column(df: pd.DataFrame, period_of_report: str) -> str | None:
    """
    Pick the statement's value column matching the filing's fiscal period.

    Statement dataframes name their period columns by date, but the format
    differs by statement type: point-in-time statements (balance sheet) use a
    bare end date — "2023-12-31" — while duration statements (income, cash
    flow) suffix it to mark the full-year span — "2023-12-31 (FY)". So an
    exact-equality match works only for the balance sheet and silently nulls
    every income-statement and cash-flow field. This matches the column whose
    date component equals period_of_report, tolerating the suffix.

    The period_of_report is a fiscal-year-end date (Apple's is 2021-09-25),
    not necessarily a calendar year-end, so the date itself is read from the
    filing rather than assumed.

    Args:
        df: A statement dataframe from Statement.to_dataframe().
        period_of_report: The filing's fiscal-period-end date string
            (e.g. "2023-12-31").

    Returns:
        The matching column name, or None if no column's date matches.
    """
    # Prefer an exact match (balance-sheet / point-in-time columns).
    if period_of_report in df.columns:
        return period_of_report
    # Fall back to a column whose date component equals the period, ignoring a
    # trailing annotation like " (FY)" on duration-statement columns.
    for col in df.columns:
        col_str = str(col)
        # The date is the leading token; split off any " (FY)"-style suffix.
        if col_str.split(" ")[0] == period_of_report:
            return col
    return None


def _read_concept(
    df: pd.DataFrame, concepts: list[str], value_col: str
) -> float | None:
    """
    Read the first matching concept's consolidated value, strictly.

    Tries each concept in `concepts` in order and returns the value of the
    first one that resolves to a real consolidated figure. For each concept,
    selects the single row where:
      - concept matches exactly,
      - dimension is False (undimensioned consolidated total — excludes
        per-segment / per-geography breakdown rows sharing the concept), and
      - abstract is False (excludes section-header rows).
    Returns None only if NONE of the candidate concepts resolve — i.e. the
    company tagged none of the true-synonym variants. Never returns a
    substituted or dimensional value.

    Args:
        df: The statement dataframe.
        concepts: Ordered list of exact XBRL concept identifiers to try.
        value_col: The period column to read the value from.

    Returns:
        The value as a raw-dollar float, or None if no candidate resolves.
    """
    for concept in concepts:
        rows = df[
            (df["concept"] == concept)
            & (df["dimension"] == False)  # noqa: E712 — pandas needs ==, not `is`
            & (df["abstract"] == False)  # noqa: E712
        ]
        if rows.empty:
            continue
        value = rows.iloc[0][value_col]
        if value is None or pd.isna(value):
            continue
        return float(value)
    return None


def capture_ground_truth(ticker: str, year: int) -> dict | None:
    """
    Capture XBRL ground-truth financials for one company-year 10-K, strictly.

    Locates the SAME filing acquisition uses, then for each of the five fields
    reads its exact XBRL concept from the relevant statement — undimensioned,
    consolidated, for the filing's own period — storing the value if tagged and
    null if not. No fuzzy accessors, no derivation: a field the company did not
    tag is recorded null (excluded as n/a by the evaluator), never substituted.

    Never raises: any failure is logged and returns None so one bad
    company-year cannot interrupt corpus-wide capture.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        year: Calendar year in which the target fiscal period ends.

    Returns:
        A dict with company_name, cik (zero-padded), fiscal_year,
        fiscal_year_end, and the five financial fields (each float | None);
        or None if the filing or its statements cannot be obtained.
    """
    logger.info("capture_ground_truth_start", ticker=ticker, year=year)
    try:
        filing = select_10k_for_fiscal_year(ticker, year)
        if filing is None:
            logger.warning(
                "capture_ground_truth_not_found", ticker=ticker, year=year
            )
            return None

        tenk = filing.obj()
        financials = getattr(tenk, "financials", None) if tenk else None
        if financials is None:
            logger.error(
                "capture_ground_truth_no_financials", ticker=ticker, year=year
            )
            return None

        period_of_report = str(getattr(filing, "period_of_report", "")) or None

        # Render each needed statement once and cache its dataframe, so we read
        # each statement a single time even though three fields share one.
        statement_dfs: dict[str, pd.DataFrame | None] = {}

        def _statement_df(name: str) -> pd.DataFrame | None:
            if name not in statement_dfs:
                try:
                    stmt = getattr(financials, name)()
                    statement_dfs[name] = stmt.to_dataframe() if stmt else None
                except Exception as exc:
                    logger.debug(
                        "ground_truth_statement_failed",
                        statement=name, error=str(exc),
                    )
                    statement_dfs[name] = None
            return statement_dfs[name]

        record: dict = {
            "company_name": getattr(filing, "company", None) or ticker,
            "cik": normalize_cik(filing.cik),
            "fiscal_year": year,
            "fiscal_year_end": period_of_report,
        }

        for field, spec in _FIELD_SPECS.items():
            df = _statement_df(spec["statement"])
            if df is None or period_of_report is None:
                record[field] = None
                continue
            value_col = _period_column(df, period_of_report)
            if value_col is None:
                logger.warning(
                    "ground_truth_period_column_missing",
                    ticker=ticker, year=year,
                    statement=spec["statement"],
                    period=period_of_report,
                )
                record[field] = None
                continue
            record[field] = _read_concept(df, spec["concepts"], value_col)

        captured = sum(1 for f in _FIELD_SPECS if record[f] is not None)
        nulled = [f for f in _FIELD_SPECS if record[f] is None]
        logger.info(
            "capture_ground_truth_success",
            ticker=ticker, year=year, cik=record["cik"],
            metrics_captured=captured, metrics_total=len(_FIELD_SPECS),
            nulled_fields=nulled,
        )
        return record

    except Exception as exc:
        logger.error(
            "capture_ground_truth_error",
            ticker=ticker, year=year,
            error_type=type(exc).__name__, error=str(exc),
        )
        return None
def write_ground_truth(records: list[dict]) -> str:
    """
    Write ground-truth records to data/eval/ground_truth.json, keyed by
    {cik}_{year}, MERGING with any existing file.

    Merge rather than overwrite: the acquisition pipeline skips
    already-acquired documents, so a re-run's `records` list holds only newly
    captured company-years. Overwriting would erase the ground truth for every
    previously captured document. This file is committed portfolio evidence,
    so silent data loss on re-run would be expensive.

    Args:
        records: Dicts as returned by capture_ground_truth; each must contain
                 "cik" and "fiscal_year". Falsy entries or entries missing
                 those keys are skipped (logged).

    Returns:
        The path to the written ground_truth.json file.

    Raises:
        OSError: If the eval directory cannot be created or the file written.
    """
    eval_dir = Path(settings.EVAL_DATA_PATH)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "ground_truth.json"

    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ground_truth_existing_unreadable", error=str(exc))
            existing = {}

    merged = 0
    for record in records:
        if not record:
            continue
        cik = record.get("cik")
        fiscal_year = record.get("fiscal_year")
        if cik is None or fiscal_year is None:
            logger.warning("ground_truth_record_unkeyable", record=record)
            continue
        existing[f"{cik}_{fiscal_year}"] = record
        merged += 1

    out_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "ground_truth_written",
        path=str(out_path), records_merged=merged, total_records=len(existing),
    )
    return str(out_path)