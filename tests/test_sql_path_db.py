"""The SQL path against a real SQLite database built for each test.

The SQL path was first checked only by the evaluation harness and end-to-end
scripts against the project database. These tests build a throwaway database
in pytest's tmp_path — no network, no API key, and never the project's
db/metadata.db — so the real SQL runs: the canonical latest-extraction rule,
the figure quality flags, the mixed-fiscal-years note, the ordering caveat for
flagged rows, and the query that names flagged companies a list left out.

Figures and cited lines follow the real filings' shapes (millions, BAC's mixed
balance-sheet scale, GS's thousand-times-small operating cash flow), so each
company's quality verdict is what the production data gives.
"""

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.query.schemas import (
    Comparison,
    Direction,
    Intent,
    MetricField,
    ResolvedEntity,
    Task,
)
from src.query.sql_path import SqlPathError, render, run_financial_task
from src.storage import metadata_store
from src.storage.metadata_store import Base, Document, ExtractedRecord
from src.storage.quality import QualityReason, QualityVerdict

M = 1e6      # filings print millions
K = 1e3      # BAC's balance sheet, GS's cash flow line: thousands


def _record(doc: Document, when: datetime, status: str = "success", **fields: object) -> ExtractedRecord:
    """An extraction attempt for a document at a given time; figures passed as fields."""
    return ExtractedRecord(
        document_id=doc.id, extraction_strategy_used="S3", prompt_version="v1",
        model_name="gpt-4o-mini", extraction_status=status, extracted_at=when, **fields,
    )


def _figures(rev: float, ni: float, assets: float, liab: float, ocf: float | None,
             scale: tuple[float, float, float, float, float] = (M, M, M, M, M)) -> dict[str, object]:
    """Values in actual dollars, each with a cited line printing it at its scale."""
    names = ["total_revenue", "net_income", "total_assets", "total_liabilities", "operating_cash_flow"]
    out: dict[str, object] = {}
    for name, value, unit in zip(names, (rev, ni, assets, liab, ocf), scale):
        out[name] = None if value is None else value * unit
        out[f"{name}_source_line"] = None if value is None else f"{name} {value:,.0f}"
    return out


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Document]:
    """A four-company database; get_session() is pointed at it for this test only."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(metadata_store, "_SessionFactory", factory)

    docs = {
        "AAPL": Document(ticker="AAPL", company_name="Apple Inc.", cik="0000320193", filing_year=2021),
        "JPM": Document(ticker="JPM", company_name="JPMORGAN CHASE & CO", cik="0000019617", filing_year=2023),
        "GS": Document(ticker="GS", company_name="GOLDMAN SACHS GROUP INC", cik="0000886982", filing_year=2021),
        "BAC": Document(ticker="BAC", company_name="BANK OF AMERICA CORP /DE/", cik="0000070858", filing_year=2022),
    }
    t0, t1, t2 = datetime(2026, 8, 1), datetime(2026, 9, 1), datetime(2026, 9, 15)
    with factory() as session:
        session.add_all(docs.values())
        session.flush()
        session.add_all([
            # Apple: an OLDER success with a wrong value, the CURRENT success,
            # and a NEWER failed attempt. Serve the latest SUCCESS only.
            _record(docs["AAPL"], t0, fiscal_year_end="2021-09-25", **_figures(1, 1, 1, 1, 1)),
            _record(docs["AAPL"], t1, fiscal_year_end="2021-09-25",
                    **_figures(365_817, 94_680, 351_002, 287_912, 104_038)),
            _record(docs["AAPL"], t2, status="failed", fiscal_year_end="2021-09-25"),
            _record(docs["JPM"], t1, fiscal_year_end="2023-12-31",
                    **_figures(128_695, 37_676, 3_875_393, 3_548_620, 12_974)),
            # GS: operating cash flow cited at thousands while the rest is millions.
            _record(docs["GS"], t1, fiscal_year_end="2021-12-31",
                    **_figures(59_339, 21_635, 1_463_988, 1_353_998, 921, scale=(M, M, M, M, K))),
            # BAC: two fields at millions, two at thousands, cash flow missing: a 2-2 tie.
            _record(docs["BAC"], t1, fiscal_year_end="2022-12-31",
                    **_figures(94_950, 27_528, 3_051_375, 2_779_999, None, scale=(M, M, K, K, M))),
        ])
        session.commit()
    return docs


def _apple(db: dict[str, Document]) -> ResolvedEntity:
    doc = db["AAPL"]
    return ResolvedEntity(mention="Apple", ticker="AAPL", cik=doc.cik, company_name="Apple Inc.",
                          document_id=doc.id, filing_year=2021)


def _tickers(rows) -> list[str]:
    return [row.ticker for row in rows]


# --- Lookup -------------------------------------------------------------------------------


def test_lookup_serves_the_latest_successful_extraction(db) -> None:
    """Not the older success (value 1), not the newer failed attempt (no value)."""
    task = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, company_mention="Apple",
                metric=MetricField.TOTAL_REVENUE)
    result = run_financial_task(task, _apple(db))
    assert [row.value for row in result.rows] == [365_817 * M]
    assert result.rows[0].quality.verdict is QualityVerdict.VERIFIED
    assert result.flagged_not_shown == []                     # lookups never look for dropped rows
    assert render(task, result, _apple(db)) == (
        "Apple Inc.'s total revenue for fiscal year 2021 (period ending 2021-09-25) "
        "was $365.82 billion."
    )


def test_lookup_without_a_company_is_refused(db) -> None:
    task = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, metric=MetricField.TOTAL_REVENUE)
    with pytest.raises(SqlPathError):
        run_financial_task(task, None)


# --- Filter: flagged companies left out, mixed years -----------------------------------


def test_filter_names_the_flagged_company_it_left_out(db) -> None:
    task = Task(task_id="t1", intent=Intent.FINANCIAL_FILTER, metric=MetricField.TOTAL_ASSETS,
                comparison=Comparison.GT, threshold_text="$1 trillion")
    result = run_financial_task(task, None)
    assert _tickers(result.rows) == ["JPM", "GS"]             # ordered by value, descending
    assert _tickers(result.flagged_not_shown) == ["BAC"]
    assert result.flagged_not_shown[0].quality.reason is QualityReason.SCALE_AMBIGUOUS
    assert result.threshold_value == 1e12
    assert result.mixed_years                                 # JPM FY2023, GS FY2021
    text = render(task, result, None)
    assert ("Not included: Bank of America Corp. (BAC) — its total assets are flagged, "
            "and at a 1,000x different scale it would be in this list.") in text
    assert "not like-for-like" in text


def test_filter_threshold_in_words_is_refused_not_guessed(db) -> None:
    task = Task(task_id="t1", intent=Intent.FINANCIAL_FILTER, metric=MetricField.TOTAL_ASSETS,
                comparison=Comparison.GT, threshold_text="fifty billion dollars")
    with pytest.raises(SqlPathError):
        run_financial_task(task, None)


def test_filter_no_correction_could_pass_names_nobody(db) -> None:
    """BAC is $3.05 trillion at most: it cannot pass $5 trillion, so it is not named."""
    task = Task(task_id="t1", intent=Intent.FINANCIAL_FILTER, metric=MetricField.TOTAL_ASSETS,
                comparison=Comparison.GT, threshold_text="$5 trillion")
    result = run_financial_task(task, None)
    assert result.rows == [] and result.flagged_not_shown == []
    assert render(task, result, None) == "No company in the corpus matches that total assets criterion."


# --- Ranking: ordering caveat, null exclusion ------------------------------------------


def test_highest_cash_flow_names_the_null_but_not_the_flag_that_could_not_win(db) -> None:
    """BAC's cash flow is missing (named); GS's likely $921 million cannot beat Apple (not named)."""
    task = Task(task_id="t1", intent=Intent.FINANCIAL_RANKING,
                metric=MetricField.OPERATING_CASH_FLOW, direction=Direction.HIGHEST, limit=1)
    result = run_financial_task(task, None)
    assert _tickers(result.rows) == ["AAPL"]
    assert _tickers(result.excluded_null) == ["BAC"] and result.excluded_null_count == 1
    assert result.flagged_not_shown == []
    assert "so it may be higher than any figure shown." in render(task, result, None)


def test_lowest_cash_flow_puts_the_flagged_figure_first_and_says_so(db) -> None:
    """GS sorts first only because its value is 1,000x too small, so the ordering is caveated."""
    task = Task(task_id="t1", intent=Intent.FINANCIAL_RANKING,
                metric=MetricField.OPERATING_CASH_FLOW, direction=Direction.LOWEST, limit=1)
    result = run_financial_task(task, None)
    assert _tickers(result.rows) == ["GS"]
    assert result.rows[0].quality.reason is QualityReason.SCALE_DISAGREEMENT
    assert result.flagged_not_shown == []                     # shown, so disclosed beside it
    text = render(task, result, None)
    assert "the ordering itself may be wrong" in text
    assert ("Not included: Bank of America Corp. (BAC) — its operating cash flow was not "
            "extracted from the filing, so it may be lower than any figure shown.") in text


def test_ranking_orders_every_company_by_value(db) -> None:
    task = Task(task_id="t1", intent=Intent.FINANCIAL_RANKING, metric=MetricField.TOTAL_REVENUE,
                direction=Direction.HIGHEST, limit=20)
    result = run_financial_task(task, None)
    assert _tickers(result.rows) == ["AAPL", "JPM", "BAC", "GS"]
