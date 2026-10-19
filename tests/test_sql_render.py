"""Flagged companies a ranking or filter left out are named (D36 extended).

No database and no API: the selection rule and the renderer are pure, and the
one database-backed helper is checked only on its no-query branch.
"""

from src.query.schemas import (
    Comparison,
    Direction,
    Intent,
    MetricField,
    MetricResult,
    MetricRow,
    Task,
)
from src.query.sql_path import _flagged_outside_result, flagged_not_shown, render
from src.storage.quality import QualityFlag, QualityReason, QualityVerdict

_VERIFIED = QualityFlag(field="total_assets", verdict=QualityVerdict.VERIFIED,
                        reason=QualityReason.CONSISTENT)
_AMBIGUOUS = QualityFlag(field="total_assets", verdict=QualityVerdict.DISPUTED,
                         reason=QualityReason.SCALE_AMBIGUOUS)


def _row(ticker: str, value: float, quality: QualityFlag = _VERIFIED) -> MetricRow:
    return MetricRow(ticker=ticker, company_name=ticker, fiscal_year=2022,
                     fiscal_year_end="2022-12-31", metric=MetricField.TOTAL_ASSETS,
                     value=value, quality=quality)


JPM = _row("JPM", 3.87e12)
GS = _row("GS", 1.46e12)
BAC = _row("BAC", 3.05e9, _AMBIGUOUS)          # true scale unknown: shown as $3.05 billion
AAPL = _row("AAPL", 3.51e11)

FILTER = Task(task_id="t1", intent=Intent.FINANCIAL_FILTER, metric=MetricField.TOTAL_ASSETS,
              comparison=Comparison.GT, threshold_text="$1 trillion")
RANKING = Task(task_id="t1", intent=Intent.FINANCIAL_RANKING, metric=MetricField.TOTAL_ASSETS,
               direction=Direction.HIGHEST, limit=2)
LOOKUP = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, metric=MetricField.TOTAL_ASSETS,
              company_mention="JPMorgan")


def test_selects_only_flagged_rows_that_are_not_shown() -> None:
    """Verified companies left out are not mentioned; flagged ones are."""
    assert flagged_not_shown([JPM, GS, BAC, AAPL], shown=[JPM, GS]) == [BAC]


def test_flagged_row_already_shown_is_not_repeated() -> None:
    """A shown flagged row is disclosed beside its figure, not a second time."""
    assert flagged_not_shown([JPM, BAC], shown=[JPM, BAC]) == []


def test_selection_is_ordered_by_ticker() -> None:
    """Stable text between runs, whatever order the database returns."""
    other = _row("AMZN", 1.0, _AMBIGUOUS)
    assert [r.ticker for r in flagged_not_shown([BAC, other], shown=[])] == ["AMZN", "BAC"]


def test_filter_answer_names_the_dropped_flagged_company() -> None:
    """The motivating case: BAC fails 'above $1 trillion' only because of its scale."""
    result = MetricResult(rows=[JPM, GS], flagged_not_shown=[BAC], threshold_value=1e12)
    text = render(FILTER, result, None)
    assert "2 companies had total assets above $1 trillion:" in text
    assert "BAC — not listed, but its total assets are flagged, so it may belong here." in text
    assert "two different scales" in text            # the D29 note travels with it


def test_ranking_answer_names_the_dropped_flagged_company() -> None:
    result = MetricResult(rows=[JPM, GS], flagged_not_shown=[BAC])
    assert "BAC — not listed" in render(RANKING, result, None)


def test_dropped_flag_alone_does_not_claim_a_shown_figure_is_flagged() -> None:
    """The ordering caveat speaks of figures 'here'; it stays tied to shown rows."""
    result = MetricResult(rows=[JPM, GS], flagged_not_shown=[BAC], threshold_value=1e12)
    assert "at least one figure here is flagged" not in render(FILTER, result, None)


def test_empty_filter_result_still_names_the_dropped_flagged_company() -> None:
    """No rows must not mean no disclosure."""
    result = MetricResult(rows=[], flagged_not_shown=[BAC], threshold_value=5e12)
    text = render(FILTER, result, None)
    assert text.startswith("No company in the corpus matches that total assets criterion.")
    assert "BAC — not listed" in text


def test_answer_without_dropped_flags_is_unchanged() -> None:
    """No new text when nothing flagged was left out."""
    result = MetricResult(rows=[JPM, GS], threshold_value=1e12)
    assert "not listed" not in render(FILTER, result, None)


def test_single_company_lookup_never_queries_for_dropped_rows() -> None:
    """A lookup cannot drop a company, so the helper returns before any database read."""
    assert _flagged_outside_result(LOOKUP, [JPM]) == []
