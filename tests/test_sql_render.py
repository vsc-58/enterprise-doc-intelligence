"""Which left-out companies an answer names, and how (D53, narrowed by D56).

A company a ranking or filter left out is named when:
  - its figure is scale-flagged AND a 1,000x scale correction would put it in
    the answer, or
  - its figure was not extracted (named with what that means for the question).

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
from src.query.sql_path import (
    SCALE_STEP,
    _flagged_outside_result,
    flagged_not_shown,
    plausible_values,
    render,
)
from src.storage.quality import QualityFlag, QualityReason, QualityVerdict


def _flag(reason: QualityReason, suspected: float | None = None) -> QualityFlag:
    verdict = QualityVerdict.VERIFIED if reason is QualityReason.CONSISTENT else QualityVerdict.DISPUTED
    return QualityFlag(field="x", verdict=verdict, reason=reason, suspected_true_value=suspected)


def _row(ticker: str, value: float | None, reason: QualityReason = QualityReason.CONSISTENT,
         metric: MetricField = MetricField.TOTAL_ASSETS, name: str | None = None,
         suspected: float | None = None) -> MetricRow:
    return MetricRow(ticker=ticker, company_name=name or ticker, fiscal_year=2022,
                     fiscal_year_end="2022-12-31", metric=metric, value=value,
                     quality=_flag(reason, suspected))


# Total assets, as stored
JPM = _row("JPM", 3.87e12)
GS = _row("GS", 1.46e12)
BAC = _row("BAC", 3.05e9, QualityReason.SCALE_AMBIGUOUS, name="Bank of America Corp.")

# Net income, as stored: Step 10's #11 case
NI = MetricField.NET_INCOME
AAPL_NI = _row("AAPL", 94.68e9, metric=NI)
BAC_NI = _row("BAC", 27.53e9, QualityReason.SCALE_AMBIGUOUS, metric=NI)
INTC_NI = _row("INTC", 1.689e9, QualityReason.EVIDENCE_MISMATCH, metric=NI)
TSLA_NI = _row("TSLA", 12.556e9, QualityReason.EVIDENCE_MISMATCH, metric=NI)

# Operating cash flow: GS stored 1,000x small, with a suspected true value
OCF = MetricField.OPERATING_CASH_FLOW
AAPL_OCF = _row("AAPL", 104.0e9, metric=OCF)
GS_OCF = _row("GS", 921e3, QualityReason.SCALE_DISAGREEMENT, metric=OCF, suspected=921e6)
BAC_OCF_NULL = _row("BAC", None, QualityReason.NULL_FIELD, metric=OCF, name="Bank of America Corp.")


def _filter(metric: MetricField = MetricField.TOTAL_ASSETS,
            comparison: Comparison = Comparison.GT) -> Task:
    return Task(task_id="t1", intent=Intent.FINANCIAL_FILTER, metric=metric,
                comparison=comparison, threshold_text="x")


def _ranking(direction: Direction, metric: MetricField = OCF, limit: int = 1) -> Task:
    return Task(task_id="t1", intent=Intent.FINANCIAL_RANKING, metric=metric,
                direction=direction, limit=limit)


# --- plausible_values ------------------------------------------------------------------


def test_scale_step_is_one_printed_scale() -> None:
    assert SCALE_STEP == 1000


def test_even_split_tries_one_step_up_and_down() -> None:
    assert plausible_values(BAC) == [3.05e12, 3.05e6]


def test_clear_majority_uses_the_suspected_value() -> None:
    assert plausible_values(GS_OCF) == [921e6]


# --- Who is named: filters ------------------------------------------------------------------


def test_scale_flagged_company_a_correction_would_bring_in_is_named() -> None:
    """BAC at $3.05 billion misses $1 trillion; at x1,000 ($3.05 trillion) it passes."""
    assert flagged_not_shown([JPM, GS, BAC], [JPM, GS], _filter(), 1e12) == [BAC]


def test_scale_flagged_company_no_correction_would_bring_in_is_not_named() -> None:
    """Under $5 trillion neither reading of BAC passes: naming it was over-disclosure."""
    assert flagged_not_shown([JPM, GS, BAC], [], _filter(), 5e12) == []


def test_unconfirmed_but_not_mis_scaled_figures_are_not_named() -> None:
    """Step 10's #11: Intel ($1.69bn) and Tesla ($12.56bn) were named under '$50 billion'."""
    named = flagged_not_shown([AAPL_NI, BAC_NI, INTC_NI, TSLA_NI], [AAPL_NI], _filter(NI), 50e9)
    assert named == [BAC_NI]                                  # BAC x1,000 = $27.5 trillion passes


def test_less_than_filter_checks_the_smaller_reading() -> None:
    """At /1,000 BAC's $3.05 billion is $3.05 million, which passes 'below $1 billion'."""
    assert flagged_not_shown([JPM, BAC], [], _filter(comparison=Comparison.LT), 1e9) == [BAC]


def test_verified_and_shown_companies_are_never_named() -> None:
    assert flagged_not_shown([JPM, GS, BAC], [JPM, GS, BAC], _filter(), 1e12) == []


def test_named_companies_are_ordered_by_ticker() -> None:
    other = _row("AMZN", 3.0e9, QualityReason.SCALE_AMBIGUOUS)
    named = flagged_not_shown([BAC, other], [], _filter(), 1e12)
    assert [r.ticker for r in named] == ["AMZN", "BAC"]


# --- Who is named: rankings -----------------------------------------------------------------


def test_ranking_names_only_a_company_that_would_beat_the_last_row_shown() -> None:
    """GS's likely $921 million cash flow would not beat Apple's $104 billion."""
    assert flagged_not_shown([AAPL_OCF, GS_OCF], [AAPL_OCF], _ranking(Direction.HIGHEST), None) == []


def test_ranking_names_a_company_whose_correction_would_take_the_top_spot() -> None:
    small_leader = _row("XOM", 500e6, metric=OCF)
    assert flagged_not_shown([small_leader, GS_OCF], [small_leader],
                             _ranking(Direction.HIGHEST), None) == [GS_OCF]


def test_ranking_shorter_than_its_limit_names_any_scale_flagged_company() -> None:
    """With room left in the list, any corrected value would be listed."""
    task = _ranking(Direction.HIGHEST, limit=5)
    assert flagged_not_shown([AAPL_OCF, GS_OCF], [AAPL_OCF], task, None) == [GS_OCF]


# --- Wording -----------------------------------------------------------------------------------


def test_flagged_line_says_why_it_may_belong() -> None:
    result = MetricResult(rows=[JPM, GS], flagged_not_shown=[BAC], threshold_value=1e12)
    text = render(_filter(), result, None)
    assert ("Not included: Bank of America Corp. (BAC) — its total assets are flagged, "
            "and at a 1,000x different scale it would be in this list.") in text
    assert "two different scales" in text                    # the D29 note travels with it


def test_missing_figure_in_a_lowest_ranking_may_be_lower() -> None:
    """Step 10's #4: BAC's cash flow is missing because it is negative — probably the answer."""
    result = MetricResult(rows=[GS_OCF], excluded_null=[BAC_OCF_NULL], excluded_null_count=1)
    text = render(_ranking(Direction.LOWEST), result, None)
    assert ("Not included: Bank of America Corp. (BAC) — its operating cash flow was not "
            "extracted from the filing, so it may be lower than any figure shown.") in text
    assert "1 company was excluded" not in text               # the old, unnamed count


def test_missing_figure_in_a_highest_ranking_may_be_higher() -> None:
    result = MetricResult(rows=[AAPL_OCF], excluded_null=[BAC_OCF_NULL], excluded_null_count=1)
    assert "so it may be higher than any figure shown." in render(_ranking(Direction.HIGHEST), result, None)


def test_missing_figure_in_a_filter_is_unknown() -> None:
    task = _filter(OCF)
    result = MetricResult(rows=[AAPL_OCF], excluded_null=[BAC_OCF_NULL], threshold_value=50e9)
    assert "so whether it meets this threshold is unknown." in render(task, result, None)


def test_plural_figure_takes_plural_verb() -> None:
    bac_assets_null = _row("BAC", None, QualityReason.NULL_FIELD, name="Bank of America Corp.")
    result = MetricResult(rows=[JPM], excluded_null=[bac_assets_null], threshold_value=1e12)
    assert "its total assets were not extracted" in render(_filter(), result, None)


def test_empty_result_still_names_left_out_companies() -> None:
    result = MetricResult(rows=[], flagged_not_shown=[BAC], excluded_null=[BAC_OCF_NULL],
                          threshold_value=1e12)
    text = render(_filter(), result, None)
    assert text.startswith("No company in the corpus matches that total assets criterion.")
    assert text.count("Not included:") == 2


def test_shown_flag_keeps_the_ordering_caveat_and_left_out_flag_does_not_add_it() -> None:
    """The D36 caveat speaks of figures 'here'; it stays tied to shown rows."""
    result = MetricResult(rows=[JPM, GS], flagged_not_shown=[BAC], threshold_value=1e12)
    assert "at least one figure here is flagged" not in render(_filter(), result, None)


def test_answer_with_nothing_left_out_is_unchanged() -> None:
    result = MetricResult(rows=[JPM, GS], threshold_value=1e12)
    assert "Not included" not in render(_filter(), result, None)


def test_single_company_lookup_never_queries_for_dropped_rows() -> None:
    lookup = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, company_mention="JPMorgan",
                  metric=MetricField.TOTAL_ASSETS)
    assert _flagged_outside_result(lookup, [JPM], None) == []
