"""mapping.py: plan + outcomes + answer -> the /query body, as pure functions.

Uses the real assemble(), so the answer text and source numbers in these tests
are exactly what the synthesizer produces.
"""

from src.api.mapping import build_query_response, nothing_answered, refusal_reason
from src.api.models import RefusalReason
from src.query.schemas import QueryPlan, TaskOutcome, TaskStatus
from src.query.synthesizer import assemble
from tests.query_samples import (
    APPLE_OUTCOME,
    APPLE_SENTENCE,
    APPLE_SQL_FAILED,
    APPLE_TASK,
    FILTER_TASK,
    FILTER_WITH_DROPPED_FLAG,
    INTEL_FIGURE,
    INTEL_FIGURE_TASK,
    INTEL_MARGIN_TASK,
    INTEL_RAG_FAILED,
    INTEL_SECTION_ABSENT,
    NVIDIA_REFUSED,
    SURVEY_OUTCOME,
    SURVEY_TASK,
    THRESHOLD_REFUSED,
    UNSUPPORTED_PLAN,
    WALMART_OUTCOME,
    WALMART_TASK,
    YEAR_REFUSED,
    hybrid,
    single,
)


def _build(plan: QueryPlan, outcomes: list[TaskOutcome]):
    answer = assemble(plan, outcomes)
    return build_query_response(
        "q", plan, outcomes, answer,
        planner_version="p", chain_version="c", request_id="r", latency_ms=12,
    )


# --- Answers ------------------------------------------------------------------------


def test_figure_answer_carries_value_and_quality() -> None:
    body = _build(single(APPLE_TASK), [APPLE_OUTCOME])
    assert body.answer == APPLE_SENTENCE                     # text passed through unchanged
    figure = body.tasks[0].figures[0]
    assert (figure.ticker, figure.value, figure.fiscal_year_end) == ("AAPL", 365817000000.0, "2021-09-25")
    assert figure.quality.verdict.value == "verified"
    assert figure.quality.note is None
    assert body.tasks[0].refusal_reason is None
    assert body.sources == []


def test_narrative_sources_match_the_answer_text() -> None:
    body = _build(single(WALMART_TASK), [WALMART_OUTCOME])
    source = body.sources[0]
    assert source.number == 1
    assert source.label == "[1] Walmart Inc., FY2023, MD&A (Item 7), chunk 25"
    assert source.label in body.answer                       # same line as in the Sources list
    assert "[1]" in body.answer


def test_filter_reports_dropped_flagged_company_and_counts() -> None:
    body = _build(single(FILTER_TASK), [FILTER_WITH_DROPPED_FLAG])
    task = body.tasks[0]
    assert task.figures == []
    assert [f.ticker for f in task.flagged_not_shown] == ["BAC"]
    assert task.flagged_not_shown[0].quality.reason.value == "scale_ambiguous"
    assert "two different scales" in task.flagged_not_shown[0].quality.note
    assert task.excluded_null_count == 1


def test_survey_findings_use_display_names() -> None:
    body = _build(single(SURVEY_TASK), [SURVEY_OUTCOME])
    findings = {f.ticker: f for f in body.tasks[0].findings}
    assert findings["GS"].company_name == "The Goldman Sachs Group, Inc."
    assert findings["GS"].status.value == "failed"           # one company, not the survey
    assert body.tasks[0].status is TaskStatus.SUCCESS


def test_taskless_plan_has_no_tasks_and_keeps_its_reason() -> None:
    body = _build(UNSUPPORTED_PLAN, [])
    assert body.tasks == []
    assert body.unsupported_reason.value == "aggregation"
    assert "aggregates" in body.answer


def test_missing_outcome_surfaces_as_failed_task() -> None:
    """The executor never drops a task; if it ever did, it must not vanish."""
    body = _build(single(APPLE_TASK), [])
    assert body.tasks[0].status is TaskStatus.FAILED


# --- refusal_reason (contract §3.1) -----------------------------------------------------


def test_resolver_refusals_map_to_their_own_code() -> None:
    assert refusal_reason(YEAR_REFUSED) is RefusalReason.YEAR_NOT_HELD
    assert refusal_reason(NVIDIA_REFUSED) is RefusalReason.NON_CORPUS_COMPANY


def test_narrative_refusal_maps_to_its_code() -> None:
    assert refusal_reason(INTEL_SECTION_ABSENT) is RefusalReason.SECTION_ABSENT


def test_untyped_sql_filter_refusal_is_threshold_unparsed() -> None:
    assert refusal_reason(THRESHOLD_REFUSED) is RefusalReason.THRESHOLD_UNPARSED


def test_untyped_sql_lookup_refusal_is_sql_not_runnable() -> None:
    lookup = THRESHOLD_REFUSED.model_copy(update={"intent": APPLE_TASK.intent})
    assert refusal_reason(lookup) is RefusalReason.SQL_NOT_RUNNABLE


def test_failed_and_successful_tasks_have_no_refusal_reason() -> None:
    """Failing is not refusing: a failed narrative carries NarrativeRefusal.FAILED internally."""
    assert refusal_reason(INTEL_RAG_FAILED) is None
    assert refusal_reason(APPLE_OUTCOME) is None


# --- nothing_answered (contract §7) ------------------------------------------------------


def test_hybrid_with_one_answer_and_one_refusal_is_answered() -> None:
    """R38: Intel's figure answered, its MD&A question refused -> 200."""
    plan = hybrid(INTEL_FIGURE_TASK, INTEL_MARGIN_TASK)
    assert not nothing_answered(assemble(plan, [INTEL_FIGURE, INTEL_SECTION_ABSENT]))


def test_refused_plus_failed_is_answered() -> None:
    """The refused part is a correct answer; the failed part says it could not be answered."""
    plan = hybrid(INTEL_FIGURE_TASK, INTEL_MARGIN_TASK)
    refused_figure = YEAR_REFUSED.model_copy(update={"task_id": "t1"})
    assert not nothing_answered(assemble(plan, [refused_figure, INTEL_RAG_FAILED]))


def test_every_task_failed_is_nothing_answered() -> None:
    plan = hybrid(INTEL_FIGURE_TASK, INTEL_MARGIN_TASK)
    assert nothing_answered(assemble(plan, [APPLE_SQL_FAILED, INTEL_RAG_FAILED]))


def test_taskless_plan_is_an_answer() -> None:
    assert not nothing_answered(assemble(UNSUPPORTED_PLAN, []))


def test_refused_task_without_any_reason_maps_to_none() -> None:
    """Defensive: the executor never does this, but the API must not crash if it did."""
    odd = INTEL_SECTION_ABSENT.model_copy(update={"narrative_result": None})
    assert refusal_reason(odd) is None
