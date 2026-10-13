"""Execution-eval tests: oracle plans from labels, outcome classes, label alignment.

No API and no stores: outcomes are built by hand.
"""

import pytest

from src.eval.execution_eval import (
    OracleError,
    expected_by_key,
    is_executable,
    load_oracle_params,
    matched_alternative,
    oracle_plan,
    outcome_class,
)
from src.eval.query_eval import EvalQuestion, ObservedPlan, ObservedTask, load_eval_set
from src.query.schemas import (
    CitedClaim,
    Intent,
    MetricField,
    MetricResult,
    MetricRow,
    NarrativeRefusal,
    NarrativeResult,
    PlanType,
    ResolutionFailure,
    SourceRef,
    TaskOutcome,
    TaskStatus,
    Tool,
)
from src.storage.quality import QualityFlag, QualityReason, QualityVerdict


# --- Oracle plans ---------------------------------------------------------------


def test_every_executable_label_builds_a_valid_oracle_plan() -> None:
    """The instrument check: a label the oracle cannot execute would fail mid-run."""
    filters = load_oracle_params()
    executable = [q for q in load_eval_set().questions if is_executable(q)]
    assert len(executable) > 70
    for question in executable:
        plan = oracle_plan(question, filters)
        assert len(plan.tasks) == len(question.strict.tasks), question.id


def test_oracle_maps_label_companies_to_resolvable_mentions() -> None:
    questions = {q.id: q for q in load_eval_set().questions}
    filters = load_oracle_params()
    assert oracle_plan(questions["R12"], filters).tasks[0].company_mention.startswith("Nonexistent")
    assert oracle_plan(questions["R01"], filters).tasks[0].company_mention == "AAPL"
    r06 = oracle_plan(questions["R06"], filters).tasks[0]
    assert r06.threshold_text == "$50 billion" and r06.company_mention is None


def test_oracle_narrative_task_uses_the_original_question() -> None:
    question = {q.id: q for q in load_eval_set().questions}["R22"]
    task = oracle_plan(question, {}).tasks[0]
    assert task.question == question.question


def test_refusal_labels_are_not_executable() -> None:
    question = {q.id: q for q in load_eval_set().questions}["R45"]
    assert not is_executable(question)
    with pytest.raises(OracleError):
        oracle_plan(question, {})


# --- Outcome classes ---------------------------------------------------------------


def _flag(verdict: QualityVerdict) -> QualityFlag:
    return QualityFlag(field="total_assets", verdict=verdict, reason=list(QualityReason)[0])


def _sql(verdict: QualityVerdict) -> TaskOutcome:
    row = MetricRow(ticker="BAC", company_name="Bank of America", fiscal_year=2022,
                    fiscal_year_end=None, metric=MetricField.TOTAL_ASSETS, value=1.0,
                    quality=_flag(verdict))
    return TaskOutcome(task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL,
                       status=TaskStatus.SUCCESS, metric_result=MetricResult(rows=[row]))


def _rag(result: NarrativeResult, status: TaskStatus = TaskStatus.SUCCESS) -> TaskOutcome:
    return TaskOutcome(task_id="t1", intent=Intent.MARKET_RISK, tool=Tool.RAG,
                       status=status, narrative_result=result)


_CLAIM = CitedClaim(text="x", sources=[SourceRef(
    ticker="JPM", company_name="JPM", fiscal_year=2023, section="Item 1A",
    chunk_index=1, distance=0.4)])


@pytest.mark.parametrize("outcome,expected", [
    (_sql(QualityVerdict.VERIFIED), "answered"),
    (_sql(QualityVerdict.DISPUTED), "answered_flagged"),
    (_rag(NarrativeResult(claims=[_CLAIM])), "answered"),
    (_rag(NarrativeResult(claims=[_CLAIM], fallback_from=frozenset({"Item 7", "Item 7A"}))),
     "fallback_answered"),
    (_rag(NarrativeResult(refusal=NarrativeRefusal.SECTION_ABSENT), TaskStatus.REFUSED),
     "section_absent"),
    (TaskOutcome(task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL,
                 status=TaskStatus.REFUSED, resolution_failure=ResolutionFailure.YEAR_NOT_HELD),
     "year_not_held"),
    (TaskOutcome(task_id="t1", intent=Intent.FINANCIAL_FILTER, tool=Tool.SQL,
                 status=TaskStatus.REFUSED, message="bad threshold"), "sql_refused"),
])
def test_outcomes_reduce_to_the_label_vocabulary(outcome, expected) -> None:
    assert outcome_class(outcome) == expected


# --- Alignment with labels -----------------------------------------------------------


def _question(alternatives: list[dict]) -> EvalQuestion:
    return EvalQuestion(id="X", question="q", split="dev", source="t", stratum="s",
                        alternatives=alternatives)


def test_expected_outcomes_align_by_key_not_order() -> None:
    question = _question([{"plan_type": "hybrid", "tasks": [
        {"intent": "financial_metric", "company": "INTC", "metric": "net_income",
         "outcome": "answered_flagged"},
        {"intent": "management_commentary", "company": "INTC", "outcome": "section_absent"},
    ]}])
    observed = ObservedPlan(plan_type="hybrid", tasks=[
        ObservedTask(intent="management_commentary", mention="Intel", company="INTC"),
        ObservedTask(intent="financial_metric", mention="Intel", company="INTC",
                     metric="net_income"),
    ])
    alternative = matched_alternative(observed, question)
    assert alternative is not None
    assert expected_by_key(observed, alternative) == ["section_absent", "answered_flagged"]


def test_unmatched_plan_has_no_alternative() -> None:
    question = _question([{"plan_type": "single", "tasks": [
        {"intent": "risk_factors", "company": "GS"}]}])
    observed = ObservedPlan(plan_type=PlanType.OUT_OF_SCOPE.value)
    assert matched_alternative(observed, question) is None
