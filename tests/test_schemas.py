"""Contract tests: the enums, the policy table and the plan validators."""

import pytest
from pydantic import ValidationError

from src.query.intents import CORPUS_SECTIONS, INTENT_POLICY
from src.query.schemas import (
    Intent,
    MetricField,
    PlanType,
    QueryPlan,
    Task,
    Tool,
    UnsupportedReason,
)
from src.storage.metadata_store import EVIDENCE_FIELDS


def test_every_intent_has_a_policy() -> None:
    assert set(INTENT_POLICY) == set(Intent)


def test_metric_fields_match_the_evidence_columns() -> None:
    assert {m.value for m in MetricField} == set(EVIDENCE_FIELDS)


def test_rag_intents_search_known_sections() -> None:
    for intent, policy in INTENT_POLICY.items():
        if policy.tool is Tool.RAG:
            assert policy.sections, intent
            assert policy.sections <= CORPUS_SECTIONS, intent
        else:
            assert not policy.sections, intent


def test_no_two_rag_intents_share_a_section_set() -> None:
    """Two intents over the same union execute identically: merge them (D28, D38)."""
    seen: dict[frozenset[str], Intent] = {}
    for intent, policy in INTENT_POLICY.items():
        if policy.tool is not Tool.RAG:
            continue
        assert policy.sections not in seen, (
            f"{intent.value} duplicates {seen[policy.sections].value}"
        )
        seen[policy.sections] = intent


def _metric_task(task_id: str = "t1") -> Task:
    return Task(task_id=task_id, intent=Intent.FINANCIAL_METRIC,
                company_mention="Apple", metric=MetricField.TOTAL_REVENUE)


def test_financial_task_requires_a_metric() -> None:
    with pytest.raises(ValidationError):
        Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, company_mention="Apple")


def test_narrative_task_requires_a_question() -> None:
    with pytest.raises(ValidationError):
        Task(task_id="t1", intent=Intent.RISK_FACTORS, company_mention="Apple")


def test_single_plan_takes_exactly_one_task() -> None:
    QueryPlan(plan_type=PlanType.SINGLE, tasks=[_metric_task()])
    with pytest.raises(ValidationError):
        QueryPlan(plan_type=PlanType.SINGLE,
                  tasks=[_metric_task("t1"), _metric_task("t2")])


def test_plan_rejects_a_fourth_task() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(plan_type=PlanType.HYBRID,
                  tasks=[_metric_task(f"t{i}") for i in range(4)])


def test_plan_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(plan_type=PlanType.HYBRID,
                  tasks=[_metric_task("t1"), _metric_task("t1")])


def test_refusal_plans_carry_no_tasks_and_name_their_reason() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(plan_type=PlanType.UNSUPPORTED, tasks=[_metric_task()],
                  unsupported_reason=UnsupportedReason.AGGREGATION)
    with pytest.raises(ValidationError):
        QueryPlan(plan_type=PlanType.UNSUPPORTED)
    with pytest.raises(ValidationError):
        QueryPlan(plan_type=PlanType.CLARIFICATION)


def test_corpus_membership_is_not_a_planner_judgment() -> None:
    """The planner is corpus-blind (D32), so its reasons exclude membership."""
    assert "non_corpus_company" not in {r.value for r in UnsupportedReason}