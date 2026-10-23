"""Executor guard: a ranking or filter that names a company becomes that company's lookup.

No API, no stores: resolution and the SQL task runner are replaced.
"""

import asyncio

import pytest

from src.query import executor
from src.query.executor import scoped_to_company
from src.query.schemas import (
    Comparison,
    Direction,
    Intent,
    MetricField,
    Resolution,
    ResolvedEntity,
    Task,
    TaskOutcome,
    TaskStatus,
    Tool,
)

APPLE = ResolvedEntity(mention="Apple", ticker="AAPL", cik="0000320193",
                       company_name="Apple Inc.", document_id=1, filing_year=2021)


def _ranking(mention: str | None) -> Task:
    return Task(task_id="t2", intent=Intent.FINANCIAL_RANKING, company_mention=mention,
                metric=MetricField.TOTAL_REVENUE, direction=Direction.HIGHEST, year=2021)


def test_ranking_naming_a_company_becomes_its_lookup() -> None:
    task = scoped_to_company(_ranking("Apple"))
    assert task.intent is Intent.FINANCIAL_METRIC
    assert (task.task_id, task.company_mention, task.metric, task.year) == (
        "t2", "Apple", MetricField.TOTAL_REVENUE, 2021)
    assert task.direction is None


def test_filter_naming_a_company_becomes_its_lookup() -> None:
    task = scoped_to_company(Task(
        task_id="t1", intent=Intent.FINANCIAL_FILTER, company_mention="Apple",
        metric=MetricField.NET_INCOME, comparison=Comparison.GT, threshold_text="$50 billion",
    ))
    assert task.intent is Intent.FINANCIAL_METRIC
    assert task.threshold_text is None and task.comparison is None


@pytest.mark.parametrize("mention", [None, "", "  "])
def test_corpus_wide_ranking_is_untouched(mention) -> None:
    task = _ranking(mention)
    assert scoped_to_company(task) is task


def test_executor_serves_the_named_company_not_the_corpus_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    def resolve(mention, requested_year=None, index=None, require_embedded=False,
                require_extracted=False):
        """Record the call; a SQL task must ask for an extracted filing."""
        seen["mention"] = mention
        assert require_extracted and not require_embedded
        return Resolution(entity=APPLE)

    def run_sql(task, entity):
        seen["intent"], seen["entity"] = task.intent, entity
        return TaskOutcome(task_id=task.task_id, intent=task.intent, tool=Tool.SQL,
                           status=TaskStatus.SUCCESS, message="ok")

    monkeypatch.setattr(executor, "resolve", resolve)
    monkeypatch.setattr(executor, "_run_sql_task", run_sql)
    outcome = asyncio.run(executor._execute_task(_ranking("Apple"), None))
    assert seen == {"mention": "Apple", "intent": Intent.FINANCIAL_METRIC, "entity": APPLE}
    assert outcome.intent is Intent.FINANCIAL_METRIC
