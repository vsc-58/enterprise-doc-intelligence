# src/query/executor.py
# Module: Plan execution
# Purpose: Validate a plan, resolve its entities, run its tasks concurrently, and
#          return one typed outcome per task.
# Depends on: asyncio, src.query.*, src.utils.logger
#
# Deterministic. No LLM call happens here: the planner has already run, the tools
# are called with validated parameters, and nothing in this module decides what a
# question means.

import asyncio

from src.query.intents import policy_for
from src.query.resolve import CorpusIndex, resolve
from src.query.schemas import (
    Intent,
    PlanType,
    QueryPlan,
    ResolvedEntity,
    Task,
    TaskOutcome,
    TaskStatus,
    Tool,
)
from src.query.sql_path import SqlPathError, render, run_financial_task
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Intents that range over the whole corpus. A ranking or filter carries no company
# mention by design, so resolution must not run for them — doing so would fail
# every ranking with "no company named".
CORPUS_WIDE_INTENTS = frozenset({Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER})


def _resolve_for(task: Task, index: CorpusIndex | None):
    """
    Resolve a task's company, if its intent needs one.

    Args:
        task: The task.
        index: Corpus index, loaded from the database when omitted.

    Returns:
        A Resolution, or None when the intent is corpus-wide.
    """
    if task.intent in CORPUS_WIDE_INTENTS:
        return None
    return resolve(
        task.company_mention,
        requested_year=task.year,
        index=index,
        require_embedded=policy_for(task.intent).tool is Tool.RAG,
    )


def _run_sql_task(task: Task, entity: ResolvedEntity | None) -> TaskOutcome:
    """
    Run one financial task and render its answer.

    Args:
        task: A SQL-tool task.
        entity: Its resolved company, or None for corpus-wide intents.

    Returns:
        A TaskOutcome. A task that cannot be expressed is REFUSED, not FAILED:
        a capability limit is a correct answer about the system, while FAILED
        means something broke.
    """
    try:
        result = run_financial_task(task, entity)
    except SqlPathError as exc:
        logger.info("sql_task_refused", task_id=task.task_id, reason=exc.message)
        return TaskOutcome(
            task_id=task.task_id, intent=task.intent, tool=Tool.SQL,
            status=TaskStatus.REFUSED, message=exc.message,
        )
    except Exception as exc:  # noqa: BLE001 — one task must not sink the plan
        logger.error(
            "sql_task_failed", task_id=task.task_id,
            error_type=type(exc).__name__, error=str(exc),
        )
        return TaskOutcome(
            task_id=task.task_id, intent=task.intent, tool=Tool.SQL,
            status=TaskStatus.FAILED,
            message="I couldn't retrieve that figure from the structured store.",
        )

    return TaskOutcome(
        task_id=task.task_id, intent=task.intent, tool=Tool.SQL,
        status=TaskStatus.SUCCESS, metric_result=result,
        message=render(task, result, entity),
    )


async def _execute_task(task: Task, index: CorpusIndex | None) -> TaskOutcome:
    """
    Execute one task: enforce its tool, resolve its entity, dispatch it.

    The tool comes from the policy table, never from the plan — the planner emits
    intents and the application maps them, so a plan cannot route itself (D25).

    Blocking work runs in a worker thread, so a plan's tasks genuinely overlap and
    an async API handler is never stalled by a synchronous database or vector-store
    call.

    Args:
        task: The task to execute.
        index: Corpus index, loaded from the database when omitted.

    Returns:
        A TaskOutcome.
    """
    tool = policy_for(task.intent).tool

    resolution = await asyncio.to_thread(_resolve_for, task, index)
    if resolution is not None and resolution.entity is None:
        logger.info(
            "task_resolution_failed",
            task_id=task.task_id, failure=resolution.failure.value,
        )
        return TaskOutcome(
            task_id=task.task_id, intent=task.intent, tool=tool,
            status=TaskStatus.REFUSED,
            resolution_failure=resolution.failure,
            message=resolution.detail,
        )

    entity = resolution.entity if resolution else None

    if tool is Tool.SQL:
        return await asyncio.to_thread(_run_sql_task, task, entity)

    # 5B replaces this with the retrieval chain.
    return TaskOutcome(
        task_id=task.task_id, intent=task.intent, tool=Tool.RAG,
        status=TaskStatus.REFUSED,
        message="Narrative retrieval is not wired up yet.",
    )


async def execute_plan(plan: QueryPlan, index: CorpusIndex | None = None) -> list[TaskOutcome]:
    """
    Execute every task in a plan concurrently.

    Args:
        plan: A validated plan.
        index: Corpus index, loaded from the database when omitted.

    Returns:
        One outcome per task, in plan order. Empty for the three refusal plan
        types, which are answers in themselves and have nothing to execute.
    """
    if plan.plan_type in {PlanType.CLARIFICATION, PlanType.UNSUPPORTED, PlanType.OUT_OF_SCOPE}:
        return []

    outcomes = await asyncio.gather(*(_execute_task(task, index) for task in plan.tasks))
    logger.info(
        "plan_executed",
        plan_type=plan.plan_type.value,
        statuses=[outcome.status.value for outcome in outcomes],
    )
    return list(outcomes)