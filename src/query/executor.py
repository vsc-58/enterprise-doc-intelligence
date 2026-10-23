"""Plan execution: run a validated plan's tasks and return one outcome per task.

Validates the plan, resolves each task's company, maps each intent to its tool
through the policy table, runs the tasks concurrently, and returns one typed
TaskOutcome per task.

No LLM call is made here directly: the planner has already run, the SQL tool is
called with validated parameters, and the narrative tool's model calls live in
src.rag.chain. Nothing in this module decides what a question means.
"""

import asyncio

from src.query.intents import policy_for
from src.query.resolve import CorpusIndex, load_index, resolve
from src.query.schemas import (
    Intent,
    NarrativeRefusal,
    NarrativeResult,
    PlanType,
    QueryPlan,
    ResolvedEntity,
    Task,
    TaskOutcome,
    TaskStatus,
    Tool,
)
from src.query.sql_path import SqlPathError, render, run_financial_task
from src.rag.chain import (
    answer_for_company,
    fallback_note,
    refusal_message,
    survey_companies,
)
from src.rag.retriever import CompanyRef
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

# Intents that range over the whole corpus. A ranking or filter carries no company
# mention by design, so resolution must not run for them — doing so would fail
# every ranking with "no company named".
CORPUS_WIDE_INTENTS = frozenset({Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER})

CORPUS_SUBJECT = "the companies in this corpus"


def is_corpus_wide(task: Task) -> bool:
    """Whether a task ranges over the whole corpus rather than one company.

    SQL rankings and filters always do. A narrative task does when it names no
    company: "Which companies flagged interest rate risk?" carries its scope in
    the missing mention, so this is an executor rule rather than a planner
    field: a field would add prompt surface, and change the planner version,
    only to say the same thing.

    Args:
        task: The task.

    Returns:
        True for corpus-wide tasks.
    """
    if task.intent in CORPUS_WIDE_INTENTS:
        return True
    return policy_for(task.intent).tool is Tool.RAG and not (task.company_mention or "").strip()


def _resolve_for(task: Task, index: CorpusIndex | None):
    """Resolve a task's company, if it needs one.

    Args:
        task: The task.
        index: Corpus index, loaded from the database when omitted.

    Returns:
        A Resolution, or None when the task is corpus-wide.
    """
    if is_corpus_wide(task):
        return None
    return resolve(
        task.company_mention,
        requested_year=task.year,
        index=index,
        require_embedded=policy_for(task.intent).tool is Tool.RAG,
    )


def _run_sql_task(task: Task, entity: ResolvedEntity | None) -> TaskOutcome:
    """Run one financial task and render its answer.

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


def _narrative_outcome(task: Task, result: NarrativeResult, subject: str) -> TaskOutcome:
    """Wrap a NarrativeResult as a TaskOutcome, rendering any refusal sentence.

    A narrative FAILED is a fault and maps to FAILED; every other refusal is a
    correct statement about the corpus and maps to REFUSED.

    When a single-company answer came from fallback sections, the fixed
    disclosure becomes the outcome's message: shown before the answer on
    success, and before the refusal sentence when the fallback text did not
    address the question. Not added to FAILED or FIGURE_REQUESTED, where the
    section searched is not what the reader needs to know.

    Args:
        task: The narrative task.
        result: The chain's result.
        subject: Display name of the company, or CORPUS_SUBJECT.

    Returns:
        The outcome.
    """
    note = (
        fallback_note(
            task.intent, subject, result.fallback_from, result.sections_searched,
            answered=result.refusal is None,
        )
        if result.fallback_used
        and result.refusal not in (NarrativeRefusal.FAILED, NarrativeRefusal.FIGURE_REQUESTED)
        else None
    )
    if result.refusal is None:
        status, message = TaskStatus.SUCCESS, note
    else:
        status = (
            TaskStatus.FAILED if result.refusal is NarrativeRefusal.FAILED
            else TaskStatus.REFUSED
        )
        if result.refusal is NarrativeRefusal.SECTION_ABSENT and subject == CORPUS_SUBJECT:
            message = "No company in this corpus has a captured section that covers this."
        else:
            message = refusal_message(result.refusal, subject, result.sections_searched)
        if note is not None:
            message = f"{note} {message}"
    return TaskOutcome(
        task_id=task.task_id, intent=task.intent, tool=Tool.RAG,
        status=status, narrative_result=result, message=message,
    )


async def _run_rag_task(
    task: Task,
    entity: ResolvedEntity | None,
    index: CorpusIndex | None,
) -> TaskOutcome:
    """Run one narrative task: one company's cited answer, or a corpus-wide survey.

    The chain never raises on retrieval or model faults; the guard here covers
    anything outside it — loading the corpus index for a survey — so one task
    still cannot sink the plan.

    Args:
        task: A RAG-tool task. task.question is the retrieval query.
        entity: Its resolved company, or None when corpus-wide.
        index: Corpus index, loaded from the database when omitted.

    Returns:
        A TaskOutcome carrying the NarrativeResult.
    """
    question = task.question or ""
    try:
        if entity is not None:
            company = CompanyRef(
                ticker=entity.ticker,
                company_name=entity.company_name,
                fiscal_year=entity.filing_year,
            )
            result = await answer_for_company(question, task.intent, company)
            return _narrative_outcome(
                task, result, display_name(entity.company_name, entity.ticker)
            )

        corpus = index or await asyncio.to_thread(load_index)
        companies = [
            CompanyRef(ticker=e.ticker, company_name=e.company_name, fiscal_year=e.filing_year)
            for e in corpus.entries
            if e.is_embedded
        ]
        result = await survey_companies(question, task.intent, companies)
        return _narrative_outcome(task, result, CORPUS_SUBJECT)
    except Exception as exc:  # noqa: BLE001 — one task must not sink the plan
        logger.error(
            "rag_task_failed", task_id=task.task_id, intent=task.intent.value,
            error_type=type(exc).__name__, error=str(exc),
        )
        return _narrative_outcome(
            task, NarrativeResult(refusal=NarrativeRefusal.FAILED),
            display_name(entity.company_name, entity.ticker) if entity else CORPUS_SUBJECT,
        )


def scoped_to_company(task: Task) -> Task:
    """Turn a ranking or filter that names a company into that company's lookup.

    A ranking or filter ranges over every company and ignores company_mention
    (is_corpus_wide), so a planner that expresses "which of Apple, Microsoft or
    Meta had the highest revenue?" as three rankings would get the corpus-wide
    winner three times — a correct figure for a company nobody asked about,
    looking fully verified. The contradiction is visible in the task itself, so
    it is fixed here in code rather than left to the prompt, where a rule would
    leave this wrong answer one paraphrase away. The named company's figure for
    the same metric and year is what such a task can honestly serve.

    Args:
        task: Any planned task.

    Returns:
        The task unchanged, or a financial_metric task for the named company.
    """
    if task.intent not in CORPUS_WIDE_INTENTS or not (task.company_mention or "").strip():
        return task
    logger.warning(
        "scoped_ranking_rewritten",
        task_id=task.task_id,
        from_intent=task.intent.value,
        company_mention=task.company_mention,
    )
    return Task(
        task_id=task.task_id,
        intent=Intent.FINANCIAL_METRIC,
        company_mention=task.company_mention,
        year=task.year,
        metric=task.metric,
    )


async def _execute_task(task: Task, index: CorpusIndex | None) -> TaskOutcome:
    """Execute one task: enforce its tool, resolve its entity, dispatch it.

    The tool comes from the policy table, never from the plan — the planner emits
    intents and the application maps them, so a plan cannot route itself.

    Blocking work runs in a worker thread, so a plan's tasks genuinely overlap and
    an async API handler is never stalled by a synchronous database or vector-store
    call.

    Args:
        task: The task to execute.
        index: Corpus index, loaded from the database when omitted.

    Returns:
        A TaskOutcome.
    """
    task = scoped_to_company(task)
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

    return await _run_rag_task(task, entity, index)


async def execute_plan(plan: QueryPlan, index: CorpusIndex | None = None) -> list[TaskOutcome]:
    """Execute every task in a plan concurrently.

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