# src/api/mapping.py
# Module: Query result -> API response
# Purpose: Build the /query response body from the three things the query path
#          produces: the plan, the per-task outcomes, and the assembled answer
#          (docs/api_contract.md §3, §3.1).
# Depends on: src.api.models, src.query.*, src.rag.*, src.storage.quality,
#             src.utils.text
#
# Pure functions: no I/O, no FastAPI, no model call. The answer text is passed
# through unchanged — every figure in it was rendered by sql_path in Python
# (D27) — so this module only re-shapes typed fields that already exist.
#
# Why AssembledAnswer alone is not enough: it carries text, sources and
# statuses. Figures and their D29 flags live in TaskOutcome.metric_result,
# refusal reasons in TaskOutcome.resolution_failure / narrative_result.refusal,
# and the plan type in QueryPlan.

from src.api.models import (
    FigureOut,
    FindingOut,
    QualityOut,
    QueryResponse,
    RefusalReason,
    SourceOut,
    TaskOut,
)
from src.query.intents import policy_for
from src.query.schemas import (
    CompanyFinding,
    Intent,
    MetricRow,
    NarrativeRefusal,
    QueryPlan,
    Task,
    TaskOutcome,
    TaskStatus,
    Tool,
)
from src.query.synthesizer import AssembledAnswer, NumberedSource
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)


def figure_out(row: MetricRow) -> FigureOut:
    """
    Project one served row, with its D29 flag.

    company_name is already a display name: sql_path applies display_name().

    Args:
        row: A row from MetricResult.

    Returns:
        The API figure.
    """
    return FigureOut(
        ticker=row.ticker,
        company_name=row.company_name,
        metric=row.metric,
        value=row.value,
        fiscal_year=row.fiscal_year,
        fiscal_year_end=row.fiscal_year_end,
        quality=QualityOut(
            verdict=row.quality.verdict,
            reason=row.quality.reason,
            note=row.quality.note(),
        ),
    )


def finding_out(finding: CompanyFinding) -> FindingOut:
    """
    Project one corpus-wide finding to its structured coverage line.

    Args:
        finding: A CompanyFinding. Its company_name is the registrant as filed.

    Returns:
        The API finding, with the display name used everywhere else.
    """
    return FindingOut(
        ticker=finding.ticker,
        company_name=display_name(finding.company_name, finding.ticker),
        fiscal_year=finding.fiscal_year,
        status=finding.status,
    )


def source_out(source: NumberedSource) -> SourceOut:
    """
    Project one numbered source.

    Args:
        source: An entry of AssembledAnswer.sources.

    Returns:
        The API source; label is exactly the line printed in the answer text.
    """
    ref = source.ref
    return SourceOut(
        number=source.number,
        ticker=ref.ticker,
        company_name=display_name(ref.company_name, ref.ticker),
        fiscal_year=ref.fiscal_year,
        section=ref.section,
        chunk_index=ref.chunk_index,
        label=source.label(),
    )


def refusal_reason(outcome: TaskOutcome) -> RefusalReason | None:
    """
    The one typed reason for a refused task (contract §3.1).

    Order: the resolver decided first (company or year), then the narrative
    chain, then the two API-owned codes for SQL refusals that carry no type.

    Args:
        outcome: One task's outcome.

    Returns:
        The reason, or None when the task was not refused. A failed task has
        no refusal reason: failing is not refusing.
    """
    if outcome.status is not TaskStatus.REFUSED:
        return None
    if outcome.resolution_failure is not None:
        return RefusalReason(outcome.resolution_failure.value)

    narrative = outcome.narrative_result
    if (
        narrative is not None
        and narrative.refusal is not None
        and narrative.refusal is not NarrativeRefusal.FAILED
    ):
        return RefusalReason(narrative.refusal.value)

    if outcome.tool is Tool.SQL:
        # SqlPathError carries text, not a type. For a filter the only reachable
        # cause is an unreadable threshold ("fifty billion dollars").
        if outcome.intent is Intent.FINANCIAL_FILTER:
            return RefusalReason.THRESHOLD_UNPARSED
        return RefusalReason.SQL_NOT_RUNNABLE

    logger.warning("refusal_without_reason", task_id=outcome.task_id, tool=outcome.tool.value)
    return None


def task_out(task: Task, outcome: TaskOutcome | None) -> TaskOut:
    """
    One planned task and what happened to it.

    Args:
        task: The planned task.
        outcome: Its outcome, or None if the executor returned none for it —
            which it never does, but which must surface as failed rather than
            vanish (the same rule assemble() applies).

    Returns:
        The API task.
    """
    if outcome is None:
        return TaskOut(
            task_id=task.task_id,
            intent=task.intent,
            tool=policy_for(task.intent).tool,
            status=TaskStatus.FAILED,
        )

    metric = outcome.metric_result
    narrative = outcome.narrative_result
    return TaskOut(
        task_id=outcome.task_id,
        intent=outcome.intent,
        tool=outcome.tool,
        status=outcome.status,
        refusal_reason=refusal_reason(outcome),
        figures=[figure_out(row) for row in metric.rows] if metric else [],
        flagged_not_shown=(
            [figure_out(row) for row in metric.flagged_not_shown] if metric else []
        ),
        excluded_null_count=metric.excluded_null_count if metric else 0,
        mixed_years=metric.mixed_years if metric else False,
        findings=[finding_out(f) for f in narrative.findings] if narrative else [],
    )


def nothing_answered(answer: AssembledAnswer) -> bool:
    """
    Whether every task in a plan failed (contract §7: 503 service_degraded).

    Refused tasks count as answered: a refusal is a correct statement about the
    corpus. A plan with no tasks (clarification, unsupported, out_of_scope) is
    an answer in itself.

    Args:
        answer: The assembled answer; its statuses include a FAILED entry for
            any task the executor returned no outcome for.

    Returns:
        True only when there was at least one task and all of them failed.
    """
    return bool(answer.statuses) and all(
        status is TaskStatus.FAILED for status in answer.statuses
    )


def build_query_response(
    question: str,
    plan: QueryPlan,
    outcomes: list[TaskOutcome],
    answer: AssembledAnswer,
    *,
    planner_version: str,
    chain_version: str,
    request_id: str,
    latency_ms: int,
) -> QueryResponse:
    """
    Build the /query body.

    Tasks follow plan order, like the answer text. Fields the contract keeps
    out — the planner's rationale, the SQL preview, chunk text — are simply
    never read here.

    Args:
        question: The validated question.
        plan: The executed plan.
        outcomes: The executor's outcomes.
        answer: assemble(plan, outcomes).
        planner_version: planner_version().
        chain_version: chain_version().
        request_id: The middleware's request id.
        latency_ms: Time spent so far on this request.

    Returns:
        The response model.
    """
    by_id = {outcome.task_id: outcome for outcome in outcomes}
    return QueryResponse(
        question=question,
        answer=answer.text,
        plan_type=plan.plan_type,
        unsupported_reason=plan.unsupported_reason,
        tasks=[task_out(task, by_id.get(task.task_id)) for task in plan.tasks],
        sources=[source_out(source) for source in answer.sources],
        planner_version=planner_version,
        chain_version=chain_version,
        request_id=request_id,
        latency_ms=latency_ms,
    )
