"""Query-path evaluation: execution outcomes (refusals, figure guard, provenance).

Build gold ("oracle") plans from the frozen labels, reduce each executed
TaskOutcome to an outcome class comparable with the labels, and record what the
attribution and guard metrics need.

Two modes are scored by the runner (scripts/run_execution_eval.py):
  end_to_end — the planner's plan: what a user actually receives.
  oracle     — the label's strict plan, handed straight to the executor: the
               downstream system with routing assumed perfect.
The gap between them is the cost of routing errors; a miss in BOTH modes is a
downstream failure that no routing fix would cure.

Outcome classes are enums the code already emits, never parsed text:
resolver failures, NarrativeRefusal values, the figure quality verdict and the
section-fallback flag. A label's expected outcome uses the same vocabulary.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.eval.query_eval import (
    AMBIGUOUS,
    NON_CORPUS,
    EvalQuestion,
    ExpectedTask,
    ObservedPlan,
    PlanAlternative,
    matches,
)
from src.query.schemas import (
    Comparison,
    Direction,
    Intent,
    MetricField,
    PlanType,
    QueryPlan,
    Task,
    TaskOutcome,
    TaskStatus,
    Tool,
)
from src.storage.quality import QualityVerdict
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mentions that resolve to the label's non-company outcomes. Generic tokens are
# stripped by the resolver, so "the bank" reduces to nothing (AMBIGUOUS) and the
# placeholder matches no registrant (NON_CORPUS).
_MENTION_FOR = {NON_CORPUS: "Nonexistent Holdings Placeholder", AMBIGUOUS: "the bank"}

ANSWERED = "answered"
ANSWERED_FLAGGED = "answered_flagged"
FALLBACK_ANSWERED = "fallback_answered"
SQL_REFUSED = "sql_refused"
FAILED = "failed"


class OracleError(ValueError):
    """Raised when a label cannot be turned into an executable plan."""


class FilterParams(BaseModel):
    """Threshold parameters the labels do not carry, copied from the question text."""

    model_config = ConfigDict(frozen=True)

    comparison: Comparison
    threshold_text: str


class OracleParams(BaseModel):
    """Parameters an oracle plan needs that the frozen labels do not carry.

    Attributes:
        filters: Question id -> filter threshold, copied from the question text.
        task_questions: Question id -> task_id -> the retrieval question for one
            narrative task of a multi-task label. It is the narrative clause of
            the question in the question's own words, with a pronoun replaced by
            the company name — what the planner is asked to write. Without it the
            oracle would retrieve with the whole hybrid question, figure clause
            included, which no planned task ever does.
    """

    model_config = ConfigDict(frozen=True)

    filters: dict[str, FilterParams] = Field(default_factory=dict)
    task_questions: dict[str, dict[str, str]] = Field(default_factory=dict)


def load_oracle_params(path: Path | None = None) -> OracleParams:
    """Load the oracle parameters.

    Kept outside the frozen label file: they were added after the freeze, and are
    literal copies of (or minimal edits to) each question's own text.

    Args:
        path: JSON file; settings.EVAL_DATA_PATH / ORACLE_PARAMS_FILE when omitted.

    Returns:
        The parameters.

    Raises:
        OracleError: if the file is missing or malformed.
    """
    path = path or Path(settings.EVAL_DATA_PATH) / settings.ORACLE_PARAMS_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return OracleParams.model_validate(
            {"filters": raw["filters"], "task_questions": raw.get("task_questions", {})}
        )
    except (OSError, KeyError, ValueError) as exc:
        raise OracleError(f"cannot load oracle params from {path}: {exc}") from exc


def is_executable(question: EvalQuestion) -> bool:
    """Whether a question's strict label carries tasks to execute.

    Clarification, unsupported and out-of-scope plans have nothing downstream;
    routing already scores them.

    Args:
        question: A labelled question.

    Returns:
        True for single and hybrid strict labels.
    """
    return question.strict.plan_type in (PlanType.SINGLE.value, PlanType.HYBRID.value)


def _oracle_task(
    index: int,
    expected: ExpectedTask,
    question: EvalQuestion,
    params: OracleParams,
) -> Task:
    """Build one executable Task from one labelled task.

    Raises:
        OracleError: if a filter label has no threshold, or a narrative task of a
            multi-task label has no task question.
    """
    intent = Intent(expected.intent)
    mention = _MENTION_FOR.get(expected.company, expected.company) if expected.company else None
    fields: dict = {
        "task_id": f"t{index}",
        "intent": intent,
        "company_mention": mention,
        "year": expected.year,
    }
    if expected.metric:
        fields["metric"] = MetricField(expected.metric)
    if intent is Intent.FINANCIAL_RANKING:
        fields["direction"] = Direction(expected.direction or "highest")
    if intent is Intent.FINANCIAL_FILTER:
        threshold = params.filters.get(question.id)
        if threshold is None:
            raise OracleError(f"{question.id}: filter label has no oracle threshold")
        fields["comparison"] = threshold.comparison
        fields["threshold_text"] = threshold.threshold_text
    if intent not in (Intent.FINANCIAL_METRIC, Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER):
        if len(question.strict.tasks) == 1:
            fields["question"] = question.question
        else:
            task_question = params.task_questions.get(question.id, {}).get(fields["task_id"])
            if not task_question:
                raise OracleError(
                    f"{question.id}: narrative task {fields['task_id']} of a multi-task label "
                    "has no oracle task question"
                )
            fields["question"] = task_question
    return Task(**fields)


def oracle_plan(question: EvalQuestion, params: OracleParams) -> QueryPlan:
    """The label's strict plan as an executable QueryPlan.

    A single-task narrative label retrieves with the original question text — the
    planner would normally rephrase it, which the oracle deliberately does not
    imitate. A narrative task inside a multi-task label retrieves with its own
    task question from the oracle params.

    Args:
        question: An executable labelled question.
        params: Oracle parameters.

    Returns:
        The plan.

    Raises:
        OracleError: if the label is not executable or lacks a needed parameter.
    """
    if not is_executable(question):
        raise OracleError(f"{question.id}: strict label has no tasks")
    tasks = [
        _oracle_task(i, expected, question, params)
        for i, expected in enumerate(question.strict.tasks, start=1)
    ]
    return QueryPlan(
        plan_type=PlanType(question.strict.plan_type),
        tasks=tasks,
        rationale="oracle: strict label",
    )


def outcome_class(outcome: TaskOutcome) -> str:
    """Reduce an executed task to the label vocabulary.

    Args:
        outcome: One executed task.

    Returns:
        One of: a ResolutionFailure value; answered / answered_flagged for SQL
        (flagged when any served row is disputed); answered /
        fallback_answered for narrative; a NarrativeRefusal value;
        sql_refused; failed.
    """
    if outcome.resolution_failure is not None:
        return outcome.resolution_failure.value
    if outcome.tool is Tool.SQL:
        if outcome.status is TaskStatus.FAILED:
            return FAILED
        if outcome.status is not TaskStatus.SUCCESS or outcome.metric_result is None:
            return SQL_REFUSED
        disputed = any(
            row.quality.verdict is QualityVerdict.DISPUTED for row in outcome.metric_result.rows
        )
        return ANSWERED_FLAGGED if disputed else ANSWERED
    result = outcome.narrative_result
    if result is None:
        return FAILED
    if result.refusal is not None:
        return result.refusal.value
    return FALLBACK_ANSWERED if result.fallback_used else ANSWERED


class TaskRecord(BaseModel):
    """One executed task, as recorded for scoring and later attribution."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    intent: str
    company: str | None
    outcome: str
    expected: str | None = None
    message: str | None = None
    claims: list[dict] = Field(default_factory=list)
    findings: list[dict] = Field(default_factory=list)
    figures_masked: int = 0
    figures_redacted: int = 0
    invalid_citations: int = 0
    sections_searched: list[str] = Field(default_factory=list)
    metric_rows: list[dict] = Field(default_factory=list)

    @property
    def correct(self) -> bool | None:
        """Whether the outcome matches the label; None when there is no label."""
        return None if self.expected is None else self.outcome == self.expected


def record(outcome: TaskOutcome, company: str | None, expected: str | None) -> TaskRecord:
    """Record one executed task.

    Args:
        outcome: The executed task.
        company: The task's company in label vocabulary (ticker, NON_CORPUS, …).
        expected: The labelled outcome, or None when the plan matched no label.

    Returns:
        The record, including every claim with its cited sources — the input the
        attribution judge and its reference labels need.
    """
    result = outcome.narrative_result
    rows = outcome.metric_result.rows if outcome.metric_result else []
    return TaskRecord(
        task_id=outcome.task_id,
        intent=outcome.intent.value,
        company=company,
        outcome=outcome_class(outcome),
        expected=expected,
        message=outcome.message,
        claims=[c.model_dump(mode="json") for c in result.claims] if result else [],
        findings=[f.model_dump(mode="json") for f in result.findings] if result else [],
        figures_masked=result.figures_masked if result else 0,
        figures_redacted=result.figures_redacted if result else 0,
        invalid_citations=result.invalid_citations if result else 0,
        sections_searched=sorted(result.sections_searched) if result else [],
        metric_rows=[
            {"ticker": r.ticker, "fiscal_year": r.fiscal_year, "value": r.value,
             "verdict": r.quality.verdict.value}
            for r in rows
        ],
    )


def matched_alternative(observed: ObservedPlan, question: EvalQuestion) -> PlanAlternative | None:
    """The first acceptable label an observed plan matches.

    Args:
        observed: The planner's plan, reduced.
        question: The labelled question.

    Returns:
        The matching alternative, or None on a routing miss.
    """
    return next((alt for alt in question.alternatives if matches(observed, alt)), None)


def expected_by_key(observed: ObservedPlan, alternative: PlanAlternative) -> list[str | None]:
    """Align a matched label's expected outcomes with the observed tasks.

    Matching is by task key, since task order is not scored; each labelled task
    is consumed once, so two identical tasks map to two labelled ones.

    Args:
        observed: The planner's plan, reduced.
        alternative: The label it matched.

    Returns:
        One expected outcome per observed task, in observed order; None where no
        labelled task has that key.
    """
    pool = [(task.key(), task.outcome) for task in alternative.tasks]
    expected: list[str | None] = []
    for task in observed.tasks:
        hit = next((i for i, (key, _) in enumerate(pool) if key == task.key()), None)
        expected.append(pool.pop(hit)[1] if hit is not None else None)
    return expected
