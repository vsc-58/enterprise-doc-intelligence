"""Query-path evaluation: frozen labels, routing scores and confidence intervals.

Load the frozen eval set (hash-checked), turn a QueryPlan into a comparable
signature, score it strictly and leniently against the labels, classify every
miss, and aggregate across runs with per-question disagreement (temperature 0
is not deterministic, so one run is a point estimate, not a measurement).

Pure scoring: no LLM and no database here. The runner supplies plans and the
corpus index, which is why every rule below is unit-testable.

Company is compared as a RESOLVED identity, not as text. The planner copies
mentions verbatim ("amzn", "jnj", "that bank"), so string comparison would
score correct plans as wrong. The resolver turns a mention into a ticker,
NON_CORPUS or AMBIGUOUS, which is what the labels hold. A label expecting a
held company whose observed mention resolves to NON_CORPUS or AMBIGUOUS is
flagged resolver_suspect: the planner only copies, so the likelier fault is a
resolver gap, and it is reported separately rather than charged to routing.
"""

import hashlib
import json
import math
from collections import Counter
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.query.resolve import CorpusIndex, resolve_company
from src.query.schemas import PlanType, QueryPlan, ResolutionFailure
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

NON_CORPUS = "NON_CORPUS"
AMBIGUOUS = "AMBIGUOUS"
_FALLBACK_PREFIX = "planner fallback"

# (intent, company, metric, year, direction): the parts of a task that change
# what gets executed. threshold_text and the rephrased question are not scored —
# the first is parsed in Python, the second is judged by retrieval and attribution.
TaskKey = tuple[str, str | None, str | None, int | None, str | None]


class EvalSetError(RuntimeError):
    """Raised when the eval set is missing, malformed, or not the frozen version."""


# --- Labels ------------------------------------------------------------------


class ExpectedTask(BaseModel):
    """One labelled task. Field names mirror the label file."""

    model_config = ConfigDict(frozen=True)

    intent: str
    company: str | None = None
    metric: str | None = None
    year: int | None = None
    direction: str | None = None
    outcome: str = "answered"

    def key(self) -> TaskKey:
        """The comparable identity of this task.

        Returns:
            (intent, company, metric, year, direction).
        """
        return (self.intent, self.company, self.metric, self.year, self.direction)


class PlanAlternative(BaseModel):
    """One acceptable plan for a question. The first in a list is the strict label."""

    model_config = ConfigDict(frozen=True)

    plan_type: str
    unsupported_reason: str | None = None
    tasks: list[ExpectedTask] = Field(default_factory=list)


class EvalQuestion(BaseModel):
    """One labelled question."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    split: str
    source: str
    stratum: str
    alternatives: list[PlanAlternative] = Field(min_length=1)
    narrative_set: bool = False
    notes: str = ""

    @property
    def strict(self) -> PlanAlternative:
        """The strict label: the first alternative."""
        return self.alternatives[0]


class EvalSet(BaseModel):
    """The frozen eval set and the hash it was verified against."""

    model_config = ConfigDict(frozen=True)

    sha256: str
    questions: list[EvalQuestion]


def canonical_sha256(path: Path) -> str:
    """Hash a file with line endings normalised to LF.

    Git on Windows may check text files out with CRLF; the content is unchanged,
    so the hash must be too.

    Args:
        path: File to hash.

    Returns:
        Hex SHA-256 digest.

    Raises:
        OSError: if the file cannot be read.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_eval_set(path: Path | None = None, expected_sha256: str | None = None) -> EvalSet:
    """Load the eval set, refusing anything but the frozen version.

    Args:
        path: Label file; settings.EVAL_DATA_PATH / QUERY_EVAL_SET_FILE when omitted.
        expected_sha256: Required hash; settings.QUERY_EVAL_SET_SHA256 when omitted.

    Returns:
        The parsed eval set.

    Raises:
        EvalSetError: if the file is missing, unparseable, or its hash differs
            from the frozen one.
    """
    path = path or Path(settings.EVAL_DATA_PATH) / settings.QUERY_EVAL_SET_FILE
    expected = expected_sha256 or settings.QUERY_EVAL_SET_SHA256
    try:
        digest = canonical_sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalSetError(f"cannot read eval set at {path}: {exc}") from exc

    if digest != expected:
        raise EvalSetError(
            f"eval set hash {digest[:12]}… does not match the frozen {expected[:12]}…; "
            "labels changed after the freeze"
        )
    try:
        questions = [EvalQuestion.model_validate(q) for q in payload["questions"]]
    except (KeyError, ValueError) as exc:
        raise EvalSetError(f"malformed eval set: {exc}") from exc

    logger.info("eval_set_loaded", questions=len(questions), sha256=digest[:12])
    return EvalSet(sha256=digest, questions=questions)


# --- Observations ------------------------------------------------------------


class ObservedTask(BaseModel):
    """One planned task, with its mention resolved for comparison."""

    model_config = ConfigDict(frozen=True)

    intent: str
    mention: str | None
    company: str | None
    metric: str | None = None
    year: int | None = None
    direction: str | None = None
    question: str | None = None

    def key(self) -> TaskKey:
        """Return the comparable identity of this task.

        The rephrased question is recorded for error analysis but not scored.

        Returns:
            (intent, company, metric, year, direction).
        """
        return (self.intent, self.company, self.metric, self.year, self.direction)


class ObservedPlan(BaseModel):
    """A planner output reduced to what the eval compares."""

    model_config = ConfigDict(frozen=True)

    plan_type: str
    unsupported_reason: str | None = None
    tasks: list[ObservedTask] = Field(default_factory=list)
    fallback: bool = False
    rationale: str | None = None

    def signature(self) -> str:
        """A stable string identifying this plan, for cross-run disagreement.

        Task order is ignored, as in scoring.

        Returns:
            e.g. "hybrid|-|company_overview:AMZN:-:-:-,financial_metric:AMZN:net_income:-:-".
        """
        tasks = sorted(
            ":".join("-" if part is None else str(part) for part in task.key())
            for task in self.tasks
        )
        return f"{self.plan_type}|{self.unsupported_reason or '-'}|{','.join(tasks)}"


def resolved_company(mention: str | None, index: CorpusIndex) -> str | None:
    """Map a raw mention onto the label vocabulary.

    Args:
        mention: The company as the planner copied it, or None.
        index: The corpus index.

    Returns:
        A ticker, NON_CORPUS, AMBIGUOUS, or None for no mention.
    """
    if mention is None or not mention.strip():
        return None
    resolution = resolve_company(mention, index=index)
    if resolution.entity is not None:
        return resolution.entity.ticker
    if resolution.failure is ResolutionFailure.AMBIGUOUS_COMPANY:
        return AMBIGUOUS
    if resolution.failure is ResolutionFailure.NO_COMPANY_MENTION:
        return None
    return NON_CORPUS


def observe(plan: QueryPlan, index: CorpusIndex) -> ObservedPlan:
    """Reduce a planner output to a comparable observation.

    Args:
        plan: The planner's plan.
        index: The corpus index, for resolving mentions.

    Returns:
        The observation.
    """
    return ObservedPlan(
        plan_type=plan.plan_type.value,
        unsupported_reason=plan.unsupported_reason.value if plan.unsupported_reason else None,
        tasks=[
            ObservedTask(
                intent=task.intent.value,
                mention=task.company_mention,
                company=resolved_company(task.company_mention, index),
                metric=task.metric.value if task.metric else None,
                year=task.year,
                direction=task.direction.value if task.direction else None,
                question=task.question,
            )
            for task in plan.tasks
        ],
        fallback=bool(plan.rationale and plan.rationale.startswith(_FALLBACK_PREFIX)),
        rationale=plan.rationale,
    )


# --- Scoring -----------------------------------------------------------------


class MissCategory(str, Enum):
    """The first point at which a plan departs from its closest label.

    Ordered from coarse to fine; a miss is classified at the earliest level that
    differs, so each miss is counted once. CALL_ERROR is a planner call that
    failed after a retry: counted as a miss so it cannot inflate accuracy by
    leaving the denominator, and reported so it is not mistaken for a routing
    error.
    """

    CALL_ERROR = "call_error"
    PLANNER_FALLBACK = "planner_fallback"
    PLAN_TYPE = "plan_type"
    UNSUPPORTED_REASON = "unsupported_reason"
    TASK_COUNT = "task_count"
    INTENT = "intent"
    COMPANY = "company"
    PARAMETER = "parameter"


class QuestionScore(BaseModel):
    """One question's result in one run."""

    model_config = ConfigDict(frozen=True)

    strict: bool
    lenient: bool
    miss: MissCategory | None = None
    resolver_suspect: bool = False


def matches(observed: ObservedPlan, alternative: PlanAlternative) -> bool:
    """Whether a plan matches one acceptable label.

    Args:
        observed: The planner's plan.
        alternative: One acceptable plan.

    Returns:
        True on the same plan type, the same unsupported reason, and the same
        multiset of task keys (task order ignored).
    """
    if observed.fallback or observed.plan_type != alternative.plan_type:
        return False
    if observed.plan_type == PlanType.UNSUPPORTED.value:
        return observed.unsupported_reason == alternative.unsupported_reason
    return Counter(t.key() for t in observed.tasks) == Counter(t.key() for t in alternative.tasks)


def _closest(observed: ObservedPlan, alternatives: list[PlanAlternative]) -> PlanAlternative | None:
    """The first alternative sharing the observed plan type, if any."""
    return next((a for a in alternatives if a.plan_type == observed.plan_type), None)


def classify_miss(observed: ObservedPlan, alternatives: list[PlanAlternative]) -> MissCategory:
    """Name the first level at which a plan departs from its closest label.

    Args:
        observed: A plan that matched no alternative.
        alternatives: The question's acceptable plans.

    Returns:
        The miss category.
    """
    if observed.fallback:
        return MissCategory.PLANNER_FALLBACK
    target = _closest(observed, alternatives)
    if target is None:
        return MissCategory.PLAN_TYPE
    if observed.plan_type == PlanType.UNSUPPORTED.value:
        return MissCategory.UNSUPPORTED_REASON
    if len(observed.tasks) != len(target.tasks):
        return MissCategory.TASK_COUNT
    if Counter(t.intent for t in observed.tasks) != Counter(t.intent for t in target.tasks):
        return MissCategory.INTENT
    if Counter((t.intent, t.company) for t in observed.tasks) != Counter(
        (t.intent, t.company) for t in target.tasks
    ):
        return MissCategory.COMPANY
    return MissCategory.PARAMETER


def _resolver_suspect(observed: ObservedPlan, alternatives: list[PlanAlternative]) -> bool:
    """Whether a company miss looks like a resolver gap rather than a planner error.

    True when a label expects a held company and the observed mention for the
    same intent resolved to NON_CORPUS or AMBIGUOUS.
    """
    target = _closest(observed, alternatives)
    if target is None:
        return False
    expected = Counter(
        t.intent for t in target.tasks if t.company not in (None, NON_CORPUS, AMBIGUOUS)
    )
    failed = Counter(t.intent for t in observed.tasks if t.company in (NON_CORPUS, AMBIGUOUS))
    return bool(expected & failed)


CALL_ERROR_SCORE = QuestionScore(strict=False, lenient=False, miss=MissCategory.CALL_ERROR)


def score_question(question: EvalQuestion, observed: ObservedPlan) -> QuestionScore:
    """Score one plan against one question's labels.

    Args:
        question: The labelled question.
        observed: The planner's plan for it.

    Returns:
        Strict and lenient correctness, and the miss category when lenient fails.
    """
    strict = matches(observed, question.strict)
    lenient = strict or any(matches(observed, alt) for alt in question.alternatives[1:])
    if lenient:
        return QuestionScore(strict=strict, lenient=True)
    category = classify_miss(observed, question.alternatives)
    return QuestionScore(
        strict=False,
        lenient=False,
        miss=category,
        resolver_suspect=(
            category is MissCategory.COMPANY
            and _resolver_suspect(observed, question.alternatives)
        ),
    )


# --- Aggregation -------------------------------------------------------------


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Accepts a fractional success count so a mean over runs can be passed with n
    equal to the number of QUESTIONS: repeated runs measure the model's
    variability on the same questions, not new samples, so they do not narrow
    the interval.

    Args:
        successes: Number (or mean number) of successes.
        n: Number of questions.
        z: Normal quantile; 1.96 for 95%.

    Returns:
        (lower, upper), each in [0, 1]. (0.0, 1.0) when n is zero.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class GroupSummary(BaseModel):
    """Accuracy for one group of questions across runs."""

    model_config = ConfigDict(frozen=True)

    n: int
    strict_per_run: list[int]
    lenient_per_run: list[int]
    strict_mean: float
    lenient_mean: float
    strict_ci95: tuple[float, float]
    lenient_ci95: tuple[float, float]
    plan_varied: int
    score_varied: int


def summarise(
    questions: list[EvalQuestion],
    scores: dict[str, list[QuestionScore]],
    signatures: dict[str, list[str]],
) -> GroupSummary:
    """Aggregate one group of questions across runs.

    Args:
        questions: The group.
        scores: Question id -> one score per run.
        signatures: Question id -> one plan signature per run.

    Returns:
        Per-run counts, means as proportions, Wilson intervals over n questions,
        and two disagreement counts: questions whose plan differed between runs,
        and questions whose strict correctness differed.
    """
    n = len(questions)
    runs = len(next(iter(scores.values()))) if scores else 0
    strict_runs = [sum(scores[q.id][r].strict for q in questions) for r in range(runs)]
    lenient_runs = [sum(scores[q.id][r].lenient for q in questions) for r in range(runs)]
    strict_mean = sum(strict_runs) / runs if runs else 0.0
    lenient_mean = sum(lenient_runs) / runs if runs else 0.0
    return GroupSummary(
        n=n,
        strict_per_run=strict_runs,
        lenient_per_run=lenient_runs,
        strict_mean=strict_mean / n if n else 0.0,
        lenient_mean=lenient_mean / n if n else 0.0,
        strict_ci95=wilson_interval(strict_mean, n),
        lenient_ci95=wilson_interval(lenient_mean, n),
        plan_varied=sum(len(set(signatures[q.id])) > 1 for q in questions),
        score_varied=sum(len({s.strict for s in scores[q.id]}) > 1 for q in questions),
    )
