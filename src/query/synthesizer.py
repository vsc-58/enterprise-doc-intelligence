# src/query/synthesizer.py
# Module: Answer assembly
# Purpose: Turn a plan and its task outcomes into the one answer the user sees —
#          parts in plan order, citation numbers assigned across the whole answer,
#          one Sources list, and an explicit sentence for every part that could
#          not be answered.
# Depends on: pydantic v2, src.query.schemas, src.rag.chain, src.utils.logger,
#             src.utils.text
#
# Python only. There is no synthesiser LLM: every figure was rendered by sql_path,
# every narrative sentence was written and cited by the chain, and every refusal
# sentence is fixed text. This module only orders, numbers and joins (D27, D41).
#
# Citation numbers are assigned here, not by the chain. Each chain call numbers
# its own blocks from 1, so a hybrid or corpus-wide answer would otherwise show
# two different passages both labelled [1]. A chunk cited twice keeps one number.

from pydantic import BaseModel, ConfigDict, Field

from src.query.schemas import (
    CitedClaim,
    FindingStatus,
    NarrativeResult,
    PlanType,
    QueryPlan,
    SourceRef,
    Task,
    TaskOutcome,
    TaskStatus,
    Tool,
    UnsupportedReason,
)
from src.rag.chain import FIGURE_WITHHELD_NOTE, SECTION_NAMES
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

_UNSUPPORTED_TEXT: dict[UnsupportedReason, str] = {
    UnsupportedReason.CROSS_YEAR: (
        "I hold one annual filing per company, so I can't compare a company "
        "against its own earlier years."
    ),
    UnsupportedReason.AGGREGATION: (
        "I can look up, rank and filter company figures, but I don't compute "
        "aggregates across the corpus."
    ),
    UnsupportedReason.RATIO_OR_DERIVED: (
        "I serve the five figures as filed. I don't compute ratios or derived "
        "measures from them."
    ),
    UnsupportedReason.MULTI_CONDITION: (
        "I can filter on one condition at a time. Could you ask for one of them, "
        "and I'll follow up with the other?"
    ),
    UnsupportedReason.FIELD_NOT_QUERYABLE: (
        "That field is extracted and appears in the CSV export, but the query "
        "path doesn't serve it."
    ),
}

_OUT_OF_SCOPE_TEXT = "That isn't something a 10-K annual report covers, so I don't hold it."
_CLARIFICATION_FALLBACK = "Could you rephrase that?"
_MISSING_PART_TEXT = "This part could not be answered."


class NumberedSource(BaseModel):
    """
    One entry in the answer's Sources list.

    Attributes:
        number: The citation number shown in the answer text.
        ref: The chunk it points to.
    """

    model_config = ConfigDict(frozen=True)

    number: int
    ref: SourceRef

    def label(self) -> str:
        """
        Render the Sources-list line.

        chunk_index is shown so a reviewer can find the exact passage; section
        granularity is otherwise the honest limit of the citation (Phase 4).

        Returns:
            e.g. "[2] Apple Inc., FY2021, MD&A (Item 7), chunk 2".
        """
        section = SECTION_NAMES.get(self.ref.section, self.ref.section)
        name = display_name(self.ref.company_name, self.ref.ticker)
        return (
            f"[{self.number}] {name}, FY{self.ref.fiscal_year}, {section}, "
            f"chunk {self.ref.chunk_index}"
        )


class AssembledAnswer(BaseModel):
    """
    The final answer to one question.

    Attributes:
        text: Everything the user reads, Sources list included.
        sources: The numbered sources, in citation order.
        statuses: Each task's status, in plan order. Empty for plans that carry
            no tasks. For the API response and the 5C eval.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    sources: list[NumberedSource] = Field(default_factory=list)
    statuses: list[TaskStatus] = Field(default_factory=list)


def _key(ref: SourceRef) -> tuple[str, int, str, int]:
    """
    Identity of a cited chunk, independent of its distance to any one query.

    Args:
        ref: A source.

    Returns:
        (ticker, fiscal_year, section, chunk_index).
    """
    return (ref.ticker, ref.fiscal_year, ref.section, ref.chunk_index)


class _Citations:
    """Assigns answer-wide citation numbers, first citation first."""

    def __init__(self) -> None:
        self._numbers: dict[tuple[str, int, str, int], int] = {}
        self.sources: list[NumberedSource] = []

    def markers(self, refs: list[SourceRef]) -> str:
        """
        Number a claim's sources and render its markers.

        Args:
            refs: The claim's sources, in the order the chain cited them.

        Returns:
            e.g. "[1][3]".
        """
        numbers: list[int] = []
        for ref in refs:
            key = _key(ref)
            if key not in self._numbers:
                self._numbers[key] = len(self.sources) + 1
                self.sources.append(NumberedSource(number=self._numbers[key], ref=ref))
            if self._numbers[key] not in numbers:
                numbers.append(self._numbers[key])
        return "".join(f"[{n}]" for n in numbers)


def plan_refusal_text(plan: QueryPlan) -> str:
    """
    The answer for a plan that carries no tasks.

    Args:
        plan: A clarification, unsupported or out-of-scope plan.

    Returns:
        The text shown to the user.
    """
    if plan.plan_type is PlanType.CLARIFICATION:
        return plan.clarification_question or _CLARIFICATION_FALLBACK
    if plan.plan_type is PlanType.OUT_OF_SCOPE:
        return _OUT_OF_SCOPE_TEXT
    if plan.unsupported_reason is not None:
        return _UNSUPPORTED_TEXT[plan.unsupported_reason]
    return "That's outside what this system can answer."


def _cited(claim: CitedClaim, citations: _Citations) -> str:
    """
    Render one claim with its markers.

    Args:
        claim: A cited claim.
        citations: The answer-wide numbering.

    Returns:
        The sentence followed by its markers.
    """
    return f"{claim.text} {citations.markers(claim.sources)}"


def _tickers(result: NarrativeResult, status: FindingStatus) -> str:
    """
    Comma-joined tickers of the findings with one status.

    Args:
        result: A corpus-wide result.
        status: The status to collect.

    Returns:
        Tickers in corpus order, or an empty string.
    """
    return ", ".join(f.ticker for f in result.findings if f.status is status)


def _render_findings(result: NarrativeResult, citations: _Citations) -> str:
    """
    Render a corpus-wide result: supported companies, then the coverage statement.

    "No supporting passage found" is worded as what it is — the two retrieved
    passages did not support the topic — never as "did not disclose" (D39).

    Args:
        result: A corpus-wide NarrativeResult.
        citations: The answer-wide numbering.

    Returns:
        The rendered part.
    """
    supported = [f for f in result.findings if f.status is FindingStatus.SUPPORTED]
    lines: list[str] = []
    if supported:
        lines.append("Companies whose filings discuss this:")
        for finding in supported:
            name = display_name(finding.company_name, finding.ticker)
            claim = _cited(finding.claim, citations) if finding.claim else ""
            lines.append(f"- {name} ({finding.ticker}): {claim}".rstrip())
    else:
        lines.append("No company's retrieved passages discuss this.")

    coverage = [
        ("No supporting passage found in the retrieved text", FindingStatus.NOT_FOUND),
        ("Section not captured in this corpus", FindingStatus.NOT_SEARCHABLE),
        ("Could not be checked", FindingStatus.FAILED),
    ]
    for label, status in coverage:
        tickers = _tickers(result, status)
        if tickers:
            lines.append(f"{label}: {tickers}")
    return "\n".join(lines)


def _render_narrative(result: NarrativeResult, citations: _Citations) -> str:
    """
    Render a successful narrative result.

    Args:
        result: A NarrativeResult with claims or findings.
        citations: The answer-wide numbering.

    Returns:
        The rendered part, with the figure-withheld note if the backstop fired.
    """
    if result.findings:
        body = _render_findings(result, citations)
    else:
        body = " ".join(_cited(claim, citations) for claim in result.claims)
    if result.figures_redacted:
        body = f"{body}\n{FIGURE_WITHHELD_NOTE}"
    return body


def _render_part(
    task: Task,
    outcome: TaskOutcome,
    citations: _Citations,
    multi_part: bool,
) -> str:
    """
    Render one task's part of the answer.

    In a multi-part answer an unanswered part is introduced by the task's own
    question, so the reader can tell which half could not be answered — the R38
    shape: the figure, then "Why did Intel's margins fall? — <reason>".

    Args:
        task: The planned task.
        outcome: What happened to it.
        citations: The answer-wide numbering.
        multi_part: Whether the plan has more than one task.

    Returns:
        The rendered part.
    """
    if outcome.status is TaskStatus.SUCCESS:
        if outcome.tool is Tool.RAG and outcome.narrative_result is not None:
            return _render_narrative(outcome.narrative_result, citations)
        return outcome.message or _MISSING_PART_TEXT

    reason = outcome.message or _MISSING_PART_TEXT
    if multi_part and task.question:
        return f"{task.question} — {reason}"
    return reason


def assemble(plan: QueryPlan, outcomes: list[TaskOutcome]) -> AssembledAnswer:
    """
    Assemble the final answer for a plan.

    Parts follow plan order. A task with no outcome — which the executor never
    produces, but which must not silently vanish if it did — is reported as an
    unanswered part rather than dropped.

    Args:
        plan: The executed plan.
        outcomes: The executor's outcomes, one per task in plan order.

    Returns:
        The assembled answer.
    """
    if not plan.tasks:
        return AssembledAnswer(text=plan_refusal_text(plan))

    by_id = {outcome.task_id: outcome for outcome in outcomes}
    citations = _Citations()
    parts: list[str] = []
    statuses: list[TaskStatus] = []
    multi_part = len(plan.tasks) > 1

    for task in plan.tasks:
        outcome = by_id.get(task.task_id)
        if outcome is None:
            logger.error("task_outcome_missing", task_id=task.task_id)
            parts.append(
                f"{task.question} — {_MISSING_PART_TEXT}" if task.question
                else _MISSING_PART_TEXT
            )
            statuses.append(TaskStatus.FAILED)
            continue
        parts.append(_render_part(task, outcome, citations, multi_part))
        statuses.append(outcome.status)

    text = "\n\n".join(parts)
    if citations.sources:
        text += "\n\nSources:\n" + "\n".join(s.label() for s in citations.sources)

    logger.info(
        "answer_assembled",
        plan_type=plan.plan_type.value,
        statuses=[status.value for status in statuses],
        sources=len(citations.sources),
    )
    return AssembledAnswer(text=text, sources=citations.sources, statuses=statuses)