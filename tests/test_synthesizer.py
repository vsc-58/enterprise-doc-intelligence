"""Synthesizer tests: plan order, answer-wide citation numbers, refusals, coverage."""

from src.query.schemas import (
    CitedClaim,
    CompanyFinding,
    FindingStatus,
    Intent,
    MetricField,
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
from src.query.synthesizer import assemble, plan_refusal_text
from src.rag.chain import FIGURE_WITHHELD_NOTE


def _ref(ticker: str, section: str, index: int) -> SourceRef:
    return SourceRef(ticker=ticker, company_name=f"{ticker} Inc.", fiscal_year=2021,
                     section=section, chunk_index=index, distance=0.4)


def _sql_task(task_id: str = "t1") -> Task:
    return Task(task_id=task_id, intent=Intent.FINANCIAL_METRIC, company_mention="Intel",
                metric=MetricField.NET_INCOME)


def _rag_task(task_id: str, question: str, mention: str | None = "Apple") -> Task:
    return Task(task_id=task_id, intent=Intent.MANAGEMENT_COMMENTARY,
                company_mention=mention, question=question)


def _rag_ok(task_id: str, claims: list[CitedClaim], redacted: int = 0) -> TaskOutcome:
    return TaskOutcome(task_id=task_id, intent=Intent.MANAGEMENT_COMMENTARY, tool=Tool.RAG,
                       status=TaskStatus.SUCCESS,
                       narrative_result=NarrativeResult(claims=claims, figures_redacted=redacted))


def test_r38_partial_hybrid_keeps_the_figure_and_names_the_unanswered_part() -> None:
    plan = QueryPlan(plan_type=PlanType.HYBRID, tasks=[
        _sql_task(), _rag_task("t2", "Why did Intel's margins fall?", "Intel"),
    ])
    outcomes = [
        TaskOutcome(task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL,
                    status=TaskStatus.SUCCESS, message="INTEL CORP's net income was $8.01 billion."),
        TaskOutcome(task_id="t2", intent=Intent.MANAGEMENT_COMMENTARY, tool=Tool.RAG,
                    status=TaskStatus.REFUSED, message="The MD&A section was not captured."),
    ]
    answer = assemble(plan, outcomes)
    assert answer.text == (
        "INTEL CORP's net income was $8.01 billion.\n\n"
        "Why did Intel's margins fall? — The MD&A section was not captured."
    )
    assert answer.statuses == [TaskStatus.SUCCESS, TaskStatus.REFUSED]
    assert answer.sources == []


def test_parts_follow_plan_order_not_outcome_order() -> None:
    plan = QueryPlan(plan_type=PlanType.HYBRID, tasks=[_sql_task("a"), _sql_task("b")])
    outcomes = [
        TaskOutcome(task_id=t, intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL,
                    status=TaskStatus.SUCCESS, message=t.upper())
        for t in ("b", "a")
    ]
    assert assemble(plan, outcomes).text == "A\n\nB"


def test_citation_numbers_are_answer_wide_and_deduplicated() -> None:
    plan = QueryPlan(plan_type=PlanType.HYBRID, tasks=[
        _rag_task("t1", "Why did sales grow?"), _rag_task("t2", "What drove services?"),
    ])
    shared, other, third = _ref("AAPL", "Item 7", 2), _ref("AAPL", "Item 7", 1), _ref("AAPL", "Item 7", 3)
    outcomes = [
        _rag_ok("t1", [CitedClaim(text="iPhone grew.", sources=[shared, other])]),
        _rag_ok("t2", [CitedClaim(text="Services grew.", sources=[third, shared])]),
    ]
    answer = assemble(plan, outcomes)
    assert "iPhone grew. [1][2]" in answer.text
    assert "Services grew. [3][1]" in answer.text
    assert [s.ref.chunk_index for s in answer.sources] == [2, 1, 3]
    assert answer.text.endswith(
        "Sources:\n"
        "[1] AAPL Inc., FY2021, MD&A (Item 7), chunk 2\n"
        "[2] AAPL Inc., FY2021, MD&A (Item 7), chunk 1\n"
        "[3] AAPL Inc., FY2021, MD&A (Item 7), chunk 3"
    )


def test_figure_withheld_note_appears_only_when_the_backstop_fired() -> None:
    plan = QueryPlan(plan_type=PlanType.SINGLE, tasks=[_rag_task("t1", "Why?")])
    claim = CitedClaim(text="Sales rose.", sources=[_ref("AAPL", "Item 7", 1)])
    assert FIGURE_WITHHELD_NOTE not in assemble(plan, [_rag_ok("t1", [claim])]).text
    assert FIGURE_WITHHELD_NOTE in assemble(plan, [_rag_ok("t1", [claim], redacted=1)]).text


def test_corpus_wide_lists_supported_then_states_coverage() -> None:
    plan = QueryPlan(plan_type=PlanType.SINGLE, tasks=[
        _rag_task("t1", "Which companies flagged interest rate risk?", None)])

    def finding(ticker: str, status: FindingStatus) -> CompanyFinding:
        claim = (CitedClaim(text=f"{ticker} discloses it.", sources=[_ref(ticker, "Item 7A", 0)])
                 if status is FindingStatus.SUPPORTED else None)
        return CompanyFinding(ticker=ticker, company_name=f"{ticker} Inc.", fiscal_year=2021,
                              status=status, claim=claim)

    result = NarrativeResult(findings=[
        finding("AAPL", FindingStatus.SUPPORTED),
        finding("JPM", FindingStatus.NOT_SEARCHABLE),
        finding("TSLA", FindingStatus.NOT_FOUND),
        finding("V", FindingStatus.SUPPORTED),
        finding("XOM", FindingStatus.NOT_SEARCHABLE),
    ])
    outcome = TaskOutcome(task_id="t1", intent=Intent.MARKET_RISK, tool=Tool.RAG,
                          status=TaskStatus.SUCCESS, narrative_result=result)
    text = assemble(plan, [outcome]).text
    assert text.startswith(
        "Companies whose filings discuss this:\n"
        "- AAPL Inc. (AAPL): AAPL discloses it. [1]\n"
        "- V Inc. (V): V discloses it. [2]\n"
        "No supporting passage found in the retrieved text: TSLA\n"
        "Section not captured in this corpus: JPM, XOM"
    )


def test_single_part_refusal_is_not_prefixed_with_the_question() -> None:
    plan = QueryPlan(plan_type=PlanType.SINGLE, tasks=[_rag_task("t1", "Why?")])
    outcome = TaskOutcome(task_id="t1", intent=Intent.MANAGEMENT_COMMENTARY, tool=Tool.RAG,
                          status=TaskStatus.REFUSED, message="Not addressed.")
    assert assemble(plan, [outcome]).text == "Not addressed."


def test_missing_outcome_is_reported_not_dropped() -> None:
    plan = QueryPlan(plan_type=PlanType.HYBRID, tasks=[_sql_task(), _rag_task("t2", "Why?")])
    outcome = TaskOutcome(task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL,
                          status=TaskStatus.SUCCESS, message="Figure.")
    answer = assemble(plan, [outcome])
    assert answer.text == "Figure.\n\nWhy? — This part could not be answered."
    assert answer.statuses == [TaskStatus.SUCCESS, TaskStatus.FAILED]


def test_plans_without_tasks_render_their_fixed_text() -> None:
    assert plan_refusal_text(QueryPlan(plan_type=PlanType.OUT_OF_SCOPE)).startswith(
        "That isn't something a 10-K")
    unsupported = QueryPlan(plan_type=PlanType.UNSUPPORTED,
                            unsupported_reason=UnsupportedReason.AGGREGATION)
    assert "aggregates" in assemble(unsupported, []).text
    clarify = QueryPlan(plan_type=PlanType.CLARIFICATION, clarification_question="Which bank?")
    assert assemble(clarify, []).text == "Which bank?"