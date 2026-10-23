"""Hand-built plans and outcomes for the API tests, from real end-to-end cases.

Each sample is what the planner and executor produced for a question run
during the evaluation. Building them by hand keeps the API tests free of any
model call, database or vector store.
"""

from src.query.schemas import (
    CitedClaim,
    CompanyFinding,
    Comparison,
    FindingStatus,
    Intent,
    MetricField,
    MetricResult,
    MetricRow,
    NarrativeRefusal,
    NarrativeResult,
    PlanType,
    QueryPlan,
    ResolutionFailure,
    SourceRef,
    Task,
    TaskOutcome,
    TaskStatus,
    Tool,
    UnsupportedReason,
)
from src.storage.quality import QualityFlag, QualityReason, QualityVerdict

APPLE_SENTENCE = (
    "Apple Inc.'s total revenue for fiscal year 2021 (period ending 2021-09-25) "
    "was $365.82 billion."
)
YEAR_SENTENCE = "I hold one Apple Inc. filing, for fiscal year 2021. I don't have 2022."
VERIFIED = QualityFlag(field="total_revenue", verdict=QualityVerdict.VERIFIED,
                       reason=QualityReason.CONSISTENT)
NOT_EXTRACTED = QualityFlag(field="total_assets", verdict=QualityVerdict.UNVERIFIED,
                            reason=QualityReason.NULL_FIELD)
AMBIGUOUS = QualityFlag(field="total_assets", verdict=QualityVerdict.DISPUTED,
                        reason=QualityReason.SCALE_AMBIGUOUS)

# --- Tasks ----------------------------------------------------------------------

APPLE_TASK = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, company_mention="Apple",
                  year=2021, metric=MetricField.TOTAL_REVENUE)
APPLE_2022_TASK = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, company_mention="Apple",
                       year=2022, metric=MetricField.TOTAL_REVENUE)
WALMART_TASK = Task(task_id="t1", intent=Intent.MANAGEMENT_COMMENTARY, company_mention="Walmart",
                    question="What does Walmart say about its liquidity?")
INTEL_FIGURE_TASK = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, company_mention="Intel",
                         metric=MetricField.NET_INCOME)
INTEL_MARGIN_TASK = Task(task_id="t2", intent=Intent.MANAGEMENT_COMMENTARY,
                         company_mention="Intel", question="Why did Intel's margins fall?")
FILTER_TASK = Task(task_id="t1", intent=Intent.FINANCIAL_FILTER, metric=MetricField.TOTAL_ASSETS,
                   comparison=Comparison.GT, threshold_text="fifty billion dollars")
SURVEY_TASK = Task(task_id="t1", intent=Intent.RISK_FACTORS,
                   question="Which companies discuss supply chain risk?")


def single(task: Task) -> QueryPlan:
    """A one-task plan."""
    return QueryPlan(plan_type=PlanType.SINGLE, tasks=[task])


def hybrid(*tasks: Task) -> QueryPlan:
    """A multi-task plan."""
    return QueryPlan(plan_type=PlanType.HYBRID, tasks=list(tasks))


UNSUPPORTED_PLAN = QueryPlan(plan_type=PlanType.UNSUPPORTED,
                             unsupported_reason=UnsupportedReason.AGGREGATION)

# --- Outcomes ---------------------------------------------------------------------


def _row(ticker: str, name: str, metric: MetricField, value: float | None, quality: QualityFlag,
         year: int = 2021, end: str | None = "2021-09-25") -> MetricRow:
    return MetricRow(ticker=ticker, company_name=name, fiscal_year=year, fiscal_year_end=end,
                     metric=metric, value=value, quality=quality)


APPLE_ROW = _row("AAPL", "Apple Inc.", MetricField.TOTAL_REVENUE, 365817000000.0, VERIFIED)
BAC_ROW = _row("BAC", "Bank of America Corp.", MetricField.TOTAL_ASSETS, 3051375000.0,
               AMBIGUOUS, year=2022, end="2022-12-31")
NULL_ASSETS_ROW = _row("JNJ", "Johnson & Johnson", MetricField.TOTAL_ASSETS, None,
                       NOT_EXTRACTED, end="2022-01-02")

APPLE_OUTCOME = TaskOutcome(
    task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL, status=TaskStatus.SUCCESS,
    metric_result=MetricResult(rows=[APPLE_ROW]), message=APPLE_SENTENCE,
)
YEAR_REFUSED = TaskOutcome(
    task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL, status=TaskStatus.REFUSED,
    resolution_failure=ResolutionFailure.YEAR_NOT_HELD, message=YEAR_SENTENCE,
)
NVIDIA_REFUSED = TaskOutcome(
    task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL, status=TaskStatus.REFUSED,
    resolution_failure=ResolutionFailure.NON_CORPUS_COMPANY,
    message="I don't hold a filing for Nvidia.",
)
THRESHOLD_REFUSED = TaskOutcome(
    task_id="t1", intent=Intent.FINANCIAL_FILTER, tool=Tool.SQL, status=TaskStatus.REFUSED,
    message="I couldn't read 'fifty billion dollars' as an amount.",
)
FILTER_WITH_DROPPED_FLAG = TaskOutcome(
    task_id="t1", intent=Intent.FINANCIAL_FILTER, tool=Tool.SQL, status=TaskStatus.SUCCESS,
    metric_result=MetricResult(rows=[], flagged_not_shown=[BAC_ROW],
                               excluded_null=[NULL_ASSETS_ROW], excluded_null_count=1,
                               mixed_years=False, threshold_value=5e10),
    message="No company in the corpus matches that total assets criterion.",
)

WALMART_SOURCE = SourceRef(ticker="WMT", company_name="Walmart Inc.", fiscal_year=2023,
                           section="Item 7", chunk_index=25, distance=0.31)
WALMART_OUTCOME = TaskOutcome(
    task_id="t1", intent=Intent.MANAGEMENT_COMMENTARY, tool=Tool.RAG, status=TaskStatus.SUCCESS,
    narrative_result=NarrativeResult(claims=[CitedClaim(
        text="As of January 31, 2023, Walmart had $15.0 billion of undrawn committed "
             "lines of credit in the U.S.",
        sources=[WALMART_SOURCE],
    )]),
)
INTEL_FIGURE = TaskOutcome(
    task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL, status=TaskStatus.SUCCESS,
    metric_result=MetricResult(rows=[_row("INTC", "Intel Corp", MetricField.NET_INCOME,
                                          8.0e9, VERIFIED, year=2022, end="2022-12-31")]),
    message="Intel Corp's net income for fiscal year 2022 was $8 billion.",
)
INTEL_SECTION_ABSENT = TaskOutcome(
    task_id="t2", intent=Intent.MANAGEMENT_COMMENTARY, tool=Tool.RAG, status=TaskStatus.REFUSED,
    narrative_result=NarrativeResult(refusal=NarrativeRefusal.SECTION_ABSENT),
    message="Intel's MD&A (Item 7) was not captured in this corpus.",
)
INTEL_RAG_FAILED = TaskOutcome(
    task_id="t2", intent=Intent.MANAGEMENT_COMMENTARY, tool=Tool.RAG, status=TaskStatus.FAILED,
    narrative_result=NarrativeResult(refusal=NarrativeRefusal.FAILED),
    message="This part could not be answered.",
)
APPLE_SQL_FAILED = TaskOutcome(
    task_id="t1", intent=Intent.FINANCIAL_METRIC, tool=Tool.SQL, status=TaskStatus.FAILED,
    message="I couldn't retrieve that figure from the structured store.",
)
SURVEY_OUTCOME = TaskOutcome(
    task_id="t1", intent=Intent.RISK_FACTORS, tool=Tool.RAG, status=TaskStatus.SUCCESS,
    narrative_result=NarrativeResult(findings=[
        CompanyFinding(ticker="GS", company_name="GOLDMAN SACHS GROUP INC", fiscal_year=2021,
                       status=FindingStatus.FAILED),
        CompanyFinding(ticker="JPM", company_name="JPMORGAN CHASE & CO", fiscal_year=2023,
                       status=FindingStatus.NOT_FOUND),
    ]),
)
