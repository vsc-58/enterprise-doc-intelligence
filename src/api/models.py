"""API request and response models: the HTTP contract as Pydantic v2 classes.

The HTTP contract of docs/api_contract.md as Pydantic v2 models. FastAPI
validates requests against these before any handler runs (malformed input ->
422, no planner call), filters responses through them, and generates the
OpenAPI schema and Swagger UI from them.

Data only. The logic that builds a QueryResponse from a plan, its outcomes and
the assembled answer lives in src/api/mapping.py, so it can be tested
as a pure function.

Enums are reused from the query layer rather than copied as strings, so the
OpenAPI schema lists their exact values and a renamed member breaks loudly.
RefusalReason is the one API-owned enum: it merges two query-layer enums plus
two API-only codes (docs/api_contract.md §3.1); a test keeps it in step with
its sources.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.query.schemas import (
    FindingStatus,
    Intent,
    MetricField,
    PlanType,
    TaskStatus,
    Tool,
    UnsupportedReason,
)
from src.storage.quality import QualityReason, QualityVerdict
from src.utils.config import settings

# --- Request -----------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body of POST /query.

    Only `question`. Extra fields are rejected, so a client sending company or
    year filters gets a 422 rather than having them silently ignored. There are
    no filters by design: the resolver alone decides which companies and years
    are held, and a filter could silently drop half of a two-company question.
    Whitespace is stripped before the length check, so "   " is empty.

    Attributes:
        question: The user's question, verbatim apart from outer whitespace.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"question": "What was Apple's total revenue in fiscal 2021?"}]
        },
    )

    question: str = Field(
        min_length=1,
        max_length=settings.QUERY_MAX_CHARS,
        description="A question about the 20 filings in the corpus.",
    )


# Named examples for the Swagger dropdown, passed to Body(openapi_examples=...)
# by the /query route, and shown as buttons on the demo page at "/". One per
# behaviour the demo should show. Summaries are plain language: recruiters read
# them on the demo page, not only developers in Swagger. Chosen from the
# end-to-end cases already run during evaluation, so each has a known outcome.
QUERY_EXAMPLES: dict[str, dict[str, object]] = {
    "figure": {
        "summary": "A figure",
        "value": {"question": "What was Apple's total revenue in fiscal 2021?"},
    },
    "year_not_held": {
        "summary": "A year that is not held",
        "value": {"question": "What was Apple's revenue in 2022?"},
    },
    "narrative": {
        "summary": "An answer with sources",
        "value": {"question": "What does Walmart say about its liquidity?"},
    },
    "flagged": {
        "summary": "A figure flagged as uncertain",
        "value": {"question": "What were Bank of America's total assets?"},
    },
    "filter_flagged_not_shown": {
        "summary": "A list that names a company it may have missed",
        "value": {"question": "Which companies had total assets above $1 trillion?"},
    },
    "hybrid_partial": {
        "summary": "Two questions in one, one not answerable",
        "value": {"question": "What was Intel's net income, and why did its margins fall?"},
    },
    "not_in_corpus": {
        "summary": "A company not in the filings",
        "value": {"question": "What was Nvidia's revenue?"},
    },
}


# Swagger descriptions for the two planner enums the API reuses. Their own
# docstrings are part of the planner prompt (sent to the model and hashed into
# planner_version), so they cannot be edited for API readers without changing
# the measured prompt. main.py puts these texts in the OpenAPI schema instead;
# the classes and the planner prompt are untouched.
OPENAPI_SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "Intent": (
        "What information one task needs. The three financial intents "
        "(financial_metric, financial_ranking, financial_filter) are answered "
        "by the SQL tool from the extracted figures. The six narrative intents "
        "are answered from cited filing text, each searching its own set of "
        "10-K sections. The application, not the planner, maps an intent to "
        "its tool."
    ),
    "UnsupportedReason": (
        "A capability gap the planner declared. cross_year: needs two fiscal "
        "years of one company; the corpus holds one. aggregation: an aggregate "
        "across companies (average, total, count). ratio_or_derived: a figure "
        "computed from stored ones (margin, growth). multi_condition: more "
        "than one filter condition at once. field_not_queryable: a fact in the "
        "filing that neither tool serves, such as the auditor's name. Whether "
        "a company or year is held is decided later, by the resolver."
    ),
}


# --- /query response ---------------------------------------------------------


class RefusalReason(str, Enum):
    """Why a refused task gave no answer (docs/api_contract.md §3.1).

    One field for three sources: the resolver's ResolutionFailure, the chain's
    NarrativeRefusal (minus FAILED, which is a failure, not a refusal), and two
    codes the API assigns to SQL refusals that carry no typed reason.

    THRESHOLD_UNPARSED: a filter threshold the Python parser could not read
        ("fifty billion dollars"). The planner copies the phrase and Python
        parses it; the model never converts scale words to numbers.
    SQL_NOT_RUNNABLE: any other SQL refusal without a resolver failure.
        Defensive: not reachable with the current resolver and policy table.
    """

    NON_CORPUS_COMPANY = "non_corpus_company"
    COMPANY_NOT_EMBEDDED = "company_not_embedded"
    COMPANY_NOT_EXTRACTED = "company_not_extracted"
    YEAR_NOT_HELD = "year_not_held"
    AMBIGUOUS_COMPANY = "ambiguous_company"
    NO_COMPANY_MENTION = "no_company_mention"
    SECTION_ABSENT = "section_absent"
    NOT_ADDRESSED = "not_addressed"
    FIGURE_REQUESTED = "figure_requested"
    UNCITED = "uncited"
    THRESHOLD_UNPARSED = "threshold_unparsed"
    SQL_NOT_RUNNABLE = "sql_not_runnable"


class QualityOut(BaseModel):
    """The data-quality flag on one served figure.

    Computed from the stored extraction evidence only, never from the
    evaluation answer key.

    Attributes:
        verdict: verified, unverified or disputed.
        reason: Why that verdict was reached.
        note: QualityFlag.note() — the same sentence the answer text and the CSV
            use. None when verified.
    """

    model_config = ConfigDict(frozen=True)

    verdict: QualityVerdict
    reason: QualityReason
    note: str | None = None


class FigureOut(BaseModel):
    """One stored figure, exactly as the SQL path served it.

    The value is in actual dollars and comes from extracted_records — never from
    XBRL (edgartools boundary rule). It was rendered in Python: no LLM produced
    or saw it.

    Attributes:
        ticker, company_name: Which company. company_name is display_name().
        metric: Which of the five fields.
        value: Actual dollars, or None when not extracted.
        fiscal_year, fiscal_year_end: The filing's period. Shown on every row
            because rows from different companies mix fiscal years.
        quality: Its data-quality flag. Always present.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str
    metric: MetricField
    value: float | None
    fiscal_year: int
    fiscal_year_end: str | None
    quality: QualityOut


class FindingOut(BaseModel):
    """One company's result in a corpus-wide narrative task (one verdict per company).

    The claim text is not repeated here: it is already in `answer`, with its
    citation markers. This gives a client the coverage in structured form.

    Attributes:
        ticker, company_name, fiscal_year: Which company.
        status: supported, not_found, not_searchable or failed.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str
    fiscal_year: int
    status: FindingStatus


class TaskOut(BaseModel):
    """One planned task and what happened to it.

    Attributes:
        task_id: Identifier within the plan.
        intent: One of the nine intents.
        tool: sql or rag, from the policy table — never chosen by the plan.
        status: success, refused or failed. Refused is a correct answer about
            the corpus; failed means something broke.
        refusal_reason: Set only when status is refused.
        figures: SQL tasks: the rows served, in answer order.
        flagged_not_shown: Rankings and filters: scale-flagged companies the
            query left out that a 1,000x correction would bring in.
        excluded_null: Rankings and filters: companies left out because the
            figure was not extracted (value null), named in the answer.
        excluded_null_count: len(excluded_null); kept for clients that only
            need the number.
        mixed_years: Rankings and filters: whether rows span fiscal years.
        findings: Corpus-wide narrative tasks: one per company.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    intent: Intent
    tool: Tool
    status: TaskStatus
    refusal_reason: RefusalReason | None = None
    figures: list[FigureOut] = Field(default_factory=list)
    flagged_not_shown: list[FigureOut] = Field(default_factory=list)
    excluded_null: list[FigureOut] = Field(default_factory=list)
    excluded_null_count: int = 0
    mixed_years: bool = False
    findings: list[FindingOut] = Field(default_factory=list)


class SourceOut(BaseModel):
    """One entry in the answer's numbered Sources list.

    Section granularity is the honest limit of a citation: a chunk cites the
    10-K item it came from, not a page.

    Attributes:
        number: The [n] used in the answer text.
        ticker, company_name, fiscal_year: Which filing.
        section: Stored section label, e.g. "Item 7".
        chunk_index: Position of the chunk within that section.
        label: The rendered Sources line, as in the answer text.
    """

    model_config = ConfigDict(frozen=True)

    number: int
    ticker: str
    company_name: str
    fiscal_year: int
    section: str
    chunk_index: int
    label: str


class QueryResponse(BaseModel):
    """Response of POST /query for every answered request — answers and refusals.

    Never carries the planner's rationale, the SQL preview, retrieved chunk text
    or any XBRL value (docs/api_contract.md §3).

    Attributes:
        question: The validated question, echoed.
        answer: Everything the user reads, Sources list included.
        plan_type: single, hybrid, clarification, unsupported or out_of_scope.
        unsupported_reason: Set only for unsupported plans.
        tasks: One per planned task, in plan order. Empty for the three plan
            types that are answers in themselves.
        sources: The numbered sources cited in `answer`.
        planner_version, chain_version: Which prompts produced this answer.
        request_id: Matches the X-Request-ID header and every log line.
        latency_ms: Time spent answering, in milliseconds.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "question": "What was Apple's total revenue in fiscal 2021?",
                    "answer": (
                        "Apple Inc.'s total revenue for fiscal year 2021 "
                        "(period ending 2021-09-25) was $365.82 billion."
                    ),
                    "plan_type": "single",
                    "unsupported_reason": None,
                    "tasks": [
                        {
                            "task_id": "t1",
                            "intent": "financial_metric",
                            "tool": "sql",
                            "status": "success",
                            "refusal_reason": None,
                            "figures": [
                                {
                                    "ticker": "AAPL",
                                    "company_name": "Apple Inc.",
                                    "metric": "total_revenue",
                                    "value": 365817000000.0,
                                    "fiscal_year": 2021,
                                    "fiscal_year_end": "2021-09-25",
                                    "quality": {
                                        "verdict": "verified",
                                        "reason": "consistent",
                                        "note": None,
                                    },
                                }
                            ],
                            "flagged_not_shown": [],
                            "excluded_null": [],
                            "excluded_null_count": 0,
                            "mixed_years": False,
                            "findings": [],
                        }
                    ],
                    "sources": [],
                    "planner_version": "d4bc8e36",
                    "chain_version": "b1d8ee5c",
                    "request_id": "6f1c2a9e0b7d4e1f",
                    "latency_ms": 1840,
                }
            ]
        },
    )

    question: str
    answer: str
    plan_type: PlanType
    unsupported_reason: UnsupportedReason | None = None
    tasks: list[TaskOut] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    planner_version: str
    chain_version: str
    request_id: str
    latency_ms: int = Field(ge=0)


# --- /documents --------------------------------------------------------------


class DocumentOut(BaseModel):
    """One filing the corpus holds, read from the corpus index (no DB call).

    Attributes:
        ticker, company_name, cik: Which company. company_name is display_name().
        fiscal_year: The filing year held — the only year the corpus answers for.
        fiscal_year_end: Period end from the latest successful extraction.
        is_extracted: Whether the figures table has a row for it.
        is_embedded: Whether its narrative text is in the vector store.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str
    cik: str
    fiscal_year: int
    fiscal_year_end: str | None
    is_extracted: bool
    is_embedded: bool


# --- /health -----------------------------------------------------------------


class CheckState(str, Enum):
    """Result of one readiness check."""

    OK = "ok"
    UNAVAILABLE = "unavailable"


class HealthState(str, Enum):
    """Overall readiness: ok only when every check is ok."""

    OK = "ok"
    NOT_READY = "not_ready"


class HealthChecks(BaseModel):
    """The individual readiness checks, one per readiness part. None calls an LLM.

    A check is ok only when the readiness state counts that part as usable AND,
    where there is one, its live check passes. So a 503 always names at least
    one unavailable part. A test keeps these fields equal to the
    readiness parts, so a new part cannot be left out of the body.

    Attributes:
        database: SQLite file present and SELECT 1 succeeds (live).
        vector_store: Chroma folder present and holds chunks (live).
        index: The corpus index is loaded (readiness state).
        figure_guard: The stored figures the guard masks are loaded (readiness
            state). /query cannot run without them.
    """

    model_config = ConfigDict(frozen=True)

    database: CheckState
    vector_store: CheckState
    index: CheckState
    figure_guard: CheckState


class HealthResponse(BaseModel):
    """Body of GET /health, for both 200 (ready) and 503 (not ready).

    Attributes:
        status: ok or not_ready.
        checks: Which part is ready.
        documents: Filings in the corpus index, or None if it did not load.
        chunks: Chunks in the vector store, or None if it is unavailable.
        planner_version, chain_version: Prompt versions this server runs.
    """

    model_config = ConfigDict(frozen=True)

    status: HealthState
    checks: HealthChecks
    documents: int | None = None
    chunks: int | None = None
    planner_version: str
    chain_version: str


# --- Errors ------------------------------------------------------------------


class ErrorCode(str, Enum):
    """Machine-readable error codes for every non-422 error (docs/api_contract.md §7).

    NOT_READY (503): database, store or index unavailable.
    LLM_UNAVAILABLE (503): OpenAI unreachable, timed out, rate-limited or
        rejected the key on the planner call.
    SERVICE_DEGRADED (503): every task in the plan failed.
    TIMEOUT (504): QUERY_TIMEOUT_SECONDS exceeded.
    RATE_LIMITED (429): one client sent too many questions in the last minute.
    DAILY_BUDGET_EXHAUSTED (429): the demo's daily question budget is used up;
        it resets at 00:00 UTC. Separate from RATE_LIMITED because the client's
        action differs: wait a moment versus come back tomorrow.
    INTERNAL_ERROR (500): anything unexpected.
    """

    NOT_READY = "not_ready"
    LLM_UNAVAILABLE = "llm_unavailable"
    SERVICE_DEGRADED = "service_degraded"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    DAILY_BUDGET_EXHAUSTED = "daily_budget_exhausted"
    INTERNAL_ERROR = "internal_error"


class ErrorBody(BaseModel):
    """The one error envelope. 422 keeps FastAPI's own format instead.

    `message` is fixed text written in our code. Exception text, stack traces,
    paths, SQL and keys go to the logs only; `request_id` connects the two.

    Attributes:
        error: Machine-readable code.
        message: Safe sentence for a human.
        request_id: Matches the X-Request-ID header and the log lines.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "error": "timeout",
                    "message": "The question took too long to answer. Try a narrower question.",
                    "request_id": "a83b0c1d22e94f70",
                }
            ]
        },
    )

    error: ErrorCode
    message: str
    request_id: str
