# src/query/schemas.py
# Module: Query-path contracts (plan, tasks, resolved entities, typed results)
# Purpose: Every object that crosses a boundary between planner, resolver,
#          executor, tools and assembler. Nothing here executes anything.
# Depends on: pydantic v2, src.storage.metadata_store, src.storage.quality

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.storage.quality import QualityFlag


class Intent(str, Enum):
    """
    What information a single task needs.

    Describes the information need, not the implementation: the intent-to-tool
    mapping lives in intents.py and is applied by the executor, so a task never
    carries a tool the planner chose (D25).

    Every narrative intent maps to a distinct set of corpus sections. Two
    intents searching the same set would execute identically, and a label the
    system cannot act on only adds disagreement to the routing eval, so such
    intents are merged (D28, D38). A test enforces this.
    """

    FINANCIAL_METRIC = "financial_metric"
    FINANCIAL_RANKING = "financial_ranking"
    FINANCIAL_FILTER = "financial_filter"

    COMPANY_OVERVIEW = "company_overview"
    STRATEGY = "strategy"
    COMPETITION_AND_REGULATION = "competition_and_regulation"
    RISK_FACTORS = "risk_factors"
    MANAGEMENT_COMMENTARY = "management_commentary"
    MARKET_RISK = "market_risk"


class Tool(str, Enum):
    """The two execution capabilities. Assigned by intents.py, never by the planner."""

    SQL = "sql"
    RAG = "rag"


class MetricField(str, Enum):
    """
    The five queryable numeric columns.

    An allowlist expressed as a type: a column name that is not a member cannot
    reach the query builder, so the parameterized-SQL guarantee is enforced by
    the schema rather than by a check somewhere in sql_path.

    Values match the column names in extracted_records and the names in
    EVIDENCE_FIELDS; a test asserts they stay in step.
    """

    TOTAL_REVENUE = "total_revenue"
    NET_INCOME = "net_income"
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    OPERATING_CASH_FLOW = "operating_cash_flow"


class Comparison(str, Enum):
    """Comparison operators available to FINANCIAL_FILTER."""

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class Direction(str, Enum):
    """Sort directions available to FINANCIAL_RANKING."""

    HIGHEST = "highest"
    LOWEST = "lowest"


class PlanType(str, Enum):
    """
    The shape of a plan.

    SINGLE and HYBRID carry tasks. The other three carry no tasks and a reason:
    they are answers in themselves, and all three return HTTP 200 with a
    structured refusal rather than an error status.
    """

    SINGLE = "single"
    HYBRID = "hybrid"
    CLARIFICATION = "clarification"
    UNSUPPORTED = "unsupported"
    OUT_OF_SCOPE = "out_of_scope"


class UnsupportedReason(str, Enum):
    """
    Capability gaps the planner may declare.

    Deliberately excludes corpus-membership failures. The planner is corpus-blind
    (D32), so it cannot know whether a company is held; those outcomes belong to
    ResolutionFailure and are produced by the resolver. Keeping them out of this
    enum means the planner's structured output cannot express a judgment it has
    no basis for.

    CROSS_YEAR: needs two fiscal years of one company. The corpus holds one.
    AGGREGATION: needs an aggregate across companies (average, total, count).
    RATIO_OR_DERIVED: needs a figure computed from stored ones (margin, growth).
    MULTI_CONDITION: needs more than one filter condition at once.
    FIELD_NOT_QUERYABLE: the field exists in the export but not the query path,
        e.g. auditor_name.
    """

    CROSS_YEAR = "cross_year"
    AGGREGATION = "aggregation"
    RATIO_OR_DERIVED = "ratio_or_derived"
    MULTI_CONDITION = "multi_condition"
    FIELD_NOT_QUERYABLE = "field_not_queryable"


class ResolutionFailure(str, Enum):
    """
    Corpus-membership outcomes, produced only by the resolver.

    NON_CORPUS_COMPANY: no document for that company.
    COMPANY_NOT_EMBEDDED: the document exists but has no chunks, so a narrative
        task would retrieve nothing and look like an empty corpus.
    YEAR_NOT_HELD: the company is held, for a different fiscal year.
    AMBIGUOUS_COMPANY: the mention matches more than one held company.
    NO_COMPANY_MENTION: the question named no company and the task needs one.
    """

    NON_CORPUS_COMPANY = "non_corpus_company"
    COMPANY_NOT_EMBEDDED = "company_not_embedded"
    YEAR_NOT_HELD = "year_not_held"
    AMBIGUOUS_COMPANY = "ambiguous_company"
    NO_COMPANY_MENTION = "no_company_mention"


class Task(BaseModel):
    """
    One information requirement.

    Carries an intent and raw mentions, never resolved identities and never a
    tool. Entity resolution happens in resolve.py and tool selection in the
    executor, so the planner's output stays a statement about the question
    rather than about the corpus or the implementation.

    threshold_text holds the literal phrase a filter was expressed with — "$50
    billion", "50bn" — rather than a number. Converting scale words to a float
    is exactly the operation the extraction pipeline's top silent failure came
    from, so it is done in Python, not by the model (D27's reasoning applied to
    parameters).

    Attributes:
        task_id: Stable identifier, unique within the plan.
        intent: What this task needs.
        company_mention: Company as the user referred to it, unresolved.
        year: Fiscal year if the question named one literally. Relative periods
            are rejected: "latest" differs per company in this corpus.
        metric: Which numeric field, for financial intents.
        comparison / threshold_text: Filter parameters.
        direction / limit: Ranking parameters.
        question: Self-contained rephrasing for narrative intents, used as the
            retrieval query.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    intent: Intent
    company_mention: str | None = None
    year: int | None = Field(default=None, ge=1990, le=2100)
    metric: MetricField | None = None
    comparison: Comparison | None = None
    threshold_text: str | None = None
    direction: Direction | None = None
    limit: int | None = Field(default=None, ge=1, le=20)
    question: str | None = None

    @model_validator(mode="after")
    def check_intent_parameters(self) -> "Task":
        """
        Reject parameter sets an intent cannot execute.

        Returns:
            The validated task.

        Raises:
            ValueError: if a required parameter for the intent is missing.
        """
        if self.intent in _FINANCIAL_INTENTS and self.metric is None:
            raise ValueError(f"{self.intent.value} requires a metric")
        if self.intent is Intent.FINANCIAL_FILTER and (
            self.comparison is None or self.threshold_text is None
        ):
            raise ValueError("financial_filter requires comparison and threshold_text")
        if self.intent is Intent.FINANCIAL_RANKING and self.direction is None:
            raise ValueError("financial_ranking requires a direction")
        if self.intent not in _FINANCIAL_INTENTS and not self.question:
            raise ValueError(f"{self.intent.value} requires a question")
        return self


_FINANCIAL_INTENTS = frozenset(
    {Intent.FINANCIAL_METRIC, Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER}
)

MAX_TASKS_PER_PLAN = 3


class QueryPlan(BaseModel):
    """
    The planner's whole output.

    Capped at three tasks. The cap is a scope statement rather than a technical
    limit: a question needing more than three independent lookups is beyond what
    a one-filing-per-company corpus can answer well, and an uncapped plan would
    let one malformed question fan out into unbounded cost.

    No depends_on and no execution mode. With one fiscal year per company no plan
    over this corpus can contain a dependency, so an engine for them would have
    no reachable path (D26). Tasks are therefore always independent and always
    executed concurrently.

    Attributes:
        plan_type: Shape of the plan.
        tasks: The information requirements, empty for the three refusal types.
        unsupported_reason: Set when plan_type is UNSUPPORTED.
        clarification_question: Set when plan_type is CLARIFICATION.
        rationale: One short line for logging and the routing eval. Never shown
            to the user and never an answer.
    """

    model_config = ConfigDict(frozen=True)

    plan_type: PlanType
    tasks: list[Task] = Field(default_factory=list, max_length=MAX_TASKS_PER_PLAN)
    unsupported_reason: UnsupportedReason | None = None
    clarification_question: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def check_shape(self) -> "QueryPlan":
        """
        Enforce the task count and required field for each plan type.

        Returns:
            The validated plan.

        Raises:
            ValueError: if the task count or accompanying fields contradict
                        plan_type, or if two tasks share a task_id.
        """
        count = len(self.tasks)
        if self.plan_type is PlanType.SINGLE and count != 1:
            raise ValueError(f"single plan must carry exactly one task, got {count}")
        if self.plan_type is PlanType.HYBRID and not 2 <= count <= MAX_TASKS_PER_PLAN:
            raise ValueError(f"hybrid plan must carry 2-{MAX_TASKS_PER_PLAN} tasks, got {count}")
        if self.plan_type in _TASKLESS_PLANS and count:
            raise ValueError(f"{self.plan_type.value} plan must carry no tasks, got {count}")
        if self.plan_type is PlanType.UNSUPPORTED and self.unsupported_reason is None:
            raise ValueError("unsupported plan must name a reason")
        if self.plan_type is PlanType.CLARIFICATION and not self.clarification_question:
            raise ValueError("clarification plan must carry a question")
        if len({task.task_id for task in self.tasks}) != count:
            raise ValueError("task_id values must be unique within a plan")
        return self


_TASKLESS_PLANS = frozenset(
    {PlanType.CLARIFICATION, PlanType.UNSUPPORTED, PlanType.OUT_OF_SCOPE}
)

class ResolvedEntity(BaseModel):
    """
    A company mention resolved against the corpus.

    Carries the identifiers each store actually keys on: the ticker for chunk
    metadata (D30), the document id for the structured read, and the filing year
    the corpus holds, which the caller did not necessarily ask for.
    """

    model_config = ConfigDict(frozen=True)

    mention: str
    ticker: str
    cik: str
    company_name: str
    document_id: int
    filing_year: int
    fiscal_year_end: str | None = None
    is_embedded: bool = True
    is_extracted: bool = True

class Resolution(BaseModel):
    """
    The outcome of resolving one company mention against the corpus.

    Either entity or failure is set, never both. A failure is an answer about the
    corpus, not an error: "I don't hold Nvidia" is correct and final, so it
    travels as data rather than as an exception.
    """

    model_config = ConfigDict(frozen=True)

    entity: ResolvedEntity | None = None
    failure: ResolutionFailure | None = None
    detail: str | None = None
    candidates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_exclusive(self) -> "Resolution":
        """
        Enforce exactly one of entity or failure.

        Returns:
            The validated resolution.

        Raises:
            ValueError: if both or neither is set.
        """
        if (self.entity is None) == (self.failure is None):
            raise ValueError("resolution must carry exactly one of entity or failure")
        return self

class MetricRow(BaseModel):
    """
    One company's figure, with the context needed to read it honestly.

    fiscal_year travels with every row because a multi-row answer over this
    corpus mixes fiscal years, and a ranking that hides that is misleading (D33).
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_year_end: str | None
    metric: MetricField
    value: float | None
    quality: QualityFlag


class MetricResult(BaseModel):
    """
    The SQL tool's output: rows plus what the renderer must disclose.

    Attributes:
        rows: Result rows, already ordered.
        excluded_null_count: Rows dropped for a null value in the queried metric.
            Reported rather than silently omitted: BAC's operating cash flow is
            null because the true figure is negative, so "lowest cash flow" would
            otherwise skip the correct answer.
        mixed_years: Whether rows span more than one fiscal year (D33).
        sql_preview: The rendered statement, for logging and the API response.
        threshold_value: The parsed filter threshold in dollars, or None for
        non-filter intents. Carried rather than reparsed so the number that
        selected the rows and the number quoted back to the user are the same
        value, computed once.
    """

    model_config = ConfigDict(frozen=True)

    rows: list[MetricRow] = Field(default_factory=list)
    excluded_null_count: int = 0
    mixed_years: bool = False
    sql_preview: str | None = None
    threshold_value: float | None = None


class SourceRef(BaseModel):
    """
    One retrieved chunk, as cited.

    distance is Chroma's cosine DISTANCE — lower is closer. Named distance, not
    score, because Phase 4 found the opposite reading easy to make and costly.
    Section granularity is the honest limit of the citation: a chunk cites the
    section it came from, not a page (Phase 4).
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str
    fiscal_year: int
    section: str
    chunk_index: int
    distance: float


class NarrativeRefusal(str, Enum):
    """
    Why a narrative task produced no answer. Each has one fixed sentence,
    written in Python (src.rag.chain.refusal_message), so the refusal metric
    compares enums rather than parsing model prose.

    SECTION_ABSENT: the company holds none of the intent's sections (INTC, JPM
        and XOM have no Item 7). Decided by retrieval, before any LLM call.
    NOT_ADDRESSED: the retrieved passages do not answer the question. Judged by
        the chain.
    FIGURE_REQUESTED: the question asks for one of the five stored metrics,
        which only the SQL tool may state (D27, D41).
    UNCITED: the chain answered, but no claim survived citation validation.
        Separate from NOT_ADDRESSED because it is a model fault, not a fact
        about the filing.
    FAILED: retrieval, the figure guard's data, or the LLM call broke.
    """

    SECTION_ABSENT = "section_absent"
    NOT_ADDRESSED = "not_addressed"
    FIGURE_REQUESTED = "figure_requested"
    UNCITED = "uncited"
    FAILED = "failed"


class CitedClaim(BaseModel):
    """
    One sentence of a narrative answer and the chunks it cites.

    Citations are a typed field, not markers parsed out of prose, and every
    source here was cited by the model for this sentence and validated to exist
    in the retrieved set. Sources are therefore not "everything retrieved",
    which is what keeps the attribution check meaningful (D41).
    """

    model_config = ConfigDict(frozen=True)

    text: str
    sources: list[SourceRef] = Field(min_length=1)


class FindingStatus(str, Enum):
    """
    One company's result in a corpus-wide narrative task (D39).

    SUPPORTED: a retrieved passage discusses the topic for this company.
    NOT_FOUND: no supporting passage in the retrieved text — NOT a claim that
        the filing is silent.
    NOT_SEARCHABLE: the company holds none of the intent's sections.
    FAILED: retrieval or the verdict call broke for this company.
    """

    SUPPORTED = "supported"
    NOT_FOUND = "not_found"
    NOT_SEARCHABLE = "not_searchable"
    FAILED = "failed"


class CompanyFinding(BaseModel):
    """One company's verdict in a corpus-wide narrative task."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str
    fiscal_year: int
    status: FindingStatus
    claim: CitedClaim | None = None


class NarrativeResult(BaseModel):
    """
    The RAG tool's output. Structured, not prose: the synthesizer renders it.

    A single-company task fills claims; a corpus-wide task fills findings. A
    refusal leaves both empty. Never carries a stored figure: matching values
    are masked in the context before the model sees it and redacted from claims
    if one still appears (D27, D41); the counters make both measurable in 5C.

    Attributes:
        claims: Cited sentences, in the model's order.
        findings: One per corpus company, in corpus order.
        refusal: Why there is no answer, or None.
        figures_masked: Stored-metric values withheld from the context.
        figures_redacted: Stored-metric values removed from claims — the
        backstop firing, i.e. masking missed one.
        invalid_citations: Cited block numbers that did not exist.
        sections_searched: Stored section labels a single-company task filtered
        on, after aliases — carried so a refusal can name what was searched
        ("no MD&A (Item 7) text") rather than a generic "narrative". Empty
        for corpus-wide tasks, whose sections are per company.
    """

    model_config = ConfigDict(frozen=True)

    claims: list[CitedClaim] = Field(default_factory=list)
    findings: list[CompanyFinding] = Field(default_factory=list)
    refusal: NarrativeRefusal | None = None
    figures_masked: int = 0
    figures_redacted: int = 0
    invalid_citations: int = 0
    sections_searched: frozenset[str] = frozenset()

    @property
    def refused(self) -> bool:
        """Whether the task produced no answer."""
        return self.refusal is not None


class TaskStatus(str, Enum):
    """
    How one task ended.

    REFUSED is not FAILED: a refusal is a correct answer about the corpus, and
    an assembled response must say so rather than dropping the task, so that a
    partially answerable question yields a partial answer plus an explicit note.
    """

    SUCCESS = "success"
    REFUSED = "refused"
    FAILED = "failed"


class TaskOutcome(BaseModel):
    """One task's result, whatever happened to it."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    intent: Intent
    tool: Tool
    status: TaskStatus
    metric_result: MetricResult | None = None
    narrative_result: NarrativeResult | None = None
    resolution_failure: ResolutionFailure | None = None
    message: str | None = None