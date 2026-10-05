# src/query/sql_path.py
# Module: Structured figure lookup
# Purpose: Run one financial task as a parameterized read-only SELECT and render
#          its answer text deterministically.
# Depends on: sqlalchemy, src.query.*, src.storage.*, src.utils.*
#
# THE NUMERIC INVARIANT (D27): the figure is read from the database and rendered
# into a sentence here, in Python. It is never placed in an LLM prompt and never
# passes through a model on its way to the user. Everything downstream — the
# executor, the assembler, the API — carries this text unchanged.
#
# All filtering, ordering and limiting is bound SQL built on the canonical read
# (D31). The column allowlist is the MetricField enum: a column name that is not
# a member cannot reach the query builder, so the allowlist is enforced by the
# type rather than by a check (D25).

from sqlalchemy import Select, and_

from src.query.intents import policy_for
from src.query.schemas import (
    Comparison,
    Direction,
    Intent,
    MetricField,
    MetricResult,
    MetricRow,
    ResolvedEntity,
    Task,
    Tool,
)
from src.query.thresholds import ThresholdParseError, parse_threshold  # parse_threshold still used in _build_statement
from src.storage.metadata_store import Document, ExtractedRecord, get_session, latest_successful_stmt
from src.storage.quality import QualityVerdict, assess_record
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

_COMPARISON_SQL = {
    Comparison.GT: lambda col, v: col > v,
    Comparison.GTE: lambda col, v: col >= v,
    Comparison.LT: lambda col, v: col < v,
    Comparison.LTE: lambda col, v: col <= v,
}

_COMPARISON_WORDS = {
    Comparison.GT: "above", Comparison.GTE: "at or above",
    Comparison.LT: "below", Comparison.LTE: "at or below",
}

METRIC_LABELS = {
    MetricField.TOTAL_REVENUE: "total revenue",
    MetricField.NET_INCOME: "net income",
    MetricField.TOTAL_ASSETS: "total assets",
    MetricField.TOTAL_LIABILITIES: "total liabilities",
    MetricField.OPERATING_CASH_FLOW: "operating cash flow",
}

# Metrics that read as plural, for verb agreement in rendered answers.
_PLURAL_METRICS = frozenset({MetricField.TOTAL_ASSETS, MetricField.TOTAL_LIABILITIES})

class SqlPathError(Exception):
    """Raised when a task cannot be run as written. Carries the user-facing text."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _column(metric: MetricField):
    """Map a MetricField to its ORM column. Membership is the allowlist."""
    return getattr(ExtractedRecord, metric.value)


def _build_statement(task: Task, entity: ResolvedEntity | None) -> tuple[Select, float | None]:
    """
    Build the bound statement for one financial task.

    Args:
        task: The task, already validated by its schema.
        entity: The resolved company, for single-company intents; None otherwise.

    Returns:
        The Select over (Document, ExtractedRecord), and the parsed threshold in
        dollars for filter intents (None otherwise). The threshold is returned
        rather than recomputed later so the value that selected the rows is the
        same value quoted back to the user.

    Raises:
        SqlPathError: if the task cannot be expressed — case (a), raised before
            anything runs, so a bounded-capability message is never confused with
            an empty result (case b).
    """
    stmt = latest_successful_stmt()
    column = _column(task.metric)

    if task.intent is Intent.FINANCIAL_METRIC:
        if entity is None:
            raise SqlPathError("I need to know which company you mean.")
        return stmt.where(Document.id == entity.document_id), None

    if task.intent is Intent.FINANCIAL_RANKING:
        order = column.desc() if task.direction is Direction.HIGHEST else column.asc()
        limit = min(task.limit or 1, settings.SQL_RESULT_LIMIT_CAP)
        return stmt.where(column.is_not(None)).order_by(order).limit(limit), None

    if task.intent is Intent.FINANCIAL_FILTER:
        try:
            threshold = parse_threshold(task.threshold_text or "")
        except ThresholdParseError as exc:
            logger.warning("threshold_unparsed", text=task.threshold_text, error=str(exc))
            raise SqlPathError(
                f"I couldn't read {task.threshold_text!r} as an amount, so I'd rather "
                "not guess at the threshold. Could you restate it, for example as "
                "'$50 billion'?"
            ) from exc

        predicate = _COMPARISON_SQL[task.comparison](column, threshold)
        return (
            stmt.where(and_(column.is_not(None), predicate))
            .order_by(column.desc())
            .limit(settings.SQL_RESULT_LIMIT_CAP),
            threshold,
        )

    raise SqlPathError("That isn't a question I can answer from the figures table.")


def _count_null_rows(task: Task) -> int:
    """
    Count companies excluded because the queried metric is null.

    Reported rather than silently dropped: BAC's operating cash flow is null
    because the true value is negative, so "lowest operating cash flow" would
    otherwise omit the correct answer without saying so.

    Args:
        task: The task being run.

    Returns:
        The number of excluded rows.
    """
    with get_session() as session:
        rows = session.execute(
            latest_successful_stmt().where(_column(task.metric).is_(None))
        ).all()
    return len(rows)


def run_financial_task(task: Task, entity: ResolvedEntity | None) -> MetricResult:
    """
    Execute one financial task.

    Args:
        task: A task whose intent maps to the SQL tool.
        entity: The resolved company, or None for corpus-wide intents.

    Returns:
        A MetricResult, possibly with no rows. An empty result is a valid
        outcome (case b) and is distinguished from an unrunnable task (case a),
        which raises before any query is issued.

    Raises:
        SqlPathError: case (a) — the task cannot be expressed as a query.
    """
    if policy_for(task.intent).tool is not Tool.SQL:
        raise SqlPathError("That question doesn't belong to the figures table.")

    stmt, threshold = _build_statement(task, entity)
    with get_session() as session:
        pairs = session.execute(stmt).all()

    rows = [
        MetricRow(
            ticker=doc.ticker,
            company_name=display_name(doc.company_name, doc.ticker),
            fiscal_year=doc.filing_year,
            fiscal_year_end=record.fiscal_year_end,
            metric=task.metric,
            value=getattr(record, task.metric.value),
            quality=assess_record(record)[task.metric.value],
        )
        for doc, record in pairs
    ]

    excluded = (
        _count_null_rows(task)
        if task.intent in {Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER}
        else 0
    )
    mixed_years = len({row.fiscal_year for row in rows}) > 1

    logger.info(
        "sql_task_complete",
        intent=task.intent.value,
        metric=task.metric.value,
        rows=len(rows),
        excluded_null=excluded,
        mixed_years=mixed_years,
    )
    return MetricResult(
        rows=rows,
        excluded_null_count=excluded,
        mixed_years=mixed_years,
        threshold_value=threshold,
        sql_preview=str(stmt.compile(compile_kwargs={"literal_binds": False})),
    )


def format_value(value: float | None) -> str:
    """
    Render a dollar figure at a readable scale.

    Args:
        value: The figure in actual dollars.

    Returns:
        Text such as "$365.82 billion", or "not reported" for None.
    """
    if value is None:
        return "not reported"
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if magnitude >= threshold:
            scaled = f"{value / threshold:,.2f}".rstrip("0").rstrip(".")
            return f"${scaled} {suffix}"
    return f"${value:,.0f}"


def _period(row: MetricRow) -> str:
    """Render a row's fiscal year with its period end, which can differ by a year."""
    if row.fiscal_year_end:
        return f"fiscal year {row.fiscal_year} (period ending {row.fiscal_year_end})"
    return f"fiscal year {row.fiscal_year}"


def render(task: Task, result: MetricResult, entity: ResolvedEntity | None) -> str:
    """
    Render the answer text for a completed financial task.

    Deterministic string building. This function is the reason no stored figure
    ever reaches an LLM (D27), so it must stay free of model calls.

    Multi-row answers print each row's fiscal year and, when the years differ,
    append one sentence saying so: this corpus holds one filing per company, so a
    ranking that hid the mixture would be misleading (D33).

    Args:
        task: The task that was run.
        result: Its result.
        entity: The resolved company, for single-company intents.

    Returns:
        The answer text, including any quality disclosure.
    """
    label = METRIC_LABELS[task.metric]

    if not result.rows:
        if entity is not None:
            return (
                f"I hold one {entity.company_name} filing, for fiscal year "
                f"{entity.filing_year}, and it does not report {label}."
            )
        return f"No company in the corpus matches that {label} criterion."



    lines: list[str] = []
    if task.intent is Intent.FINANCIAL_METRIC:
        row = result.rows[0]
        verb = "were" if task.metric in _PLURAL_METRICS else "was"
        lines.append(
            f"{row.company_name}'s {label} for {_period(row)} {verb} "
            f"{format_value(row.value)}."
        )
    else:
        if task.intent is Intent.FINANCIAL_RANKING:
            word = "highest" if task.direction is Direction.HIGHEST else "lowest"
            lines.append(f"By {label}, {word} first:")
        else:
            word = _COMPARISON_WORDS[task.comparison]
            lines.append(
                f"{len(result.rows)} "
                f"{'company' if len(result.rows) == 1 else 'companies'} had {label} "
                f"{word} {format_value(result.threshold_value)}:"
            )
        for row in result.rows:
            lines.append(
                f"  {row.company_name} ({row.ticker}), FY{row.fiscal_year}: "
                f"{format_value(row.value)}"
            )

    for row in result.rows:
        if row.quality.verdict is not QualityVerdict.VERIFIED:
            note = row.quality.note()
            if note:
                lines.append(f"{row.ticker} — {note}")

    if task.intent in {Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER} and any(
        row.quality.verdict is not QualityVerdict.VERIFIED for row in result.rows
    ):
        lines.append(
            "Because at least one figure here is flagged, the ordering itself may "
            "be wrong — a mis-scaled value sorts into the wrong position."
        )

    if result.excluded_null_count:
        lines.append(
            f"{result.excluded_null_count} "
            f"{'company was' if result.excluded_null_count == 1 else 'companies were'} "
            f"excluded because {label} was not extracted from the filing."
        )

    if result.mixed_years:
        years = sorted({row.fiscal_year for row in result.rows})
        lines.append(
            f"These figures come from different fiscal years ({years[0]}–{years[-1]}), "
            "because the corpus holds one annual filing per company. They are not "
            "like-for-like comparisons."
        )

    return "\n".join(lines)