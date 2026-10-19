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


# Intents that range over every company. Only these can drop a company from
# the answer, so only these need the flagged-but-not-shown disclosure.
_CORPUS_INTENTS = frozenset({Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER})


def _to_row(doc: Document, record: ExtractedRecord, metric: MetricField) -> MetricRow:
    """
    Project one (Document, ExtractedRecord) pair into a MetricRow.

    Args:
        doc: The filing.
        record: Its latest successful extraction.
        metric: The queried field.

    Returns:
        The row, with the field's quality flag attached (D29).
    """
    return MetricRow(
        ticker=doc.ticker,
        company_name=display_name(doc.company_name, doc.ticker),
        fiscal_year=doc.filing_year,
        fiscal_year_end=record.fiscal_year_end,
        metric=metric,
        value=getattr(record, metric.value),
        quality=assess_record(record)[metric.value],
    )


def flagged_not_shown(candidates: list[MetricRow], shown: list[MetricRow]) -> list[MetricRow]:
    """
    Select the flagged companies that a ranking or filter left out.

    Pure, so the rule is testable without a database. A row the answer already
    shows is excluded: its flag is disclosed beside it. Null values never reach
    here (the caller selects non-null rows); they are counted separately by
    _count_null_rows.

    Args:
        candidates: Every company's non-null row for the queried metric.
        shown: The rows the answer serves.

    Returns:
        The non-verified candidates not in the answer, ordered by ticker so the
        rendered text is stable between runs.
    """
    shown_tickers = {row.ticker for row in shown}
    dropped = [
        row for row in candidates
        if row.ticker not in shown_tickers
        and row.quality.verdict is not QualityVerdict.VERIFIED
    ]
    return sorted(dropped, key=lambda row: row.ticker)


def _flagged_outside_result(task: Task, shown: list[MetricRow]) -> list[MetricRow]:
    """
    Find flagged companies a ranking or filter dropped.

    D36 caveats the ordering when a flagged row is in the result, but a figure
    1,000x too small can fail a threshold it should pass, and then no flagged
    row is shown to trigger anything. BAC's total assets (stored $3.05 billion,
    ambiguous scale) drop out of "total assets above $1 trillion" exactly this
    way. Costs one extra read of one row per company; no model call.

    Args:
        task: The task that was run.
        shown: The rows the answer serves.

    Returns:
        Flagged rows not shown; empty for single-company intents.
    """
    if task.intent not in _CORPUS_INTENTS:
        return []
    with get_session() as session:
        pairs = session.execute(
            latest_successful_stmt().where(_column(task.metric).is_not(None))
        ).all()
    candidates = [_to_row(doc, record, task.metric) for doc, record in pairs]
    return flagged_not_shown(candidates, shown)


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

    rows = [_to_row(doc, record, task.metric) for doc, record in pairs]

    excluded = _count_null_rows(task) if task.intent in _CORPUS_INTENTS else 0
    flagged_dropped = _flagged_outside_result(task, rows)
    mixed_years = len({row.fiscal_year for row in rows}) > 1

    logger.info(
        "sql_task_complete",
        intent=task.intent.value,
        metric=task.metric.value,
        rows=len(rows),
        excluded_null=excluded,
        flagged_not_shown=[row.ticker for row in flagged_dropped],
        mixed_years=mixed_years,
    )
    return MetricResult(
        rows=rows,
        flagged_not_shown=flagged_dropped,
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


def _not_shown_lines(task: Task, result: MetricResult) -> list[str]:
    """
    One disclosure line per flagged company a ranking or filter left out.

    Worded as "may belong", never "belongs": for an ambiguous scale (BAC) there
    is no trusted value to say which side of the threshold it falls on.

    Args:
        task: The task that was run.
        result: Its result.

    Returns:
        The lines, possibly empty.
    """
    label = METRIC_LABELS[task.metric]
    verb = "are" if task.metric in _PLURAL_METRICS else "is"
    lines: list[str] = []
    for row in result.flagged_not_shown:
        note = row.quality.note()
        line = f"{row.ticker} — not listed, but its {label} {verb} flagged, so it may belong here."
        lines.append(f"{line} {note}" if note else line)
    return lines


def render(task: Task, result: MetricResult, entity: ResolvedEntity | None) -> str:
    """
    Render the answer text for a completed financial task.

    Deterministic string building. This function is the reason no stored figure
    ever reaches an LLM (D27), so it must stay free of model calls.

    Multi-row answers print each row's fiscal year and, when the years differ,
    append one sentence saying so: this corpus holds one filing per company, so a
    ranking that hid the mixture would be misleading (D33).

    Rankings and filters also name every flagged company they left out, even
    when the result is empty: a mis-scaled figure decides membership, so a
    dropped flagged company is as unreliable as a shown one (D36 extended).

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
        empty = [f"No company in the corpus matches that {label} criterion."]
        return "\n".join(empty + _not_shown_lines(task, result))

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

    lines.extend(_not_shown_lines(task, result))

    if task.intent in _CORPUS_INTENTS and any(
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