"""The SQL tool: one financial task as a parameterized, read-only SELECT.

Runs a figure lookup, ranking or filter over the figures table and renders the
answer sentence deterministically.

The numeric invariant: the figure is read from the database and rendered into a
sentence here, in Python. It is never placed in an LLM prompt and never passes
through a model on its way to the user. Everything downstream (the executor,
the synthesizer, the API) carries this text unchanged. A prompt rule saying
"do not change the numbers" would be a request; keeping numbers out of every
prompt is a guarantee.

All filtering, ordering and limiting is bound SQL built on the canonical read,
latest_successful_stmt(). The column allowlist is the MetricField enum: a
column name that is not a member cannot reach the query builder, so the
allowlist is enforced by the type rather than by a check.
"""

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
from src.query.thresholds import ThresholdParseError, parse_threshold
from src.query.resolve import not_extracted_sentence
from src.storage.metadata_store import (
    Document,
    ExtractedRecord,
    get_session,
    latest_successful_stmt,
    unextracted_documents_stmt,
)
from src.storage.quality import QualityFlag, QualityReason, QualityVerdict, assess_record
from src.utils.config import settings
from src.utils.numbers import SCALE_MULTIPLIERS
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
    """Raised when a task cannot be run as written.

    Attributes:
        message: The sentence shown to the user, e.g. asking them to restate an
            unreadable threshold.
    """

    def __init__(self, message: str) -> None:
        """Store the user-facing sentence.

        Args:
            message: Text shown to the user as the task's refusal.
        """
        super().__init__(message)
        self.message = message


def _column(metric: MetricField):
    """Map a MetricField to its ORM column. Membership is the allowlist."""
    return getattr(ExtractedRecord, metric.value)


def _build_statement(task: Task, entity: ResolvedEntity | None) -> tuple[Select, float | None]:
    """Build the bound statement for one financial task.

    Args:
        task: The task, already validated by its schema.
        entity: The resolved company, for single-company intents; None otherwise.

    Returns:
        The Select over (Document, ExtractedRecord), and the parsed threshold in
        dollars for filter intents (None otherwise). The threshold is returned
        rather than recomputed later so the value that selected the rows is the
        same value quoted back to the user.

    Raises:
        SqlPathError: if the task cannot be expressed as a query. Raised before
            anything runs, so "I can't answer that kind of question" is never
            confused with "I have no data for that" (an empty result).
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


def _null_rows(task: Task) -> list[MetricRow]:
    """The companies a ranking or filter leaves out because the figure is missing.

    Named in the answer rather than counted: BAC's operating cash flow is
    null because the true value is negative, so for "lowest operating cash flow"
    the excluded company is probably the real answer, and an unnamed count
    ("1 company was excluded") hid exactly that.

    Two cases, worded the same way to the user ("not extracted from the
    filing"): the latest successful extraction holds null for this figure, or
    no extraction of the filing has succeeded at all. Without the second case,
    a company whose extraction failed would vanish from every list silently.

    Args:
        task: The task being run.

    Returns:
        One row per excluded company (value None), ordered by ticker.
    """
    with get_session() as session:
        pairs = session.execute(
            latest_successful_stmt().where(_column(task.metric).is_(None))
        ).all()
        unextracted = session.execute(unextracted_documents_stmt()).scalars().all()
    rows = [_to_row(doc, record, task.metric) for doc, record in pairs]
    rows += [_unextracted_row(doc, task.metric) for doc in unextracted]
    return sorted(rows, key=lambda row: row.ticker)


def _unextracted_row(doc: Document, metric: MetricField) -> MetricRow:
    """A row for a filing with no successful extraction: no value, nothing checked.

    Args:
        doc: The filing.
        metric: The queried field.

    Returns:
        The row, value None, flagged unverified for a null field.
    """
    return MetricRow(
        ticker=doc.ticker,
        company_name=display_name(doc.company_name, doc.ticker),
        fiscal_year=doc.filing_year,
        fiscal_year_end=None,
        metric=metric,
        value=None,
        quality=QualityFlag(
            field=metric.value,
            verdict=QualityVerdict.UNVERIFIED,
            reason=QualityReason.NULL_FIELD,
        ),
    )


# Intents that range over every company. Only these can drop a company from
# the answer, so only these need the flagged-but-not-shown disclosure.
_CORPUS_INTENTS = frozenset({Intent.FINANCIAL_RANKING, Intent.FINANCIAL_FILTER})


def _to_row(doc: Document, record: ExtractedRecord, metric: MetricField) -> MetricRow:
    """Project one (Document, ExtractedRecord) pair into a MetricRow.

    Args:
        doc: The filing.
        record: Its latest successful extraction.
        metric: The queried field.

    Returns:
        The row, with the field's quality flag attached. The flag is computed
        from the stored evidence only, never from the evaluation answer key.
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


# Flags that say a figure may be at the wrong SCALE. Only a scale error moves a
# figure far enough (1,000x) to cross a threshold or a ranking cut-off it
# otherwise misses by orders of magnitude. evidence_mismatch is left out: the
# value is unconfirmed, not mis-scaled — Tesla's flagged net income even matches
# the answer key — and naming Intel's $1.69 billion as "may belong" in an
# over-$50-billion list was the over-disclosure a live run of the demo found.
_SCALE_REASONS = frozenset({
    QualityReason.SCALE_AMBIGUOUS,
    QualityReason.SCALE_DISAGREEMENT,
    QualityReason.SCALE_NOT_CHECKABLE,
})

# Neighbouring printed scales (thousands, millions, billions) differ by this.
SCALE_STEP = SCALE_MULTIPLIERS[2] / SCALE_MULTIPLIERS[1]


def plausible_values(row: MetricRow) -> list[float]:
    """The values a scale-flagged figure could really have.

    Args:
        row: A scale-flagged row with a value.

    Returns:
        The suspected true value when the quality check computed one (a clear
        majority scale disagrees, as for GS's cash flow); otherwise the value
        one scale step up and one down (an even split, as for BAC).
    """
    if row.quality.suspected_true_value is not None:
        return [row.quality.suspected_true_value]
    return [row.value * SCALE_STEP, row.value / SCALE_STEP]


def _would_appear(value: float, task: Task, shown: list[MetricRow], threshold: float | None) -> bool:
    """Whether a company with this value would be in the answer.

    Args:
        value: A plausible value.
        task: The ranking or filter.
        shown: The rows the answer serves, in answer order.
        threshold: The parsed filter threshold; None for rankings.

    Returns:
        For a filter, whether the value passes the comparison. For a ranking,
        whether it beats the last row shown — or True when the ranking shows
        fewer rows than its limit, since any value would then be listed.
    """
    if task.intent is Intent.FINANCIAL_FILTER:
        return threshold is not None and _COMPARISON_SQL[task.comparison](value, threshold)
    limit = min(task.limit or 1, settings.SQL_RESULT_LIMIT_CAP)
    values = [row.value for row in shown if row.value is not None]
    if len(values) < limit:
        return True
    if task.direction is Direction.HIGHEST:
        return value > min(values)
    return value < max(values)


def flagged_not_shown(
    candidates: list[MetricRow],
    shown: list[MetricRow],
    task: Task,
    threshold: float | None,
) -> list[MetricRow]:
    """The flagged companies a ranking or filter left out that may belong in it.

    Named only when the flag is a scale flag AND a plausible true value would
    put the company in the answer. A warning that cannot be true teaches the
    reader to skip the one that matters. Pure, so the rule is
    testable without a database. A shown row is skipped: its flag is disclosed
    beside it. Null values never reach here (the caller selects non-null rows).

    Args:
        candidates: Every company's non-null row for the queried metric.
        shown: The rows the answer serves.
        task: The ranking or filter that was run.
        threshold: The parsed filter threshold; None for rankings.

    Returns:
        The companies to name, ordered by ticker so the text is stable.
    """
    shown_tickers = {row.ticker for row in shown}
    named = [
        row for row in candidates
        if row.ticker not in shown_tickers
        and row.value is not None
        and row.quality.reason in _SCALE_REASONS
        and any(_would_appear(v, task, shown, threshold) for v in plausible_values(row))
    ]
    return sorted(named, key=lambda row: row.ticker)


def _flagged_outside_result(
    task: Task, shown: list[MetricRow], threshold: float | None
) -> list[MetricRow]:
    """Find flagged companies a ranking or filter dropped.

    render() caveats the ordering when a flagged row is in the result, but a
    figure 1,000x too small can fail a threshold it should pass, and then no
    flagged row is shown to trigger anything. BAC's total assets (stored $3.05 billion,
    ambiguous scale) drop out of "total assets above $1 trillion" exactly this
    way. Costs one extra read of one row per company; no model call.

    Args:
        task: The task that was run.
        shown: The rows the answer serves.
        threshold: The parsed filter threshold; None for rankings.

    Returns:
        Flagged rows not shown that may belong; empty for single-company intents.
    """
    if task.intent not in _CORPUS_INTENTS:
        return []
    with get_session() as session:
        pairs = session.execute(
            latest_successful_stmt().where(_column(task.metric).is_not(None))
        ).all()
    candidates = [_to_row(doc, record, task.metric) for doc, record in pairs]
    return flagged_not_shown(candidates, shown, task, threshold)


def run_financial_task(task: Task, entity: ResolvedEntity | None) -> MetricResult:
    """Execute one financial task.

    Args:
        task: A task whose intent maps to the SQL tool.
        entity: The resolved company, or None for corpus-wide intents.

    Returns:
        A MetricResult, possibly with no rows. An empty result is a valid
        outcome ("no data for that") and is kept apart from an unrunnable task
        ("can't answer that kind of question"), which raises before any query
        is issued.

    Raises:
        SqlPathError: if the task cannot be expressed as a query.
    """
    if policy_for(task.intent).tool is not Tool.SQL:
        raise SqlPathError("That question doesn't belong to the figures table.")

    stmt, threshold = _build_statement(task, entity)
    with get_session() as session:
        pairs = session.execute(stmt).all()

    rows = [_to_row(doc, record, task.metric) for doc, record in pairs]

    excluded = _null_rows(task) if task.intent in _CORPUS_INTENTS else []
    flagged_dropped = _flagged_outside_result(task, rows, threshold)
    mixed_years = len({row.fiscal_year for row in rows}) > 1

    logger.info(
        "sql_task_complete",
        intent=task.intent.value,
        metric=task.metric.value,
        rows=len(rows),
        excluded_null=[row.ticker for row in excluded],
        flagged_not_shown=[row.ticker for row in flagged_dropped],
        mixed_years=mixed_years,
    )
    return MetricResult(
        rows=rows,
        flagged_not_shown=flagged_dropped,
        excluded_null=excluded,
        excluded_null_count=len(excluded),
        mixed_years=mixed_years,
        threshold_value=threshold,
    )


def format_value(value: float | None) -> str:
    """Render a dollar figure at a readable scale.

    Args:
        value: The figure in actual dollars.

    Returns:
        Text such as "$365.82 billion", or "not extracted" for None. Never "not
        reported": a missing value means the extraction did not find it, not
        that the filing lacks it.
    """
    if value is None:
        return "not extracted"
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
    """One disclosure line per flagged company a ranking or filter left out.

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
        line = (
            f"Not included: {row.company_name} ({row.ticker}) — its {label} {verb} "
            "flagged, and at a 1,000x different scale it would be in this list."
        )
        lines.append(f"{line} {note}" if note else line)
    return lines


def _null_lines(task: Task, result: MetricResult) -> list[str]:
    """One line per company left out because its figure was not extracted.

    Says what the gap means for THIS question: a missing figure could be the
    lowest, the highest, or pass the threshold — and for BAC's operating cash
    flow (null because the true value is negative) it probably is the lowest.

    Args:
        task: The ranking or filter that was run.
        result: Its result.

    Returns:
        The lines, possibly empty.
    """
    label = METRIC_LABELS[task.metric]
    verb = "were" if task.metric in _PLURAL_METRICS else "was"
    if task.intent is Intent.FINANCIAL_RANKING:
        side = "higher" if task.direction is Direction.HIGHEST else "lower"
        consequence = f"so it may be {side} than any figure shown"
    else:
        consequence = "so whether it meets this threshold is unknown"
    return [
        f"Not included: {row.company_name} ({row.ticker}) — its {label} {verb} "
        f"not extracted from the filing, {consequence}."
        for row in result.excluded_null
    ]


def render(task: Task, result: MetricResult, entity: ResolvedEntity | None) -> str:
    """Render the answer text for a completed financial task.

    Deterministic string building. This function is the reason no stored figure
    ever reaches an LLM, so it must stay free of model calls.

    Multi-row answers print each row's fiscal year and, when the years differ,
    append one sentence saying so: this corpus holds one filing per company, so a
    ranking that hid the mixture would be misleading.

    Rankings and filters also name, even when the result is empty, every
    company they left out that may belong: scale-flagged figures a 1,000x
    correction would bring in, and figures that were not extracted, with what
    the gap means for the question.

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
            # Defensive: the resolver refuses a company with no successful
            # extraction before this runs. Never "the filing does not report it".
            return not_extracted_sentence(entity)
        empty = [f"No company in the corpus matches that {label} criterion."]
        return "\n".join(empty + _not_shown_lines(task, result) + _null_lines(task, result))

    lines: list[str] = []
    if task.intent is Intent.FINANCIAL_METRIC:
        row = result.rows[0]
        verb = "were" if task.metric in _PLURAL_METRICS else "was"
        if row.value is None:
            # Says it in the sentence, so the null-field note below is skipped.
            lines.append(
                f"{row.company_name}'s {label} for {_period(row)} {verb} not "
                "extracted from the filing, so I can't give it."
            )
        else:
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
        if row.quality.verdict is not QualityVerdict.VERIFIED and row.value is not None:
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

    lines.extend(_null_lines(task, result))

    if result.mixed_years:
        years = sorted({row.fiscal_year for row in result.rows})
        lines.append(
            f"These figures come from different fiscal years ({years[0]}–{years[-1]}), "
            "because the corpus holds one annual filing per company. They are not "
            "like-for-like comparisons."
        )

    return "\n".join(lines)