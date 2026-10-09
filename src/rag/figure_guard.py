# src/rag/figure_guard.py
# Module: Figure guard for the narrative path
# Purpose: Keep the five stored financial metrics out of narrative answers.
#          Masks matching values in retrieved text before the LLM sees it, and
#          redacts any that still appear in a claim (D27, D41).
# Depends on: re, src.query.schemas, src.storage.metadata_store,
#             src.storage.quality, src.utils.config, src.utils.logger,
#             src.utils.numbers
#
# Why: every financial figure in an answer must have one auditable source, the
# structured store. MD&A text carries the same figures, so without this the RAG
# answer can restate revenue beside the SQL answer — at a different year, a
# misread scale, or (BAC) the correct value beside a stored value that is wrong.
#
# Matching uses stored evidence only, never ground truth (D29). For a field whose
# quality flag reports a scale problem, the stored value is also tried at x1,000
# and /1,000, so the guard still recognises the figure where the store is off by
# that factor — the case where two conflicting numbers would otherwise reach the
# user.
#
# Only the five MetricFields count as figures. Percentages, years, segment
# figures and prior-year amounts pass: those are not values the SQL tool owns.

import re
from collections.abc import Iterable
from functools import lru_cache

from src.query.schemas import MetricField
from src.storage.metadata_store import latest_successful_records
from src.storage.quality import QualityReason, assess_record
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.numbers import SCALE_MULTIPLIERS

logger = get_logger(__name__)

PLACEHOLDER = "[figure withheld]"

_SCALE_WORDS: dict[str, float] = {
    "thousand": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
}

# A possible amount: optional "$", a number (thousands-separated or plain,
# optional decimals), optional scale word. The lookbehind stops a match starting
# inside a longer token ("FY2021", "10-K", "3.14").
_AMOUNT = re.compile(
    r"(?<![\w.])"
    r"(?P<dollar>\$\s?)?"
    r"(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|mn|bn)s?\b)?",
    re.IGNORECASE,
)

# Flags under which the stored value may be off by a factor of 1,000.
_SCALE_SUSPECT = frozenset({QualityReason.SCALE_AMBIGUOUS, QualityReason.SCALE_DISAGREEMENT})


class FigureGuardError(RuntimeError):
    """Raised when the stored values the guard matches against cannot be loaded."""


@lru_cache(maxsize=1)
def load_guard_values() -> dict[str, frozenset[float]]:
    """
    Load, per ticker, the dollar values the guard treats as stored figures.

    Read once per process: the query path is read-only, and the guard must not
    add a database read to every narrative task. Clear with
    load_guard_values.cache_clear() after re-extraction.

    Returns:
        Ticker to the magnitudes of its five stored metrics, plus x1,000 and
        /1,000 variants for fields flagged with a scale problem.

    Raises:
        FigureGuardError: if the structured store cannot be read.
    """
    try:
        pairs = latest_successful_records()
    except Exception as exc:
        logger.error(
            "figure_guard_load_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise FigureGuardError("Could not load stored figures for the guard") from exc

    values: dict[str, frozenset[float]] = {}
    for document, record in pairs:
        flags = assess_record(record)
        targets: set[float] = set()
        for metric in MetricField:
            stored = getattr(record, metric.value)
            if stored is None or stored == 0:
                continue
            magnitude = abs(float(stored))
            targets.add(magnitude)
            flag = flags.get(metric.value)
            if flag is not None and flag.reason in _SCALE_SUSPECT:
                targets.update({magnitude * 1e3, magnitude / 1e3})
        values[document.ticker] = frozenset(targets)

    logger.info(
        "figure_guard_loaded",
        companies=len(values),
        values=sum(len(v) for v in values.values()),
    )
    return values


def _readings(number: float, scale_word: str | None) -> Iterable[float]:
    """
    Every dollar value a printed amount could denote.

    An explicit scale word fixes the reading. Without one, the scale may sit in
    a table header elsewhere ("in millions"), so every SCALE_MULTIPLIER is tried
    — the same multipliers the extraction grounding check uses.

    Args:
        number: The parsed magnitude.
        scale_word: The scale word that followed it, if any.

    Returns:
        Candidate dollar values.
    """
    if scale_word:
        return (number * _SCALE_WORDS[scale_word.lower()],)
    return tuple(number * multiplier for multiplier in SCALE_MULTIPLIERS)


def _is_stored(readings: Iterable[float], targets: frozenset[float]) -> bool:
    """
    Whether any reading matches any target within the relative tolerance.

    Args:
        readings: Candidate dollar values for one printed amount.
        targets: The company's stored values.

    Returns:
        True on a match.
    """
    tolerance = settings.FIGURE_GUARD_REL_TOLERANCE
    return any(
        abs(reading - target) <= tolerance * target
        for reading in readings
        for target in targets
    )


def withhold_figures(text: str, targets: frozenset[float]) -> tuple[str, int]:
    """
    Replace every amount in text that matches a stored value with PLACEHOLDER.

    Only tokens that look like money are candidates: a "$", a thousands
    separator, or a scale word. A bare "27" or "2021" is never one, which keeps
    percentages and years out — read at x1,000,000, "2021" would otherwise match
    a company with $2.02B of net income.

    Args:
        text: Retrieved chunk text or a model-written claim.
        targets: The company's stored values, from load_guard_values().

    Returns:
        The text with matches replaced, and the number replaced.
    """
    if not targets:
        return text, 0

    replaced = 0

    def substitute(match: re.Match[str]) -> str:
        nonlocal replaced
        raw = match.group("number")
        is_money = bool(match.group("dollar") or match.group("scale") or "," in raw)
        if not is_money:
            return match.group(0)
        number = float(raw.replace(",", ""))
        if not _is_stored(_readings(number, match.group("scale")), targets):
            return match.group(0)
        replaced += 1
        return PLACEHOLDER

    return _AMOUNT.sub(substitute, text), replaced