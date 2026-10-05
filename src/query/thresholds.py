# src/query/thresholds.py
# Module: Threshold text to dollars
# Purpose: Convert the literal threshold phrase a planner copied into a number.
# Depends on: src.utils.logger
#
# In Python, never in the model. Scale words are exactly where extraction's worst
# silent failure came from — a figure correct in digits and wrong by three orders
# of magnitude (D29) — and a filter threshold has the same failure mode with none
# of the evidence columns that caught it there. A parse that is not certain fails
# loudly rather than guessing.

import re

from src.utils.logger import get_logger

logger = get_logger(__name__)

SCALE_WORDS: dict[str, float] = {
    "k": 1e3, "thousand": 1e3, "thousands": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12, "trillions": 1e12,
}

_PATTERN = re.compile(
    r"^\s*\$?\s*(?P<number>\d[\d,]*\.?\d*)\s*(?P<scale>[a-z]+)?\s*(?:dollars|usd)?\s*$"
)


class ThresholdParseError(ValueError):
    """Raised when a threshold phrase cannot be read with certainty."""


def parse_threshold(text: str) -> float:
    """
    Convert a threshold phrase into a value in actual dollars.

    Accepts "$50 billion", "50bn", "1.5 trillion", "100 million dollars",
    "50,000,000,000". A bare number is taken as dollars, matching the units the
    store holds, so "50000000000" and "$50 billion" agree.

    Args:
        text: The phrase as the planner copied it from the question.

    Returns:
        The threshold in dollars.

    Raises:
        ThresholdParseError: if the phrase does not parse, or carries a scale word
            that is not recognised. Refusing is correct here: a filter run at the
            wrong scale returns a confident, wrong company list.
    """
    match = _PATTERN.match(text.lower().replace("\u00a0", " "))
    if match is None:
        raise ThresholdParseError(f"could not read a threshold from {text!r}")

    try:
        number = float(match.group("number").replace(",", ""))
    except ValueError as exc:
        raise ThresholdParseError(f"could not read a number from {text!r}") from exc

    scale_word = match.group("scale")
    if scale_word is None:
        return number
    if scale_word not in SCALE_WORDS:
        raise ThresholdParseError(f"unrecognised scale {scale_word!r} in {text!r}")

    value = number * SCALE_WORDS[scale_word]
    logger.debug("threshold_parsed", text=text, value=value)
    return value