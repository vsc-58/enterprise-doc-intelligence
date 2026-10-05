"""
src/utils/numbers.py
Numeric parsing shared by the evaluation harness and the query path.

Holds the two primitives that read figures out of a cited source line: the
token parser and the scale multipliers a printed value may need to be
compared against a value in actual dollars. Both were private to
src/eval/grounding.py until Phase 5 needed them at query time for the
scale-coherence quality flag (D29).

They live here rather than in src/eval/ because the query path must not
import from the evaluation side, and a second copy would mean two
calibrations of one instrument — the same argument grounding.py's own
docstring makes about its two callers.

Behaviour is unchanged from the grounding.py originals. Any edit here
changes an instrument already calibrated against Phase 3 results; re-run
scripts/verify_numbers_parity.py before committing one.

Dependencies: standard library only.
"""

import re

# Scale multipliers tried when matching a value against its cited line. A filing
# printing "in millions" shows 54,228 for 54,228,000,000; the model is instructed
# to return actual dollars, so the printed token must be scaled up to compare.
SCALE_MULTIPLIERS: tuple[float, ...] = (1.0, 1e3, 1e6, 1e9)

_NUMBER_TOKEN = re.compile(r"\d[\d,]*\.?\d*")


def numbers_in_line(line: str) -> list[float]:
    """
    Pull every numeric token out of a cited source line.

    Currency symbols, non-breaking spaces and thousands separators are removed
    before parsing. Sign is discarded: accounting statements print negatives in
    parentheses, and the comparison is on magnitude, so a value of -6,327,000,000
    still matches a printed "(6,327)".

    Args:
        line: the model's cited source line, verbatim.

    Returns:
        The magnitudes of every number found, in order of appearance.
    """
    cleaned = line.replace("\u00a0", " ").replace("$", " ")
    numbers: list[float] = []
    for token in _NUMBER_TOKEN.findall(cleaned):
        stripped = token.replace(",", "").rstrip(".")
        if not stripped:
            continue
        try:
            numbers.append(float(stripped))
        except ValueError:
            continue
    return numbers