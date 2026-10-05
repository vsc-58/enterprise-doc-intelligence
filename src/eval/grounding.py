"""
src/eval/grounding.py
Evidence verification for S3's per-field source lines.

Two independent verdicts on a cited figure:
  - check_grounding            : is the cited line real? (fabrication detector)
  - check_evidence_consistency : does the cited line contain the value?
                                 (misattribution detector)

Both are pure text functions over an evidence map. They live here rather than in
evaluate.py because BOTH the offline evaluator and the write-time extraction
pipeline call them, and importing evaluate.py from extractor.py is circular
(evaluate.py imports the extractor to run strategies). One implementation, two
callers, no cycle — a second copy would mean two calibrations of one instrument,
which is the failure D12 was about.

Dependencies: src.utils.config, src.utils.logger, src.utils.numbers.
"""

import re
from src.utils.numbers import SCALE_MULTIPLIERS, numbers_in_line
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


_EVIDENCE_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def _flatten_for_grounding(text: str) -> str:
    """
    Reduce text to a form comparable across cosmetic differences.

    Lowercases, removes ALL whitespace, and strips currency symbols. The text
    extractor runs labels into their values ('total assets$85,501'), so a
    faithful quote cannot preserve the original spacing — and the model
    normalises it back to readable form. Neither difference can hide a
    fabricated citation: the label text, the digits, and their order all still
    have to match.

    Args:
        text: the string to normalise.

    Returns:
        The flattened comparison form.
    """
    return "".join(text.lower().split()).replace("$", "")


def check_grounding(
    evidence: dict[str, dict] | None, source_text: str
) -> dict[str, bool | None]:
    """
    A quoted line that is not a substring of the source was fabricated, so the
    value it supports cannot be trusted regardless of whether it happens to be
    correct. Comparison ignores whitespace and currency symbols (see
    _flatten_for_grounding): the extractor mangles spacing around table values,
    so requiring exact spacing would flag correct citations as fabricated.
    """
    if not evidence:
        return {}

    haystack = _flatten_for_grounding(source_text)
    results: dict[str, bool | None] = {}
    for field, record in evidence.items():
        line = record.get("source_line")
        if not line:
            results[field] = None
            continue
        needle = _flatten_for_grounding(str(line))
        results[field] = needle in haystack
    return results




def check_evidence_consistency(
    evidence: dict[str, dict] | None,
) -> dict[str, bool | None]:
    """
    Report, per field, whether the extracted value appears in its own cited line.

    The second and independent grounding verdict. check_grounding answers "was
    this line fabricated?"; this answers "does the line the model cited actually
    contain the number the model reported?" — a real line cited for a figure it
    does not carry passes the first check and fails this one.

    Found on the eval-10: INTC/WMT/AMZN total_liabilities, where no standalone
    total-liabilities line exists (Phase 1 D5) and the model cited "Total
    liabilities and stockholders' equity" for a figure it had derived; and INTC
    net_income, where the model returned NetIncomeLoss (1,689) while quoting the
    ProfitLoss line (1,675). All four scored `correct` and passed check_grounding.

    A False is a lineage failure, not necessarily a wrong value — the figure may
    be right and merely miscited. It marks the figure as unverified.

    Args:
        evidence: the per-field evidence map (field -> {value, source_line}), or
            None for a non-evidence strategy.

    Returns:
        Field -> True (value found in the line), False (not found), or None (not
        checkable: no value or no cited line). Empty dict when evidence is absent.
    """
    if not evidence:
        return {}

    results: dict[str, bool | None] = {}
    for field, record in evidence.items():
        value = record.get("value")
        line = record.get("source_line")
        if value is None or not line:
            results[field] = None
            continue

        target = abs(float(value))
        tolerance = max(target * settings.EVIDENCE_MATCH_REL_TOLERANCE, 1e-6)
        printed = numbers_in_line(str(line))
        results[field] = any(
            abs(number * scale - target) <= tolerance
            for number in printed
            for scale in SCALE_MULTIPLIERS
        )
    return results