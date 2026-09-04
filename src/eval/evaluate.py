"""
src/eval/evaluate.py
Scores extraction output against the XBRL ground-truth answer key.

Two levels:
  - score_extraction()  : one document -> per-field outcome (correct / wrong /
                          missing / n_a) plus the relative error for numerics.
  - evaluate_strategy() : one strategy across the eval set -> per-field
                          precision / recall / F1 with explicit denominators,
                          token cost, a provenance split (clean Item 8 vs
                          fallback input), and — for evidence strategies — the
                          grounding rate.

The n_a outcome is load-bearing: where the answer key has no value (bank revenue
excluded by D4, untagged liabilities by D5), the field is EXCLUDED from the
denominator rather than counted against the model. Scoring a null truth as a
miss would punish the model for correctly reporting an absent value.

Ground truth is read here only; it is never written and never enters the
extraction store (the boundary rule).

Dependencies: pydantic v2, src.extraction.*, src.utils.*
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.extraction.extractor import (
    ExtractionResult,
    assemble_input,
    load_sections,
)
from src.extraction.schemas import FilingExtraction
from src.utils.config import settings
from src.utils.logger import get_logger

from src.eval.grounding import (
    _flatten_for_grounding,
    check_evidence_consistency,
    check_grounding,
)

logger = get_logger(__name__)

NUMERIC_FIELDS: tuple[str, ...] = (
    "total_revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "operating_cash_flow",
)
STRING_FIELDS: tuple[str, ...] = ("company_name", "fiscal_year_end")
SCORED_FIELDS: tuple[str, ...] = NUMERIC_FIELDS + STRING_FIELDS

CORRECT, WRONG, MISSING, N_A = "correct", "wrong", "missing", "n_a"

import re

# Scale multipliers tried when matching a value against its cited line. A filing
# printing "in millions" shows 54,228 for 54,228,000,000; the model is instructed
# to return actual dollars, so the printed token must be scaled up to compare.
_EVIDENCE_SCALES: tuple[float, ...] = (1.0, 1e3, 1e6, 1e9)

_EVIDENCE_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def _numbers_in_line(line: str) -> list[float]:
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
    for token in _EVIDENCE_NUMBER.findall(cleaned):
        stripped = token.replace(",", "").rstrip(".")
        if not stripped:
            continue
        try:
            numbers.append(float(stripped))
        except ValueError:
            continue
    return numbers


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
        printed = _numbers_in_line(str(line))
        results[field] = any(
            abs(number * scale - target) <= tolerance
            for number in printed
            for scale in _EVIDENCE_SCALES
        )
    return results

class FieldScore(BaseModel):
    """
    The outcome of one field on one document.

    Attributes:
        outcome: correct | wrong | missing | n_a.
        extracted: what the model returned.
        truth: the ground-truth value.
        rel_error: |extracted - truth| / |truth| for numerics that were
            compared; None otherwise. Recorded even when the outcome is
            `correct`, so strategies that land digits exactly can be
            distinguished from ones that merely land inside tolerance.
    """

    model_config = ConfigDict(str_strip_whitespace=False)

    outcome: str
    extracted: float | str | None = None
    truth: float | str | None = None
    rel_error: float | None = None


def normalize_string(value: str | None) -> str | None:
    """
    Normalise a string field for comparison.

    Lowercases, removes periods and commas (so 'Netflix, Inc.' matches
    'NETFLIX INC'), and collapses internal whitespace. Punctuation is stripped
    throughout, not just at the ends — registrant names differ in internal
    punctuation between the filing text and the XBRL entity name.

    Args:
        value: the raw string, or None.

    Returns:
        The normalised string, or None if the input was None/blank.
    """
    if value is None:
        return None
    cleaned = value.lower().replace(".", " ").replace(",", " ")
    collapsed = " ".join(cleaned.split())
    return collapsed or None


def _numeric_outcome(
    extracted: float | None, truth: float | None
) -> tuple[str, float | None]:
    """
    Classify a numeric field and compute its relative error.

    Args:
        extracted: the model's value.
        truth: the ground-truth value.

    Returns:
        (outcome, rel_error). rel_error is None when no comparison was made.
    """
    if truth is None:
        return N_A, None
    if extracted is None:
        return MISSING, None
    if truth == 0:
        return (CORRECT, 0.0) if extracted == 0 else (WRONG, None)
    rel_error = abs(extracted - truth) / abs(truth)
    outcome = CORRECT if rel_error <= settings.NUMERIC_REL_TOLERANCE else WRONG
    return outcome, rel_error


def _string_outcome(extracted: str | None, truth: str | None) -> str:
    """
    Classify a string field after normalisation.

    Args:
        extracted: the model's value.
        truth: the ground-truth value.

    Returns:
        One of correct | wrong | missing | n_a.
    """
    norm_truth = normalize_string(truth if isinstance(truth, str) else None)
    if norm_truth is None:
        return N_A
    norm_extracted = normalize_string(extracted)
    if norm_extracted is None:
        return MISSING
    return CORRECT if norm_extracted == norm_truth else WRONG


def score_extraction(
    extracted: FilingExtraction | None, truth: dict
) -> dict[str, FieldScore]:
    """
    Score one document's extraction against its ground-truth record.

    A None extraction (schema validation failed) is scored as `missing` on every
    field that has ground truth — the model produced no usable value, which
    costs recall but not precision.

    Args:
        extracted: the validated extraction, or None on validation failure.
        truth: the ground-truth record for this document.

    Returns:
        Mapping of field name -> FieldScore for every scored field.
    """
    scores: dict[str, FieldScore] = {}

    for field in NUMERIC_FIELDS:
        truth_value = truth.get(field)
        if extracted is None:
            outcome = N_A if truth_value is None else MISSING
            scores[field] = FieldScore(
                outcome=outcome, extracted=None, truth=truth_value
            )
            continue
        value = getattr(extracted, field)
        outcome, rel_error = _numeric_outcome(value, truth_value)
        scores[field] = FieldScore(
            outcome=outcome,
            extracted=value,
            truth=truth_value,
            rel_error=rel_error,
        )

    for field in STRING_FIELDS:
        truth_value = truth.get(field)
        value = None if extracted is None else getattr(extracted, field, None)
        scores[field] = FieldScore(
            outcome=_string_outcome(value, truth_value),
            extracted=value,
            truth=truth_value,
        )

    return scores


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


def load_ground_truth() -> dict:
    """
    Load the XBRL answer key.

    Returns:
        The full ground-truth mapping keyed '{cik}_{year}'.

    Raises:
        FileNotFoundError: if the answer key has not been generated.
    """
    path = Path(settings.EVAL_DATA_PATH) / "ground_truth.json"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_field(scores: list[FieldScore]) -> dict:
    """
    Aggregate one field's scores across documents into precision/recall/F1.

    Denominators are reported explicitly: a precision computed over four
    documents is not the same evidence as one computed over ten, and a bare
    ratio hides that.

    Args:
        scores: this field's FieldScore across all evaluated documents.

    Returns:
        Metrics plus counts, denominators, and mean relative error.
    """
    correct = sum(1 for s in scores if s.outcome == CORRECT)
    wrong = sum(1 for s in scores if s.outcome == WRONG)
    missing = sum(1 for s in scores if s.outcome == MISSING)
    n_a = sum(1 for s in scores if s.outcome == N_A)

    asserted = correct + wrong
    findable = correct + wrong + missing

    precision = correct / asserted if asserted else None
    recall = correct / findable if findable else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else None
    )

    errors = [s.rel_error for s in scores if s.rel_error is not None]
    mean_rel_error = sum(errors) / len(errors) if errors else None

    return {
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "n_a": n_a,
        "n_scored": findable,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_rel_error": mean_rel_error,
    }


def evaluate_strategy(
    strategy_id: str, results: list[ExtractionResult]
) -> dict:
    """
    Aggregate one strategy's results across the evaluation set.

    Takes already-run ExtractionResults rather than calling the model itself, so
    scoring is decoupled from extraction: the scorer can be re-run repeatedly
    against cached outputs at no API cost.

    Args:
        strategy_id: the strategy being scored.
        results: one ExtractionResult per evaluated document.

    Returns:
        A dict of per-field metrics, per-document scores, token cost, the
        provenance split, and (for evidence strategies) the grounding rate.
    """
    truth_all = load_ground_truth()

    per_document: dict[str, dict] = {}
    by_field: dict[str, list[FieldScore]] = {f: [] for f in SCORED_FIELDS}
    by_provenance: dict[str, dict[str, list[FieldScore]]] = {}
    grounding_checks: list[bool] = []

    input_tokens = output_tokens = 0
    validation_failures = 0

    for result in results:
        key = f"{result.cik}_{result.year}"
        truth = truth_all.get(key, {})
        if not truth:
            logger.warning("ground_truth_missing", key=key, strategy=strategy_id)

        scores = score_extraction(result.extraction, truth)
        per_document[key] = {
            "section_source": result.section_source,
            "scores": {f: s.model_dump() for f, s in scores.items()},
        }

        if result.extraction is None:
            validation_failures += 1

        for field, score in scores.items():
            by_field[field].append(score)
            bucket = by_provenance.setdefault(
                result.section_source, {f: [] for f in SCORED_FIELDS}
            )
            bucket[field].append(score)

        if result.evidence:
            try:
                sections = load_sections(result.cik, result.year)
                source_text, _ = assemble_input(sections)
                grounded = check_grounding(result.evidence, source_text)
                per_document[key]["grounding"] = grounded
                grounding_checks.extend(
                    v for v in grounded.values() if v is not None
                )
            except (FileNotFoundError, KeyError) as exc:
                logger.error(
                    "grounding_check_failed",
                    key=key,
                    strategy=strategy_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        input_tokens += result.input_tokens
        output_tokens += result.output_tokens

    fields = {f: _aggregate_field(s) for f, s in by_field.items()}
    provenance = {
        source: {f: _aggregate_field(s) for f, s in buckets.items()}
        for source, buckets in by_provenance.items()
    }

    grounding_rate = (
        sum(grounding_checks) / len(grounding_checks)
        if grounding_checks
        else None
    )

    logger.info(
        "strategy_evaluated",
        strategy=strategy_id,
        documents=len(results),
        validation_failures=validation_failures,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        grounding_rate=grounding_rate,
    )

    return {
        "strategy_id": strategy_id,
        "n_documents": len(results),
        "validation_failures": validation_failures,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "grounding_rate": grounding_rate,
        "fields": fields,
        "by_provenance": provenance,
        "per_document": per_document,
    }