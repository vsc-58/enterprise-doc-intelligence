# src/storage/quality.py
# Module: Extraction quality flags (scale coherence + evidence agreement)
# Purpose: Decides how much assurance a stored figure carries, from the evidence
#          columns alone, so the query path and /export can both disclose it.
# Depends on: pydantic v2, src.storage.metadata_store, src.utils.config,
#             src.utils.logger, src.utils.numbers
#
# BOUNDARY RULE: this module never reads data/eval/ground_truth.json. Every input
# is something the extraction pipeline itself produced. If the check ever needs
# the answer key to work, it is an evaluation metric, not a runtime feature, and
# it does not ship (D29).

from collections import Counter
from collections.abc import Mapping
from enum import Enum

from pydantic import BaseModel, ConfigDict

from src.storage.metadata_store import EVIDENCE_FIELDS, ExtractedRecord
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.numbers import SCALE_MULTIPLIERS, numbers_in_line

logger = get_logger(__name__)


class QualityVerdict(str, Enum):
    """
    How much assurance a stored figure carries.

    VERIFIED: the cited line contains the value, and the scale implied by that
        citation agrees with the rest of the document.
    UNVERIFIED: the check could not be completed — the field is null, or the
        document has too few checkable fields to establish a scale.
    DISPUTED: a check completed and failed. The figure is served anyway, with
        the reason disclosed; the structured store remains the source of record
        for figures, so disclosure rather than suppression is the response.

    str-valued so the member persists and serialises as readable text.
    """

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"


class QualityReason(str, Enum):
    """
    Why a verdict was reached.

    CONSISTENT: both checks passed.
    NULL_FIELD: no value was extracted, so there is nothing to check.
    EVIDENCE_MISMATCH: the cited line does not contain the stored value at any
        scale. Observed where the target line does not exist in the filing and
        the model derived a figure while citing a neighbouring row (Phase 1 D5),
        and where it returned one accounting concept while quoting another.
    SCALE_DISAGREEMENT: the citation matches, but at a different scale from the
        document's other fields. The minority field is the suspect.
    SCALE_AMBIGUOUS: the document's fields split evenly between two scales, so
        there is no majority and no basis for calling either side wrong. Every
        field in the tied groups is flagged, deliberately including ones that
        may be correct — the system reports the contradiction it can see rather
        than guessing which half is authoritative.
    SCALE_NOT_CHECKABLE: fewer than SCALE_COHERENCE_MIN_FIELDS citable fields.
    """

    CONSISTENT = "consistent"
    NULL_FIELD = "null_field"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    SCALE_DISAGREEMENT = "scale_disagreement"
    SCALE_AMBIGUOUS = "scale_ambiguous"
    SCALE_NOT_CHECKABLE = "scale_not_checkable"


class QualityFlag(BaseModel):
    """
    The assurance attached to one stored figure.

    Attributes:
        field: The extraction field this describes.
        verdict: Assurance level.
        reason: Why that verdict was reached.
        implied_multiplier: Scale the citation implies (stored value divided by
            the printed figure), or None when no printed figure matched.
        document_scale: The document's majority scale, or None when absent.
        suspected_true_value: Where a minority field disagrees with a known
            document scale, the value it would hold at that scale. None in every
            other case — including ambiguity, where no correction is inferable.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    verdict: QualityVerdict
    reason: QualityReason
    implied_multiplier: float | None = None
    document_scale: float | None = None
    suspected_true_value: float | None = None

    def note(self) -> str | None:
        """
        Render the one-sentence disclosure shown beside the figure.

        Wording lives here so the API response, the answer text and the export
        never drift apart.

        Returns:
            The sentence, or None when the figure is verified and needs none.
        """
        if self.verdict is QualityVerdict.VERIFIED:
            return None
        if self.reason is QualityReason.NULL_FIELD:
            return "This figure was not extracted from the filing."
        if self.reason is QualityReason.EVIDENCE_MISMATCH:
            return (
                "Flagged: the line cited for this figure does not contain it, "
                "so the value may be derived or drawn from a neighbouring item."
            )
        if self.reason is QualityReason.SCALE_DISAGREEMENT:
            suspected = (
                f" The value is likely {self.suspected_true_value:,.0f}."
                if self.suspected_true_value is not None
                else ""
            )
            return (
                "Flagged: this figure's implied scale disagrees with the rest of "
                f"the filing.{suspected}"
            )
        if self.reason is QualityReason.SCALE_AMBIGUOUS:
            return (
                "Flagged: figures in this filing were extracted at two different "
                "scales and neither is in the majority, so which is authoritative "
                "cannot be determined from the filing alone."
            )
        return "Flagged: the scale of this figure could not be checked."


def implied_multiplier(value: float | None, source_line: str | None) -> float | None:
    """
    Infer the scale a citation implies for a stored value.

    Tries each multiplier in SCALE_MULTIPLIERS against every number printed in
    the cited line, under the same relative tolerance the grounding check uses,
    so a field consistent there always yields a multiplier here.

    Args:
        value: The stored figure, in actual dollars.
        source_line: The line the model cited for it.

    Returns:
        The multiplier, or None when the value is null, the line is missing, or
        no printed figure matches at any scale.
    """
    if value is None or not source_line:
        return None

    target = abs(float(value))
    tolerance = max(target * settings.EVIDENCE_MATCH_REL_TOLERANCE, 1e-6)
    for number in numbers_in_line(str(source_line)):
        for scale in SCALE_MULTIPLIERS:
            if abs(number * scale - target) <= tolerance:
                return scale
    return None


def document_scale(multipliers: Mapping[str, float | None]) -> tuple[float | None, bool]:
    """
    Establish the document's majority scale from its per-field multipliers.

    Args:
        multipliers: Field to implied multiplier, None where not inferable.

    Returns:
        (scale, ambiguous). scale is the strict majority multiplier, or None when
        too few fields are checkable or the top two counts are tied. ambiguous is
        True only in the tie case, which is what distinguishes "we cannot check"
        from "the document contradicts itself".
    """
    counts = Counter(m for m in multipliers.values() if m is not None)
    if sum(counts.values()) < settings.SCALE_COHERENCE_MIN_FIELDS:
        return None, False

    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, True
    return ranked[0][0], False


def assess(
    values: Mapping[str, float | None],
    source_lines: Mapping[str, str | None],
) -> dict[str, QualityFlag]:
    """
    Assess every numeric field of one extraction.

    Pure: no I/O, no database, no LLM, no answer key. Takes the two pieces of
    stored evidence and returns one flag per field.

    Args:
        values: Field to stored figure.
        source_lines: Field to the line cited for it.

    Returns:
        Field to QualityFlag, covering every name in EVIDENCE_FIELDS.
    """
    multipliers = {
        field: implied_multiplier(values.get(field), source_lines.get(field))
        for field in EVIDENCE_FIELDS
    }
    scale, ambiguous = document_scale(multipliers)

    flags: dict[str, QualityFlag] = {}
    for field in EVIDENCE_FIELDS:
        value, multiplier = values.get(field), multipliers[field]

        if value is None:
            verdict, reason = QualityVerdict.UNVERIFIED, QualityReason.NULL_FIELD
        elif multiplier is None:
            verdict, reason = QualityVerdict.DISPUTED, QualityReason.EVIDENCE_MISMATCH
        elif ambiguous:
            verdict, reason = QualityVerdict.DISPUTED, QualityReason.SCALE_AMBIGUOUS
        elif scale is None:
            verdict, reason = QualityVerdict.UNVERIFIED, QualityReason.SCALE_NOT_CHECKABLE
        elif multiplier != scale:
            verdict, reason = QualityVerdict.DISPUTED, QualityReason.SCALE_DISAGREEMENT
        else:
            verdict, reason = QualityVerdict.VERIFIED, QualityReason.CONSISTENT

        suspected = (
            value * (scale / multiplier)
            if reason is QualityReason.SCALE_DISAGREEMENT
            and value is not None
            and multiplier
            and scale
            else None
        )

        flags[field] = QualityFlag(
            field=field,
            verdict=verdict,
            reason=reason,
            implied_multiplier=multiplier,
            document_scale=scale,
            suspected_true_value=suspected,
        )

    return flags


def assess_record(record: ExtractedRecord) -> dict[str, QualityFlag]:
    """
    Assess one stored extraction row.

    Thin adapter over assess(), reading the flattened evidence columns rather
    than evidence_json — the same projection /export reads, so a figure and its
    flag always come from one source.

    Args:
        record: A row from extracted_records.

    Returns:
        Field to QualityFlag, covering every name in EVIDENCE_FIELDS.
    """
    values = {field: getattr(record, field) for field in EVIDENCE_FIELDS}
    source_lines = {
        field: getattr(record, f"{field}_source_line") for field in EVIDENCE_FIELDS
    }
    return assess(values, source_lines)