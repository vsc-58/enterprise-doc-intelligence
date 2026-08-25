"""
src/extraction/schemas.py
Pydantic v2 extraction schemas for the prompt-strategy bake-off.

Two schemas:
  - FilingExtraction            : the target shape (S1, S2, S4)
  - FilingExtractionWithEvidence: S3's grounded variant; each numeric field
                                  carries the source line it was read from,
                                  and projects down to FilingExtraction via
                                  .to_extraction() so all four arms are scored
                                  on identical `value` fields.

Dependencies: pydantic v2.
"""

from pydantic import BaseModel, ConfigDict, Field


class FilingExtraction(BaseModel):
    """
    Structured fields extracted from a single 10-K filing.

    Financial fields are `float | None`: extraction may legitimately fail per
    field (a value genuinely absent from the section, e.g. no standalone
    total-liabilities line), and a null is recorded rather than a guessed
    number. Monetary values are in ACTUAL DOLLARS (full units), not the
    scaled "in millions" figure shown in the source table — this is enforced
    by the prompt and is what lets the evaluator compare against raw-dollar
    XBRL ground truth.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    company_name: str = Field(
        description="Registrant's legal name as it appears in the filing."
    )
    fiscal_year_end: str | None = Field(
        default=None,
        description="Fiscal year-end date in ISO format YYYY-MM-DD "
        "(e.g. 2022-09-24). Null if not determinable from the text.",
    )

    total_revenue: float | None = Field(
        default=None,
        description="Total revenue / net revenues / net sales for the most "
        "recent fiscal year, in ACTUAL DOLLARS. Null if absent.",
    )
    net_income: float | None = Field(
        default=None,
        description="Net income for the most recent fiscal year, in ACTUAL "
        "DOLLARS. Null if absent.",
    )
    total_assets: float | None = Field(
        default=None,
        description="Total assets at fiscal year-end, in ACTUAL DOLLARS. "
        "Null if absent.",
    )
    total_liabilities: float | None = Field(
        default=None,
        description="Total liabilities at fiscal year-end, in ACTUAL DOLLARS. "
        "Null if absent (some filers report no standalone line).",
    )
    operating_cash_flow: float | None = Field(
        default=None,
        description="Net cash provided by operating activities for the most "
        "recent fiscal year, in ACTUAL DOLLARS. May be negative. Null if absent.",
    )

    business_description: str | None = Field(
        default=None,
        description="Short summary of the business. Null if the relevant "
        "section was not provided.",
    )
    primary_risk_factors: list[str] = Field(
        default_factory=list,
        description="Up to three top risk-factor headings. Empty list if the "
        "relevant section was not provided.",
    )
    auditor_name: str | None = Field(
        default=None,
        description="Independent registered public accounting firm named in "
        "the auditor's report. Null if absent.",
    )


class NumericEvidence(BaseModel):
    """
    A single grounded numeric extraction: the normalised value plus the exact
    source line it was read from.

    `source_line` is copied verbatim from the provided text so it can be
    validated as a substring of the source at write time (grounding check):
    if the quoted line is not actually in the document, the value was
    fabricated and the row is flagged rather than trusted.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    value: float | None = Field(
        default=None,
        description="The figure in ACTUAL DOLLARS (full units), most recent "
        "fiscal year, first data column. Null if absent.",
    )
    source_line: str | None = Field(
        default=None,
        description="The exact line from the provided text this value was read "
        "from, copied verbatim (e.g. 'Net revenues 394,328 365,817'). Null if "
        "the value is null.",
    )


class FilingExtractionWithEvidence(BaseModel):
    """
    S3's grounded schema. Numeric fields carry evidence; string/narrative
    fields match FilingExtraction. `to_extraction()` strips evidence to a plain
    FilingExtraction so S3 is scored on the same `value` fields as the other
    arms — the evidence is a by-product used for grounding validation, not part
    of the accuracy comparison.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    company_name: str = Field(
        description="Registrant's legal name as it appears in the filing."
    )
    fiscal_year_end: str | None = Field(
            default=None,
            description="Fiscal year-end date in ISO format YYYY-MM-DD "
            "(e.g. 2022-09-24). Null if not determinable from the text.",
        )
    total_revenue: NumericEvidence = Field(default_factory=NumericEvidence)
    net_income: NumericEvidence = Field(default_factory=NumericEvidence)
    total_assets: NumericEvidence = Field(default_factory=NumericEvidence)
    total_liabilities: NumericEvidence = Field(default_factory=NumericEvidence)
    operating_cash_flow: NumericEvidence = Field(default_factory=NumericEvidence)

    business_description: str | None = Field(default=None)
    primary_risk_factors: list[str] = Field(default_factory=list)
    auditor_name: str | None = Field(default=None)

    def to_extraction(self) -> FilingExtraction:
        """
        Project down to a plain FilingExtraction, discarding evidence.

        Used before scoring so the evidence arm is compared to the other arms
        on identical value fields.

        Returns:
            A FilingExtraction with each numeric field set to its evidence
            value and all non-numeric fields copied across.
        """
        return FilingExtraction(
            company_name=self.company_name,
            fiscal_year_end=self.fiscal_year_end,
            total_revenue=self.total_revenue.value,
            net_income=self.net_income.value,
            total_assets=self.total_assets.value,
            total_liabilities=self.total_liabilities.value,
            operating_cash_flow=self.operating_cash_flow.value,
            business_description=self.business_description,
            primary_risk_factors=self.primary_risk_factors,
            auditor_name=self.auditor_name,
        )

    def evidence_map(self) -> dict[str, NumericEvidence]:
        """
        Return the five numeric fields keyed by name, for grounding validation
        in Phase 3 (check each source_line is a substring of the source text).

        Returns:
            Mapping of field name to its NumericEvidence.
        """
        return {
            "total_revenue": self.total_revenue,
            "net_income": self.net_income,
            "total_assets": self.total_assets,
            "total_liabilities": self.total_liabilities,
            "operating_cash_flow": self.operating_cash_flow,
        }