"""
src/ingestion/sections.py
Deterministic section slicing for extraction input.

Produces the text blocks fed to the extraction prompts BEFORE any LLM call, so
every strategy in the bake-off receives byte-identical input (the controlled
input the experiment depends on). Two sections:

  - cover                : head of the on-disk filing text; source for the
                           cover-page scored fields (company_name,
                           fiscal_year_end).
  - financial_statements : Item 8 via edgartools, guarded on two sides — a char
                           floor (edgartools silently truncates Item 8 to a
                           heading on ~25% of filings) and the model input
                           budget. Either failure falls back to the full filing
                           text, tagged distinctly.

Every section carries provenance (`section_source`) so scoring can later
attribute a result to input quality vs prompt quality.

edgartools' section detection is used for Item 8 (mature library for solved
plumbing) but never trusted blindly: the gate + fallback is what makes it safe.
Phase 3 narrative extraction extends this with Item 1 / Item 1A.

Dependencies: tiktoken, pydantic v2, src.utils.config, src.utils.logger.
"""

from typing import Protocol

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# gpt-4o-mini family tokeniser; used only to report token budgets per section.
_ENCODING = tiktoken.get_encoding("o200k_base")

_ITEM8_KEY = "Item 8"


class FilingLike(Protocol):
    """
    Structural type for the one capability get_sections needs from an
    edgartools filing: obj() -> a form object supporting item lookup by key.

    Declared as a Protocol (not the concrete edgar.Filing) so the slicer is
    decoupled from edgartools' class and unit-testable with a fake.
    """

    def obj(self) -> object:  # noqa: D102 - protocol stub
        ...


class Section(BaseModel):
    """
    One sliced section of a filing, with provenance.

    Whitespace is deliberately NOT stripped: flattened financial tables rely on
    runs of whitespace to keep columns aligned, and stripping would corrupt the
    numeric layout the extractor reads.

    Attributes:
        text: the section text, verbatim.
        token_count: token count under the gpt-4o-mini tokeniser (budget info).
        section_source: how this text was obtained — 'cover_head',
            'edgartools_item8', 'fallback_fulltext', or
            'fallback_fulltext_oversize'.
    """

    model_config = ConfigDict(str_strip_whitespace=False)

    text: str = Field(description="Section text, verbatim (whitespace preserved).")
    token_count: int = Field(description="Token count under the o200k_base tokeniser.")
    section_source: str = Field(description="Provenance tag for this section.")


def _n_tokens(text: str) -> int:
    """Return the token count of text under the gpt-4o-mini tokeniser."""
    return len(_ENCODING.encode(text))


def _try_item8(filing: FilingLike) -> str | None:
    """
    Attempt to extract Item 8 text via edgartools.

    Returns the item text, or None if edgartools returns nothing or raises
    (both are treated the same downstream: gate fails, fallback taken).

    Args:
        filing: the resolved filing object.

    Returns:
        Item 8 text, or None on absence/error.
    """
    try:
        tenk = filing.obj()
        text = tenk[_ITEM8_KEY]  # edgartools __getitem__: str | None
        return text
    except Exception as exc:  # never crash the slice: fall back instead
        logger.warning(
            "item8_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None


def _passes_floor(text: str | None) -> bool:
    """
    True if the Item 8 slice is present and at least the char floor long.

    The floor (settings.ITEM8_MIN_CHARS) is edgartools' own empirical minimum
    for a 10-K Item 8. A slice below it is a truncated heading, not the
    statements — the gate rejects it so the caller can fall back. Confirmed true
    positives in this corpus: JPM (92 tokens), QCOM (83), XOM (153), NFLX (54).

    Args:
        text: candidate Item 8 text, or None.

    Returns:
        Whether the slice is long enough to be the statements.
    """
    return bool(text) and len(text) >= settings.ITEM8_MIN_CHARS


def _build_cover(raw_text: str) -> Section:
    """
    Build the cover section from the head of the on-disk filing text.

    The cover page (registrant name, 'for the fiscal year ended <date>') sits
    at the very top of the filing, so a fixed head slice reliably contains the
    two cover-page scored fields.

    Args:
        raw_text: the on-disk filing text.

    Returns:
        The cover Section.
    """
    head = raw_text[: settings.COVER_MAX_CHARS]
    return Section(
        text=head,
        token_count=_n_tokens(head),
        section_source="cover_head",
    )

# Modified - the ceiling becomes a budget check that includes cover, 
# the tag is renamed, and the margin is logged so GS's 781 tokens is visible in the run output. 
def _build_financial_statements(
    filing: FilingLike, raw_text: str, cover_tokens: int
) -> Section:
    """
    Build the financial-statements section: gated Item 8, else full-text fallback.

    Item 8 is accepted when it clears the char floor AND the assembled input
    (Item 8 + cover) fits the model input budget. The budget test replaces an
    earlier absolute token ceiling: that ceiling fired on three documents in this
    corpus (PFE, GS, BAC) and boundary inspection showed all three slices were
    genuine Item 8 — correct start at the auditor's report / statement index,
    tail inside the notes. The observation that motivated it (GS's Item 8
    appearing 'larger than its whole filing') was a rendering artifact: the item
    view line-breaks every table cell, so the same content tokenises ~58% higher
    than filing.text(). In characters the slice is a proper subset, as it must be.

    What is left is the constraint that actually matters — whether the assembled
    input fits the window. Fallback tags:

      - 'fallback_fulltext'          : Item 8 truncated/missing (below the floor)
      - 'fallback_fulltext_oversize' : Item 8 + cover exceeds the input budget

    so the two failure modes stay separable at scoring time. If the fallback text
    ALSO exceeds the budget, it is returned but logged at ERROR — the caller must
    resolve it rather than silently sending an over-window prompt.

    Args:
        filing: the resolved filing object (for Item 8).
        raw_text: the on-disk filing text (single source for the fallback).
        cover_tokens: token count of the cover slice, which shares the budget.

    Returns:
        The financial-statements Section.
    """
    item8 = _try_item8(filing)

    if not _passes_floor(item8) or item8 is None:
        logger.warning(
            "item8_below_floor_fallback",
            item8_chars=len(item8) if item8 else 0,
            floor_chars=settings.ITEM8_MIN_CHARS,
        )
        fallback_source = "fallback_fulltext"
    else:
        item8_tokens = _n_tokens(item8)
        assembled_tokens = item8_tokens + cover_tokens
        margin = settings.MODEL_INPUT_TOKEN_BUDGET - assembled_tokens

        if margin >= 0:
            logger.info(
                "item8_accepted",
                item8_tokens=item8_tokens,
                cover_tokens=cover_tokens,
                assembled_tokens=assembled_tokens,
                budget=settings.MODEL_INPUT_TOKEN_BUDGET,
                margin_tokens=margin,
            )
            return Section(
                text=item8,
                token_count=item8_tokens,
                section_source="edgartools_item8",
            )

        logger.warning(
            "item8_over_budget_fallback",
            item8_tokens=item8_tokens,
            cover_tokens=cover_tokens,
            assembled_tokens=assembled_tokens,
            budget=settings.MODEL_INPUT_TOKEN_BUDGET,
            overflow_tokens=-margin,
        )
        fallback_source = "fallback_fulltext_oversize"

    fallback_tokens = _n_tokens(raw_text)
    fallback_assembled = fallback_tokens + cover_tokens
    if fallback_assembled > settings.MODEL_INPUT_TOKEN_BUDGET:
        logger.error(
            "fallback_exceeds_model_budget",
            fallback_tokens=fallback_tokens,
            cover_tokens=cover_tokens,
            assembled_tokens=fallback_assembled,
            budget=settings.MODEL_INPUT_TOKEN_BUDGET,
            fallback_source=fallback_source,
        )
    return Section(
        text=raw_text,
        token_count=fallback_tokens,
        section_source=fallback_source,
    )


def get_sections(filing: FilingLike, raw_text: str) -> dict[str, Section]:
    """
    Slice a filing into the sections the extraction prompts consume.

    Pure logic: takes the resolved filing (for Item 8) and the on-disk filing
    text (for the cover slice and the fallback), and returns the section dict.
    All disk I/O lives in the calling script, so this is unit-testable with a
    fake filing and an in-memory string.

    Cover is built first because its token count is part of the budget the Item 8
    gate tests against — the constraint is on the assembled input, not on Item 8
    in isolation.

    Args:
        filing: the resolved edgartools filing object.
        raw_text: the on-disk filing text (data/raw/{cik}_{year}.txt contents).

    Returns:
        Mapping with keys 'cover' and 'financial_statements'.
    """
    cover = _build_cover(raw_text)
    return {
        "cover": cover,
        "financial_statements": _build_financial_statements(
            filing, raw_text, cover.token_count
        ),
    }