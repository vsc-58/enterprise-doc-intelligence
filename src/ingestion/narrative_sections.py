"""
src/ingestion/narrative_sections.py
Section slicing for the RAG corpus (Phase 4).

Separate from sections.py by design. That module slices the EXTRACTION input
(cover + Item 8) and its output is hashed into the Phase 2/3 cache key — any
change there re-bills the eval-10. This module slices the RETRIEVAL corpus and
shares nothing with it but the FilingLike Protocol.

Items sliced, in acceptance priority:

  Item 1A  Risk Factors      20/20 present, 0 collapsed. Highest-value section
                             for the RAG path and the architecture's headline
                             demo query ("which companies flagged X risk").
  Item 7   MD&A              Management's explanation of the figures.
  Item 1   Business          What the company does.
  Item 7A  Market Risk       Rate/currency/commodity exposure.

Item 8 is deliberately EXCLUDED. Two reasons: its boundary is the least reliable
in the corpus (4/20 collapse to a heading, 2/20 over-capture), and embedding the
financial statements would let a question mis-routed to NARRATIVE retrieve a
balance sheet and have the LLM SYNTHESISE a figure — outside every guarantee the
parameterized SQL path exists to provide. The mis-route in the other direction
degrades safely (sql_path returns a bounded-capability refusal); this one does
not. Design boundary: the vector store holds narrative sections only; financial
figures come from the structured store or not at all.

Two guards, both measured rather than assumed (scripts/probe_sections.py):

  - Token floor. Collapsed items are 52-157 tokens (JPM Item 7 = 73, XOM Item 7
    = 52). Real items are 588+. The floor sits between, with margin.
  - Overlap rejection. edgartools item boundaries over-capture, so one slice can
    contain another (QCOM: Item 1A is 181,796 of Item 1's 181,831 chars; META:
    Item 1 sits inside Item 1A). Chunking both embeds the same passage twice
    under two section labels, producing duplicate top-k hits and a Sources block
    that cites one passage as two sections. First accepted slice wins.

There is NO full-text fallback. Item 1A is present on every document, so no
document can vanish from the corpus. A per-item gap (INTC/JPM/XOM have no usable
Item 7) is accepted and logged: the RAG chain then refuses MD&A questions about
those companies rather than answering from the wrong section.

Dependencies: tiktoken, pydantic v2, src.ingestion.sections (Protocol only),
src.utils.config, src.utils.logger.
"""

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.sections import FilingLike
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Same tokeniser as sections.py (gpt-4o-mini family). The probe that set the
# floor used cl100k_base; the two differ by a few percent on prose, far inside
# the margin between a collapsed item (52-157) and a real one (588+).
_ENCODING = tiktoken.get_encoding("o200k_base")

# Acceptance priority. Item 1A leads because it is the only item present and
# uncollapsed on all 20 documents, so where two slices overlap the one that
# survives is the one whose label is most reliable.
NARRATIVE_ITEMS: tuple[str, ...] = ("Item 1A", "Item 7", "Item 1", "Item 7A")


class NarrativeSection(BaseModel):
    """
    One narrative section of a filing, ready to chunk.

    Whitespace is NOT stripped, matching sections.py: the retrieval corpus and
    the extraction corpus should render the same source text the same way, and
    RecursiveCharacterTextSplitter splits on paragraph breaks first — collapsing
    them would remove the separator the splitter most wants to use.

    Attributes:
        item_label: edgartools item key, e.g. "Item 1A". Becomes the chunk's
            source_section metadata verbatim. Exact by construction: the text
            IS what that item returned, so no containment check is needed.
        text: section text, verbatim.
        token_count: token count under the o200k_base tokeniser.
        char_count: character count (the unit overlap is measured in).
    """

    model_config = ConfigDict(str_strip_whitespace=False)

    item_label: str = Field(description="edgartools item key for this slice.")
    text: str = Field(description="Section text, verbatim (whitespace preserved).")
    token_count: int = Field(description="Token count under the o200k_base tokeniser.")
    char_count: int = Field(description="Character count of the raw slice.")


def _n_tokens(text: str) -> int:
    """Return the token count of text under the gpt-4o-mini tokeniser.

    Args:
        text: Text to measure.

    Returns:
        Token count; 0 for empty text.
    """
    if not text:
        return 0
    return len(_ENCODING.encode(text))


def _normalize(text: str) -> str:
    """Remove all whitespace, for containment comparison only.

    edgartools' item view line-breaks table cells differently from other text
    paths, so a raw substring test reports false negatives on identical content
    (the same artifact that made Phase 3 D14's token counts diverge ~58%). Never
    used for stored text — only inside the overlap test.

    Args:
        text: Text to normalise.

    Returns:
        The text with all whitespace removed.
    """
    return "".join(text.split())


def _try_item(filing_obj: object, item_label: str) -> str | None:
    """Attempt to read one item from a resolved form object.

    Args:
        filing_obj: edgartools form object supporting item lookup by key.
        item_label: Item key, e.g. "Item 1A".

    Returns:
        Item text, or None if absent, empty, or the lookup raised. Absence and
        error are equivalent downstream — both mean "no usable slice" — so both
        return None, with the error logged for diagnosis.
    """
    try:
        text = filing_obj[item_label]  # type: ignore[index]
    except Exception as exc:
        logger.warning(
            "narrative_item_lookup_failed",
            item=item_label,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None

    if not text or not str(text).strip():
        return None
    return str(text)


def _overlaps(candidate: str, accepted: str) -> bool:
    """Test whether two slices share substantial text, in either direction.

    Bidirectional because over-capture runs both ways in this corpus: META's
    Item 1 sits inside Item 1A (candidate contained in accepted), while QCOM's
    Item 1 CONTAINS Item 1A (accepted contained in candidate). A one-directional
    test would catch META and miss QCOM.

    Samples from the middle of each slice rather than the edges: two adjacent
    items legitimately share a boundary sentence, so an edge match proves
    nothing. A mid-slice match means real containment.

    Args:
        candidate: Slice being considered.
        accepted: Slice already accepted for this document.

    Returns:
        Whether either slice's midpoint sample appears inside the other.
    """
    probe = settings.SECTION_OVERLAP_PROBE_CHARS
    left, right = _normalize(candidate), _normalize(accepted)

    for inner, outer in ((left, right), (right, left)):
        if len(inner) < probe * 2 or len(outer) <= len(inner):
            continue
        mid = len(inner) // 2
        if inner[mid : mid + probe] in outer:
            return True
    return False


def get_narrative_sections(
    filing: FilingLike, company: str, year: int
) -> list[NarrativeSection]:
    """Slice a filing into the narrative sections that form the RAG corpus.

    Walks NARRATIVE_ITEMS in priority order. An item is accepted when it clears
    the token floor AND does not overlap a slice already accepted for this
    document. Rejections are logged with the reason so a run's output shows
    exactly which sections each company contributed.

    Pure logic apart from filing.obj(): takes the resolved filing and returns
    sections. Disk I/O and persistence live in the calling script.

    Args:
        filing: Resolved edgartools filing object.
        company: Company name, for log context only.
        year: Filing year, for log context only.

    Returns:
        Accepted sections in priority order. Never raises. An empty list means
        no item cleared the guards — this should not occur (Item 1A is present
        on all 20 documents) and is logged at ERROR for the caller to handle.
    """
    try:
        filing_obj = filing.obj()
    except Exception as exc:
        logger.error(
            "filing_object_failed",
            company=company,
            year=year,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return []

    accepted: list[NarrativeSection] = []

    for item_label in NARRATIVE_ITEMS:
        text = _try_item(filing_obj, item_label)
        if text is None:
            logger.info(
                "narrative_item_absent", company=company, year=year, item=item_label
            )
            continue

        tokens = _n_tokens(text)
        if tokens < settings.NARRATIVE_MIN_TOKENS:
            logger.warning(
                "narrative_item_below_floor",
                company=company,
                year=year,
                item=item_label,
                tokens=tokens,
                floor_tokens=settings.NARRATIVE_MIN_TOKENS,
            )
            continue

        clash = next(
            (s for s in accepted if _overlaps(text, s.text)),
            None,
        )
        if clash is not None:
            logger.warning(
                "narrative_item_overlaps_accepted",
                company=company,
                year=year,
                item=item_label,
                overlaps_with=clash.item_label,
                candidate_chars=len(text),
                accepted_chars=clash.char_count,
            )
            continue

        accepted.append(
            NarrativeSection(
                item_label=item_label,
                text=text,
                token_count=tokens,
                char_count=len(text),
            )
        )
        logger.info(
            "narrative_item_accepted",
            company=company,
            year=year,
            item=item_label,
            tokens=tokens,
        )

    if not accepted:
        logger.error(
            "no_narrative_sections", company=company, year=year,
        )
        return []

    logger.info(
        "narrative_sections_built",
        company=company,
        year=year,
        items=[s.item_label for s in accepted],
        total_tokens=sum(s.token_count for s in accepted),
    )
    return accepted