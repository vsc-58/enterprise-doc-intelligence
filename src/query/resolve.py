# src/query/resolve.py
# Module: Entity and period resolution
# Purpose: Turn the raw company mention a planner emitted into a corpus identity,
#          and settle which fiscal year a question refers to.
# Depends on: src.query.schemas, src.storage.metadata_store, src.utils.logger
#
# No LLM and no network. The planner is corpus-blind (D32), so this module is the
# single place that knows which companies and years are held — one definition to
# update when the corpus grows, rather than a prompt and a table drifting apart.

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from src.utils.text import display_name

from src.query.schemas import (
    ResolutionFailure,
    Resolution,
    ResolvedEntity,
)
from src.storage.metadata_store import (
    Document,
    get_session,
    latest_successful_stmt,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tokens that carry no identifying information. Applied to BOTH sides, so
# "Bank of America" still resolves while a bare "the bank" reduces to nothing
# and is reported as too vague rather than matched to whichever bank sorts first.
GENERIC_TOKENS: frozenset[str] = frozenset(
    {
        "the", "inc", "incorporated", "corp", "corporation", "co", "company",
        "companies", "plc", "ltd", "limited", "llc", "lp", "holdings", "holding",
        "group", "com", "sa", "nv", "ag", "class", "and", "of", "bank", "firm",
        "platforms", "technologies", "international",
    }
)

# Short names people use that no token match would reach, because the registrant
# name does not contain them. Deliberately small: anything resolvable from the
# EDGAR name is left to token matching rather than hardcoded here.
ALIASES: dict[str, str] = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "bofa": "BAC",
    "bank of america": "BAC",
    "jpm": "JPM",
    "j p morgan": "JPM",
    "goldman": "GS",
    "jnj": "JNJ",
    "j j": "JNJ",
    "exxon": "XOM",
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class CorpusEntry(BaseModel):
    """
    One held filing, flattened for matching.

    Constructed from the database by load_index(), or by hand in tests — which is
    why resolution is testable with no database, no API key and no network.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    cik: str
    company_name: str
    document_id: int
    filing_year: int
    fiscal_year_end: str | None = None
    is_extracted: bool = True
    is_embedded: bool = True


class CorpusIndex(BaseModel):
    """Corpus entries with their precomputed identifying tokens."""

    model_config = ConfigDict(frozen=True)

    entries: list[CorpusEntry]
    tokens_by_ticker: dict[str, frozenset[str]]


_cached_index: CorpusIndex | None = None


def normalise(text: str) -> str:
    """
    Lowercase a name and strip punctuation, dropping any state suffix.

    EDGAR registrant names carry forms like "BANK OF AMERICA CORP /DE/" and
    "QUALCOMM INC/DE"; everything from the first slash is discarded before
    normalising, since the incorporation state is never part of how anyone
    refers to a company.

    Args:
        text: A company name or mention.

    Returns:
        Lowercased, space-separated text with punctuation removed.
    """
    head = text.split("/")[0]
    return _NON_ALNUM.sub(" ", head.lower()).strip()


def identifying_tokens(text: str) -> frozenset[str]:
    """
    Reduce a name to the tokens that actually identify a company.

    Args:
        text: A company name or mention.

    Returns:
        The token set with generic words removed. Empty when the text carried
        nothing identifying — "the company", "the bank".
    """
    return frozenset(normalise(text).split()) - GENERIC_TOKENS


def build_index(entries: Sequence[CorpusEntry]) -> CorpusIndex:
    """
    Precompute the token sets used for matching.

    Args:
        entries: The held filings.

    Returns:
        An index ready for resolve_company().
    """
    return CorpusIndex(
        entries=list(entries),
        tokens_by_ticker={e.ticker: identifying_tokens(e.company_name) for e in entries},
    )


def load_index(force_refresh: bool = False) -> CorpusIndex:
    """
    Build the corpus index from the metadata store, caching it per process.

    Reads every Document, then joins the fiscal period end from the latest
    successful extraction via the canonical read (D31) rather than re-deriving
    "latest" here.

    Args:
        force_refresh: Rebuild even if an index is cached. Needed after a
            pipeline run adds documents inside a live process.

    Returns:
        The index.

    Raises:
        RuntimeError: if the corpus holds no documents, which means the query
            path is running against an unpopulated database.
    """
    global _cached_index
    if _cached_index is not None and not force_refresh:
        return _cached_index

    with get_session() as session:
        documents = session.execute(select(Document)).scalars().all()
        period_ends = {
            doc.id: record.fiscal_year_end
            for doc, record in session.execute(latest_successful_stmt()).all()
        }

    if not documents:
        raise RuntimeError("corpus index is empty — no documents in the metadata store")

    entries = [
        CorpusEntry(
            ticker=doc.ticker,
            cik=doc.cik,
            company_name=doc.company_name,
            document_id=doc.id,
            filing_year=doc.filing_year,
            fiscal_year_end=period_ends.get(doc.id),
            is_extracted=doc.is_extracted,
            is_embedded=doc.is_embedded,
        )
        for doc in documents
    ]
    _cached_index = build_index(entries)
    logger.info("corpus_index_built", documents=len(entries))
    return _cached_index


def _entity(entry: CorpusEntry, mention: str) -> ResolvedEntity:
    """Project a corpus entry into the contract the executor consumes."""
    return ResolvedEntity(
        mention=mention,
        ticker=entry.ticker,
        cik=entry.cik,
        company_name=display_name(entry.company_name, entry.ticker),
        document_id=entry.document_id,
        filing_year=entry.filing_year,
        fiscal_year_end=entry.fiscal_year_end,
        is_extracted=entry.is_extracted,
        is_embedded=entry.is_embedded,
    )


def resolve_company(mention: str | None, index: CorpusIndex | None = None) -> Resolution:
    """
    Resolve a company mention against the corpus.

    Matching runs in three passes, most specific first: exact ticker, then the
    alias table, then identifying-token containment against registrant names. A
    mention resolves only when exactly one entry contains all of its identifying
    tokens; several means ambiguous, none means not held.

    Args:
        mention: The company as the user referred to it, or None.
        index: Corpus index. Loaded from the database when omitted.

    Returns:
        A Resolution carrying either the entity or a failure with its reason.
    """
    if mention is None or not mention.strip():
        return Resolution(
            failure=ResolutionFailure.NO_COMPANY_MENTION,
            detail="I couldn't tell which company you mean — the question didn't name one.",
        )

    index = index or load_index()
    by_ticker = {entry.ticker: entry for entry in index.entries}
    normalised = normalise(mention)

    upper = mention.strip().upper()
    if upper in by_ticker:
        return Resolution(entity=_entity(by_ticker[upper], mention))

    alias_ticker = ALIASES.get(normalised)
    if alias_ticker and alias_ticker in by_ticker:
        return Resolution(entity=_entity(by_ticker[alias_ticker], mention))

    wanted = identifying_tokens(mention)
    if not wanted:
        return Resolution(
            failure=ResolutionFailure.AMBIGUOUS_COMPANY,
            detail=f"{mention!r} isn't specific enough — several companies match.",
        )

    matches = [
        entry
        for entry in index.entries
        if wanted <= index.tokens_by_ticker[entry.ticker]
    ]
    if len(matches) == 1:
        return Resolution(entity=_entity(matches[0], mention))
    if matches:
        return Resolution(
            failure=ResolutionFailure.AMBIGUOUS_COMPANY,
            detail=f"{mention!r} matches several companies I hold.",
            candidates=sorted(entry.ticker for entry in matches),
        )

    logger.info("company_not_in_corpus", mention=mention)
    return Resolution(
        failure=ResolutionFailure.NON_CORPUS_COMPANY,
        detail=f"I don't hold a filing for {mention}.",
    )


def resolve_period(entity: ResolvedEntity, requested_year: int | None) -> Resolution:
    """
    Settle which fiscal year a question refers to.

    The corpus holds one filing per company, so an omitted year resolves to that
    filing and the answer states which year it was. A year that is named but not
    held is reported with the year that is, rather than as an empty result with
    no explanation.

    Relative periods never arrive here: the planner may not emit them, because
    "latest" denotes a different year for each company in this corpus.

    Args:
        entity: An already-resolved company.
        requested_year: The fiscal year the question named, if any.

    Returns:
        A Resolution carrying the entity, or a YEAR_NOT_HELD failure.
    """
    if requested_year is None or requested_year == entity.filing_year:
        return Resolution(entity=entity)

    return Resolution(
        failure=ResolutionFailure.YEAR_NOT_HELD,
        detail=(
            f"I hold one {entity.company_name} filing, for fiscal year "
            f"{entity.filing_year}. I don't have {requested_year}."
        ),
    )


def resolve(
    mention: str | None,
    requested_year: int | None = None,
    index: CorpusIndex | None = None,
    require_embedded: bool = False,
) -> Resolution:
    """
    Resolve a mention and a period together, with the serving check applied.

    Args:
        mention: The company as the user referred to it.
        requested_year: The fiscal year named, if any.
        index: Corpus index. Loaded from the database when omitted.
        require_embedded: Set by narrative tasks. A document with no chunks would
            retrieve nothing and look identical to a silent corpus, so it is
            refused with a stated reason instead.

    Returns:
        A Resolution carrying the entity or the first failure encountered.
    """
    resolution = resolve_company(mention, index=index)
    if resolution.entity is None:
        return resolution

    resolution = resolve_period(resolution.entity, requested_year)
    if resolution.entity is None:
        return resolution

    entity = resolution.entity
    if require_embedded and not entity.is_embedded:
        return Resolution(
            failure=ResolutionFailure.COMPANY_NOT_EMBEDDED,
            detail=f"{entity.company_name} has no indexed narrative text",
        )
    return resolution