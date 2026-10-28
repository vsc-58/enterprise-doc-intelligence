"""Narrative retrieval: turn a narrative task into numbered context blocks.

Returns one company's blocks for a single-company task, or one set per
eligible company for a corpus-wide task, each carrying the source identity its
citation needs.

Deterministic: no LLM call. Section policy is applied here and nowhere else,
including the one section fallback: when a company holds none of an intent's
sections, the intent's fallback sections are searched instead. The trigger is
structural absence in the store, never the chain's judgment.

Block numbers are LOCAL to one CompanyContext. The chain cites by number, and
the number maps back to a block by construction; but in a corpus-wide answer
every company's blocks start at 1, so the number is not a citation key across
companies. ContextBlock.source_ref() carries the chunk's real identity, and the
synthesizer assigns final display numbers when it merges sources.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.query.intents import fallback_for, sections_for
from src.query.schemas import Intent, SourceRef
from src.storage.vector_store import (
    RetrievalStatus,
    RetrievedChunk,
    VectorStoreError,
    embed_query,
    query_by_vector,
)
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)


class CompanyRef(BaseModel):
    """The identity a retrieval needs for one company.

    Built from a ResolvedEntity (single-company) or a CorpusEntry (corpus-wide),
    so the retriever depends on neither.

    Attributes:
        ticker: Filtering key. Chunks are filtered by ticker, never by the
            registrant name, which a model-written name would not match.
        company_name: Registrant name as stored.
        fiscal_year: The filing_year the corpus holds for this company.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    company_name: str
    fiscal_year: int


class ContextBlock(BaseModel):
    """One numbered chunk as the chain will see it.

    Attributes:
        number: 1-based position within its CompanyContext.
        chunk: The retrieved chunk.
    """

    model_config = ConfigDict(frozen=True)

    number: int
    chunk: RetrievedChunk

    def source_ref(self) -> SourceRef:
        """The citation record for this block.

        Returns:
            A SourceRef carrying the chunk's identity and distance.
        """
        return SourceRef(
            ticker=self.chunk.ticker,
            company_name=self.chunk.company_name,
            fiscal_year=self.chunk.fiscal_year,
            section=self.chunk.section,
            chunk_index=self.chunk.chunk_index,
            distance=self.chunk.distance,
        )


class CompanyContext(BaseModel):
    """Retrieval for one company under one intent.

    Attributes:
        company: Which company.
        sections_searched: Stored section labels filtered on, after aliases.
            Recorded so a refusal can say what was searched, and so the retrieval
            eval can check the filter rather than infer it. When a fallback
            applied, these are the fallback sections.
        status: OK, NO_MATCHING_CHUNKS (the company holds none of the sections)
            or FAILED.
        blocks: Numbered from 1, nearest first. Empty unless status is OK.
        detail: Reason for a non-OK status.
        fallback_from: The intent's own sections, when they were absent and the
            fallback was searched instead. Empty when no fallback applied.
    """

    model_config = ConfigDict(frozen=True)

    company: CompanyRef
    sections_searched: frozenset[str]
    status: RetrievalStatus
    blocks: list[ContextBlock] = Field(default_factory=list)
    detail: str | None = None
    fallback_from: frozenset[str] = frozenset()

    def block(self, number: int) -> ContextBlock | None:
        """Look up a block by the number the chain cited.

        Args:
            number: A cited block number.

        Returns:
            The block, or None if the chain cited a number that does not exist —
            which the caller must treat as an invalid citation, not ignore.
        """
        if 1 <= number <= len(self.blocks):
            return self.blocks[number - 1]
        return None


def _number(chunks: Sequence[RetrievedChunk]) -> list[ContextBlock]:
    """Number chunks from 1 in the order given (nearest first).

    Args:
        chunks: Retrieved chunks.

    Returns:
        Numbered blocks.
    """
    return [ContextBlock(number=i, chunk=chunk) for i, chunk in enumerate(chunks, start=1)]


def _retrieve(
    vector: list[float],
    intent: Intent,
    company: CompanyRef,
    k: int,
) -> CompanyContext:
    """Run one company's filtered search with a precomputed query vector.

    If the company holds none of the intent's sections and the intent has a
    fallback, the fallback sections are searched once with the same vector. The
    fallback result is returned whatever its status, except a second
    NO_MATCHING_CHUNKS: then the original absence is returned, so a refusal
    names the sections the question actually needed.

    Args:
        vector: Query embedding.
        intent: Narrative intent, which fixes the canonical sections.
        company: The company to search.
        k: Maximum chunks.

    Returns:
        The company's context.
    """
    sections = sections_for(intent, company.ticker)
    result = query_by_vector(vector, k=k, ticker=company.ticker, sections=sections)
    primary = CompanyContext(
        company=company,
        sections_searched=sections,
        status=result.status,
        blocks=_number(result.chunks),
        detail=result.detail,
    )

    fallback = fallback_for(intent, company.ticker)
    if result.status is not RetrievalStatus.NO_MATCHING_CHUNKS or fallback is None:
        return primary

    second = query_by_vector(vector, k=k, ticker=company.ticker, sections=fallback.sections)
    if second.status is RetrievalStatus.NO_MATCHING_CHUNKS:
        return primary

    logger.info(
        "narrative_section_fallback",
        ticker=company.ticker,
        intent=intent.value,
        absent=sorted(sections),
        searched=sorted(fallback.sections),
        status=second.status.value,
        blocks=len(second.chunks),
    )
    return CompanyContext(
        company=company,
        sections_searched=fallback.sections,
        status=second.status,
        blocks=_number(second.chunks),
        detail=second.detail,
        fallback_from=sections,
    )


def _all_failed(
    intent: Intent,
    companies: Sequence[CompanyRef],
    detail: str,
) -> list[CompanyContext]:
    """One FAILED context per company, for when the shared embedding call fails.

    Args:
        intent: Narrative intent.
        companies: The companies that would have been searched.
        detail: Failure reason.

    Returns:
        FAILED contexts, in input order.
    """
    return [
        CompanyContext(
            company=company,
            sections_searched=sections_for(intent, company.ticker),
            status=RetrievalStatus.FAILED,
            detail=detail,
        )
        for company in companies
    ]


def retrieve_for_company(
    question: str,
    intent: Intent,
    company: CompanyRef,
    k: int | None = None,
) -> CompanyContext:
    """Retrieve context for a single-company narrative task.

    Never raises on a runtime fault: an embedding or store failure comes back as
    a FAILED context so the executor can refuse that task and keep the others.

    Args:
        question: The task's self-contained question, as the planner wrote it.
        intent: Narrative intent.
        company: The resolved company.
        k: Maximum chunks; settings.RAG_TOP_K when omitted.

    Returns:
        The company's context.
    """
    try:
        vector = embed_query(question)
    except VectorStoreError as exc:
        return _all_failed(intent, [company], str(exc))[0]

    context = _retrieve(vector, intent, company, k or settings.RAG_TOP_K)
    logger.info(
        "narrative_retrieved",
        ticker=company.ticker,
        intent=intent.value,
        sections=sorted(context.sections_searched),
        fallback_from=sorted(context.fallback_from),
        status=context.status.value,
        blocks=len(context.blocks),
    )
    return context


def retrieve_corpus_wide(
    question: str,
    intent: Intent,
    companies: Sequence[CompanyRef],
    k: int | None = None,
) -> list[CompanyContext]:
    """Retrieve context for a corpus-wide narrative task, one company at a time.

    Embeds the question once and runs one filtered search per company, so the
    largest filers cannot crowd the rest out of a shared top-k. The
    searches are local and take milliseconds, so they run sequentially in the
    caller's worker thread; only the verdict LLM calls downstream are concurrent.

    A company holding none of the intent's sections comes back as
    NO_MATCHING_CHUNKS and is reported as not searchable, never silently dropped
    — the coverage statement depends on it.

    Args:
        question: The task's self-contained question.
        intent: Narrative intent.
        companies: Every company eligible for narrative retrieval (embedded).
        k: Chunks per company; settings.RAG_PER_COMPANY_K when omitted.

    Returns:
        One context per company, in input order.
    """
    try:
        vector = embed_query(question)
    except VectorStoreError as exc:
        return _all_failed(intent, companies, str(exc))

    per_company_k = k or settings.RAG_PER_COMPANY_K
    contexts = [_retrieve(vector, intent, company, per_company_k) for company in companies]

    logger.info(
        "narrative_retrieved_corpus_wide",
        intent=intent.value,
        companies=len(contexts),
        searchable=sum(c.status is RetrievalStatus.OK for c in contexts),
        not_searchable=[c.company.ticker for c in contexts
                        if c.status is RetrievalStatus.NO_MATCHING_CHUNKS],
        fallback=[c.company.ticker for c in contexts if c.fallback_from],
        failed=[c.company.ticker for c in contexts if c.status is RetrievalStatus.FAILED],
    )
    return contexts


def format_context(context: CompanyContext) -> str:
    """Render a company's blocks as the text the chain receives.

    Each block is headed by its number, company, fiscal year and the stored
    section label, so the model can cite by number and the reader can see the
    provenance. The label is the stored one: for an aliased company the citation
    honestly says Item 1A, where the text actually sits (the section slicer
    found QCOM's and META's business description inside their Item 1A).

    Args:
        context: An OK context.

    Returns:
        Blocks separated by blank lines, or an empty string if there are none.
    """
    name = display_name(context.company.company_name, context.company.ticker)
    return "\n\n".join(
        f"[{block.number}] {name}, FY{block.chunk.fiscal_year}, {block.chunk.section}\n"
        f"{block.chunk.text.strip()}"
        for block in context.blocks
    )