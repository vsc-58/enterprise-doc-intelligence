"""
src/storage/vector_store.py
ChromaDB wrapper for the narrative RAG corpus.

Persists chunk text, embeddings and metadata at settings.CHROMA_DB_PATH and
serves filtered similarity search to the RAG path. The only consumer of
OpenAIEmbeddings in this project.

Cosine is set explicitly at collection creation. Chroma defaults to L2, and the
distance function CANNOT be changed on an existing collection — a collection
created without it stays L2 forever and must be deleted and rebuilt. For the
unit-length vectors OpenAI returns, all three metrics rank identically; cosine is
kept for its bounded, interpretable score in logs. It does NOT make a
distance-threshold out-of-scope guard viable: measured on the built store, a
wrong-store question scored third-closest of six (D23). Refusal rests on the
chain's prompt and on section filtering, never on a score cutoff.

Every score this module returns is a DISTANCE — lower is closer, 0 is identical.
langchain-chroma's `similarity_search_by_vector_with_relevance_scores` returns
distance despite its name; the field is called `distance` here so nobody has to
remember that.

Retrieval reports one of three statuses rather than an empty list for all of
them. A filter that matched nothing (INTC has no Item 7) is a correct answer
about the corpus; a broken or empty store is a fault. "The corpus is silent on
this" is NOT a retrieval status: nearest-neighbour search always returns the
nearest chunks whenever the filter matches anything, so silence can only be
judged by the chain reading them.

The store and the embedding client are cached at module level. Construction is
not free, and concurrent tasks would otherwise each build their own client.

The chunk-parameter hash is written to collection metadata at creation and
compared on every open. A store built at different chunk parameters or with a
different embedding model is not comparable to the current ones, and detecting
that is not left to anyone remembering to bump a version — the same reasoning as
Phase 3's derived prompt_version.

Re-chunking is handled by delete-then-insert per document, not by id collision:
a content+position hash would orphan the tail whenever a re-chunk produced fewer
chunks for a document, leaving stale chunks that no longer exist in the source
and that nothing would ever overwrite.

Dependencies: langchain-chroma, langchain-openai, chromadb, pydantic,
src.ingestion.chunker, src.utils.config, src.utils.logger.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Collection
from enum import Enum
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.chunker import chunk_params_hash
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "filing_narratives"
_PARAMS_KEY = "chunk_params_hash"

_store: Chroma | None = None
_store_lock = threading.Lock()


class VectorStoreError(RuntimeError):
    """Raised when the store or the embedding call fails on a single operation."""


class RetrievalStatus(str, Enum):
    """
    How one retrieval ended.

    OK: at least one chunk matched the filter and was returned.
    NO_MATCHING_CHUNKS: the store is healthy but holds nothing under this
        filter — a statement about the corpus, surfaced as a specific refusal.
    FAILED: the store, the embedding call, or the stored metadata is broken. An
        empty collection is FAILED, not NO_MATCHING_CHUNKS: every filter would
        miss, and calling that "no Item 7" would hide a deployment fault behind
        a plausible refusal (D22's silent-empty-store case).
    """

    OK = "ok"
    NO_MATCHING_CHUNKS = "no_matching_chunks"
    FAILED = "failed"


class RetrievedChunk(BaseModel):
    """
    One chunk as returned by a search.

    Attributes:
        text: Chunk text.
        ticker: source_ticker — the filtering key (D30).
        company_name: source_company, the EDGAR registrant name. A storage key;
            display it through src.utils.text.display_name (D37).
        cik: Zero-padded CIK.
        fiscal_year: source_year, the period-end filing_year key.
        section: Item label, exact by construction (D21).
        chunk_index: Position within its section.
        distance: Cosine distance to the query. Lower is closer.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    ticker: str
    company_name: str
    cik: str
    fiscal_year: int
    section: str
    chunk_index: int
    distance: float


class RetrievalResult(BaseModel):
    """
    The outcome of one search.

    Attributes:
        status: See RetrievalStatus.
        chunks: Nearest first. Empty unless status is OK.
        detail: Human-readable reason for a non-OK status, for logs and refusals.
    """

    model_config = ConfigDict(frozen=True)

    status: RetrievalStatus
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    detail: str | None = None


@lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    """Construct the embedding client once per process.

    api_key is passed explicitly: pydantic-settings loads .env into the Settings
    object, NOT into os.environ, where the OpenAI SDK looks for it. Phase 2
    bug 5 — the same failure will occur here otherwise, and passing it from
    settings is also what the no-hardcode rule requires.

    Returns:
        A configured OpenAIEmbeddings client.
    """
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )


def _open_store() -> Chroma:
    """Open the persistent Chroma collection, creating it if absent.

    On an existing collection, the stored chunk-parameter hash is compared with
    the current one and a mismatch is logged at ERROR. It is logged rather than
    raised because a mismatch is recoverable (reset_store, then re-embed) and
    raising here would break read-only callers such as the RAG path, which can
    still serve queries from a stale-but-valid store.

    Returns:
        A Chroma instance bound to the persistent directory.

    Raises:
        VectorStoreError: if the collection cannot be opened or created.
    """
    current = chunk_params_hash()
    try:
        store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=_embeddings(),
            persist_directory=settings.CHROMA_DB_PATH,
            collection_metadata={
                "hnsw:space": "cosine",
                _PARAMS_KEY: current,
            },
        )
    except Exception as exc:
        logger.error(
            "chroma_open_failed",
            path=settings.CHROMA_DB_PATH,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise VectorStoreError(
            f"Could not open Chroma collection at {settings.CHROMA_DB_PATH}"
        ) from exc

    stored = (store._collection.metadata or {}).get(_PARAMS_KEY)
    if stored is not None and stored != current:
        logger.error(
            "chunk_params_mismatch",
            stored_hash=stored,
            current_hash=current,
            action="reset_store() and re-embed, or revert the chunk settings",
        )
    return store


def get_store() -> Chroma:
    """Return the process-wide store, opening it on first use.

    Double-checked under a lock: the executor runs tasks in worker threads, and
    two tasks arriving together would otherwise both open the collection.

    Returns:
        The cached Chroma instance.

    Raises:
        VectorStoreError: if the collection cannot be opened.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = _open_store()
    return _store


def _invalidate_store_cache() -> None:
    """Drop the cached store so the next get_store() reopens the collection."""
    global _store
    with _store_lock:
        _store = None


def reset_store() -> None:
    """Delete the entire collection so it can be rebuilt from scratch.

    Required after any chunk-parameter or embedding-model change: the distance
    function and the parameter hash are fixed at creation, so the collection
    must be dropped rather than mutated. The caller is responsible for resetting
    Document.is_embedded, or the rebuild will skip every document and leave an
    empty store.

    Raises:
        VectorStoreError: if the collection cannot be deleted.
    """
    try:
        store = get_store()
        store.delete_collection()
        logger.warning("collection_deleted", collection=_COLLECTION_NAME)
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.error(
            "collection_delete_failed",
            collection=_COLLECTION_NAME,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise VectorStoreError("Could not delete Chroma collection") from exc
    finally:
        # The cached instance points at the deleted collection; the next caller
        # must open a fresh one, carrying the current parameter hash.
        _invalidate_store_cache()


def delete_document_chunks(cik: str, year: int) -> None:
    """Remove every chunk belonging to one document.

    Called before inserting a document's chunks so re-embedding replaces rather
    than accumulates. Deleting by metadata filter rather than by id removes
    chunks the current parameters would no longer produce — a shorter re-chunk
    would otherwise leave the tail behind.

    Args:
        cik: Zero-padded 10-digit CIK.
        year: Filing year.

    Raises:
        VectorStoreError: if the delete fails, since proceeding would duplicate.
    """
    try:
        store = get_store()
        store._collection.delete(
            where={"$and": [{"source_cik": cik}, {"source_year": year}]}
        )
        logger.debug("document_chunks_deleted", cik=cik, year=year)
    except Exception as exc:
        logger.error(
            "document_chunk_delete_failed",
            cik=cik,
            year=year,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise VectorStoreError(f"Could not delete existing chunks for {cik}_{year}") from exc


def add_chunks(chunks: list[dict[str, Any]], replace: bool = True) -> int:
    """Embed and store a document's chunks.

    All chunks must belong to one document — replace deletes by the first
    chunk's cik/year, so a mixed list would delete one document's chunks and
    insert another's.

    Batched at settings.EMBED_BATCH_SIZE with a pause between batches: batching
    bounds the blast radius of a failed call, and the pause keeps a ~1,750-chunk
    run inside rate limits. A failed batch raises rather than continuing —
    unlike the per-document batch loop in the calling script, a partial insert
    within one document would flip is_embedded on incomplete data.

    Args:
        chunks: Chunk dicts from chunker.chunk_document.
        replace: Whether to delete the document's existing chunks first.

    Returns:
        Number of chunks written.

    Raises:
        VectorStoreError: on any embedding or insert failure.
    """
    if not chunks:
        logger.warning("add_chunks_called_with_empty_list")
        return 0

    first = chunks[0]["metadata"]
    cik, year = first["source_cik"], first["source_year"]

    if replace:
        delete_document_chunks(cik, year)

    store = get_store()
    written = 0
    batch_size = settings.EMBED_BATCH_SIZE

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        documents = [
            LCDocument(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in batch
        ]
        ids = [chunk["id"] for chunk in batch]

        try:
            store.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            logger.error(
                "batch_embed_failed",
                cik=cik,
                year=year,
                batch_start=start,
                batch_size=len(batch),
                written_before_failure=written,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise VectorStoreError(
                f"Embedding failed for {cik}_{year} at batch offset {start}"
            ) from exc

        written += len(batch)
        if start + batch_size < len(chunks):
            time.sleep(settings.EMBED_BATCH_SLEEP_SECONDS)

    logger.info("chunks_embedded", cik=cik, year=year, chunk_count=written)
    return written


def embed_query(text: str) -> list[float]:
    """Embed a query string once, so it can be reused across filtered searches.

    Corpus-wide retrieval runs the same question against every company; embedding
    it per company would make N identical API calls for one vector.

    Args:
        text: Query text.

    Returns:
        The query embedding.

    Raises:
        VectorStoreError: if the embedding call fails.
    """
    try:
        return _embeddings().embed_query(text)
    except Exception as exc:
        logger.error(
            "query_embedding_failed",
            query_preview=text[:80],
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise VectorStoreError("Could not embed the query") from exc


def build_filter(
    ticker: str | None = None,
    sections: Collection[str] | None = None,
) -> dict[str, Any] | None:
    """Build a Chroma `where` clause from a ticker and a set of sections.

    Filters on source_ticker, never source_company, which holds registrant names
    such as "BANK OF AMERICA CORP /DE/" (D30). Sections are matched with $in: an
    intent's sections are a union with no priority order (D38). Sorted so the
    same policy always produces the same clause in logs.

    Args:
        ticker: Restrict to one company, or None for all.
        sections: Restrict to these item labels, or None/empty for all.

    Returns:
        The where clause, or None when nothing is filtered.
    """
    clauses: list[dict[str, Any]] = []
    if ticker:
        clauses.append({"source_ticker": ticker})
    if sections:
        clauses.append({"source_section": {"$in": sorted(sections)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _to_chunk(document: LCDocument, distance: float) -> RetrievedChunk:
    """Convert a LangChain document and its distance into a RetrievedChunk.

    Args:
        document: A search hit.
        distance: Its cosine distance.

    Returns:
        The typed chunk.

    Raises:
        KeyError / ValueError: if stored metadata lacks a field or has the wrong
            type — caught by the caller and reported as FAILED.
    """
    meta = document.metadata
    return RetrievedChunk(
        text=document.page_content,
        ticker=str(meta["source_ticker"]),
        company_name=str(meta["source_company"]),
        cik=str(meta["source_cik"]),
        fiscal_year=int(meta["source_year"]),
        section=str(meta["source_section"]),
        chunk_index=int(meta["chunk_index"]),
        distance=float(distance),
    )


def query_by_vector(
    vector: list[float],
    k: int,
    ticker: str | None = None,
    sections: Collection[str] | None = None,
) -> RetrievalResult:
    """Run one filtered similarity search with a precomputed query embedding.

    Never raises on a store or data fault: a retrieval failure must surface as a
    refusal in the answer, and in a corpus-wide loop one company's failure must
    not sink the others.

    Args:
        vector: Query embedding from embed_query.
        k: Maximum chunks to return. Fewer come back when fewer match.
        ticker: Restrict to one company.
        sections: Restrict to these item labels (union).

    Returns:
        A RetrievalResult. OK with chunks nearest first; NO_MATCHING_CHUNKS when
        the store is healthy but nothing matches the filter; FAILED otherwise.

    Raises:
        ValueError: if k < 1 — a programming error, not a runtime condition.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    where = build_filter(ticker, sections)
    try:
        hits = get_store().similarity_search_by_vector_with_relevance_scores(
            embedding=vector, k=k, filter=where
        )
        chunks = [_to_chunk(document, distance) for document, distance in hits]
    except Exception as exc:
        logger.error(
            "similarity_search_failed",
            ticker=ticker,
            sections=sorted(sections) if sections else None,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return RetrievalResult(
            status=RetrievalStatus.FAILED,
            detail=f"vector search failed: {type(exc).__name__}",
        )

    if chunks:
        logger.debug(
            "similarity_search",
            ticker=ticker,
            sections=sorted(sections) if sections else None,
            result_count=len(chunks),
            best_distance=chunks[0].distance,
        )
        return RetrievalResult(status=RetrievalStatus.OK, chunks=chunks)

    # Nothing matched. Only a non-empty, readable collection makes that a
    # statement about the corpus rather than a fault.
    count = collection_count()
    if count <= 0:
        logger.error("vector_store_empty_or_unreadable", collection_count=count)
        return RetrievalResult(
            status=RetrievalStatus.FAILED,
            detail="the narrative store is empty or unreadable",
        )

    logger.info(
        "similarity_search_no_match",
        ticker=ticker,
        sections=sorted(sections) if sections else None,
    )
    return RetrievalResult(
        status=RetrievalStatus.NO_MATCHING_CHUNKS,
        detail="no indexed text matches this company and section filter",
    )


def query(
    text: str,
    k: int,
    ticker: str | None = None,
    sections: Collection[str] | None = None,
) -> RetrievalResult:
    """Embed a query and run one filtered similarity search.

    Convenience for single searches. Callers searching the same question under
    several filters should call embed_query once and query_by_vector per filter.

    Args:
        text: Query text.
        k: Maximum chunks to return.
        ticker: Restrict to one company.
        sections: Restrict to these item labels (union).

    Returns:
        A RetrievalResult; FAILED if the embedding call fails.

    Raises:
        ValueError: if k < 1.
    """
    try:
        vector = embed_query(text)
    except VectorStoreError as exc:
        return RetrievalResult(status=RetrievalStatus.FAILED, detail=str(exc))
    return query_by_vector(vector, k=k, ticker=ticker, sections=sections)


def collection_count() -> int:
    """Return the number of chunks currently stored.

    Returns:
        Chunk count, or -1 if the collection cannot be read (logged).
    """
    try:
        return get_store()._collection.count()
    except Exception as exc:
        logger.error(
            "collection_count_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return -1