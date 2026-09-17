"""
src/storage/vector_store.py
ChromaDB wrapper for the narrative RAG corpus.

Persists chunk text, embeddings and metadata at settings.CHROMA_DB_PATH and
serves similarity search to the RAG path (Phase 5). The only consumer of
OpenAIEmbeddings in this project.

Cosine is set explicitly at collection creation. Chroma defaults to L2, and the
distance function CANNOT be changed on an existing collection — a collection
created without it stays L2 forever and must be deleted and rebuilt. For the
unit-length vectors OpenAI returns, all three metrics rank identically; cosine
is chosen for its bounded [-1, 1] score, which keeps a threshold-based
out-of-scope guard available to the RAG chain.

The chunk-parameter hash is written to collection metadata at creation and
compared on every open. A store built at different chunk parameters or with a
different embedding model is not comparable to the current ones, and detecting
that is not left to anyone remembering to bump a version — the same reasoning as
Phase 3's derived prompt_version.

Re-chunking is handled by delete-then-insert per document, not by id collision:
a content+position hash would orphan the tail whenever a re-chunk produced fewer
chunks for a document, leaving stale chunks that no longer exist in the source
and that nothing would ever overwrite.

Dependencies: langchain-chroma, langchain-openai, chromadb, src.ingestion.chunker,
src.utils.config, src.utils.logger.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings

from src.ingestion.chunker import chunk_params_hash
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "filing_narratives"
_PARAMS_KEY = "chunk_params_hash"


def _build_embeddings() -> OpenAIEmbeddings:
    """Construct the embedding client.

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


def get_store() -> Chroma:
    """Open the persistent Chroma collection, creating it if absent.

    On an existing collection, the stored chunk-parameter hash is compared with
    the current one and a mismatch is logged at ERROR. It is logged rather than
    raised because a mismatch is recoverable (reset_store, then re-embed) and
    raising here would break read-only callers such as the RAG path, which can
    still serve queries from a stale-but-valid store.

    Returns:
        A Chroma instance bound to the persistent directory.

    Raises:
        RuntimeError: if the collection cannot be opened or created.
    """
    current = chunk_params_hash()
    try:
        store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=_build_embeddings(),
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
        raise RuntimeError(
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


def reset_store() -> None:
    """Delete the entire collection so it can be rebuilt from scratch.

    Required after any chunk-parameter or embedding-model change: the distance
    function and the parameter hash are fixed at creation, so the collection
    must be dropped rather than mutated. The caller is responsible for resetting
    Document.is_embedded, or the rebuild will skip every document and leave an
    empty store.

    Raises:
        RuntimeError: if the collection cannot be deleted.
    """
    try:
        store = get_store()
        store.delete_collection()
        logger.warning("collection_deleted", collection=_COLLECTION_NAME)
    except Exception as exc:
        logger.error(
            "collection_delete_failed",
            collection=_COLLECTION_NAME,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise RuntimeError("Could not delete Chroma collection") from exc


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
        RuntimeError: if the delete fails, since proceeding would duplicate.
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
        raise RuntimeError(f"Could not delete existing chunks for {cik}_{year}") from exc


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
        RuntimeError: on any embedding or insert failure.
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
            raise RuntimeError(
                f"Embedding failed for {cik}_{year} at batch offset {start}"
            ) from exc

        written += len(batch)
        if start + batch_size < len(chunks):
            time.sleep(settings.EMBED_BATCH_SLEEP_SECONDS)

    logger.info("chunks_embedded", cik=cik, year=year, chunk_count=written)
    return written


def query(
    text: str,
    n_results: int = 5,
    company: str | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the chunks most similar to a query string.

    The score is Chroma's cosine DISTANCE, not similarity: lower is closer,
    0 is identical. Any threshold the RAG chain applies must be written in
    those terms.

    Args:
        text: Query text.
        n_results: Number of chunks to return.
        company: Optional exact company-name filter.
        section: Optional item-label filter, e.g. "Item 1A".

    Returns:
        Result dicts with text, metadata and score, nearest first. Empty on
        failure (logged) — a retrieval failure must surface as the RAG chain's
        refusal, not as a 500 from the API layer.
    """
    clauses: list[dict[str, Any]] = []
    if company:
        clauses.append({"source_company": company})
    if section:
        clauses.append({"source_section": section})

    where: dict[str, Any] | None = None
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}

    try:
        store = get_store()
        hits = store.similarity_search_with_score(
            query=text, k=n_results, filter=where
        )
    except Exception as exc:
        logger.error(
            "similarity_search_failed",
            query_preview=text[:80],
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return []

    results = [
        {
            "text": document.page_content,
            "metadata": document.metadata,
            "score": float(score),
        }
        for document, score in hits
    ]
    logger.debug("similarity_search", result_count=len(results), n_requested=n_results)
    return results


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