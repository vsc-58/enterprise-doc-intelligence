"""
scripts/embed_documents.py
Chunk and embed every document whose narrative sections have not yet been
indexed, and flip Document.is_embedded on success.

Entrypoint for the Phase 4 indexing step. Reads narrative artifacts from disk
(written by scripts/build_narrative_sections.py), chunks them, and writes chunks
plus embeddings to ChromaDB.

Discipline mirrors Phase 3's is_extracted: the flag is set ONLY after the
document's chunks are fully written. A partially embedded document therefore
keeps is_embedded=False and is picked up by a re-run — a half-indexed document
that claimed to be complete would silently under-serve retrieval with no error
anywhere.

Per-document error handling: one failure is logged with company and year and the
batch continues. Within a document, a failed batch aborts that document (the
flag stays False), because a partial insert is worse than none.

Usage:
    python -m scripts.embed_documents              # pending documents only
    python -m scripts.embed_documents --all        # re-embed everything
    python -m scripts.embed_documents --reset      # drop collection, rebuild all
    python -m scripts.embed_documents --dry-run    # chunk and cost only, no spend

Dependencies: sqlalchemy, src.ingestion.chunker, src.storage.vector_store,
src.storage.metadata_store, src.utils.config, src.utils.logger.
"""

from __future__ import annotations

import argparse

import tiktoken
from sqlalchemy import select, update

from src.ingestion.chunker import chunk_document, chunk_params_hash
from src.storage.metadata_store import Document, get_session
from src.storage.vector_store import (
    add_chunks,
    collection_count,
    reset_store,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
# text-embedding-3-small list price per 1M tokens. A module-level vendor fact,
# not a setting — same classification as the pricing constants in
# run_eval.py / run_extraction.py, and the same known staleness point.
_EMBEDDING_USD_PER_1M = 0.02
PROGRESS_EVERY = 5


def clear_embedded_flags() -> int:
    """Set is_embedded=False on every document.

    Paired with reset_store(): dropping the collection without clearing the
    flags leaves every document marked embedded against an empty store, and the
    next run would skip all of them. The two must happen together.

    Returns:
        Number of rows updated.

    Raises:
        RuntimeError: if the update fails — proceeding would produce exactly the
            empty-store-with-set-flags state this exists to prevent.
    """
    try:
        with get_session() as session:
            result = session.execute(
                update(Document).values(is_embedded=False)
            )
            return int(result.rowcount or 0)
    except Exception as exc:
        logger.error(
            "clear_embedded_flags_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise RuntimeError("Could not clear is_embedded flags") from exc


def mark_embedded(document_id: int) -> None:
    """Flip is_embedded to True for one document.

    Called only after every chunk for that document has been written.

    Args:
        document_id: Primary key of the Document row.

    Raises:
        RuntimeError: if the update fails, so the caller counts the document as
            failed rather than reporting a success the database does not record.
    """
    try:
        with get_session() as session:
            session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(is_embedded=True)
            )
    except Exception as exc:
        logger.error(
            "mark_embedded_failed",
            document_id=document_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise RuntimeError(f"Could not set is_embedded for document {document_id}") from exc


def select_targets(embed_all: bool) -> list[Document]:
    """Load the documents to process, detached from the session.

    Attributes are read into plain values before the session closes, so the
    embedding loop — which runs for minutes — never holds an open transaction.

    Args:
        embed_all: Whether to include documents already embedded.

    Returns:
        Document rows in ticker order.
    """
    with get_session() as session:
        statement = select(Document).order_by(Document.ticker)
        if not embed_all:
            statement = statement.where(Document.is_embedded.is_(False))
        documents = list(session.scalars(statement))
        for doc in documents:
            _ = (doc.id, doc.cik, doc.ticker, doc.filing_year, doc.company_name)
        session.expunge_all()
    return documents


def estimate_cost(documents: list[Document]) -> tuple[int, int, float]:
    """Chunk every target and report the cost before any spend.

    Args:
        documents: Documents to chunk.

    Returns:
        Tuple of (chunk count, token count, estimated USD).
    """
    chunks_total, tokens_total = 0, 0
    for doc in documents:
        chunks = chunk_document(doc.cik, doc.filing_year)
        chunks_total += len(chunks)
        tokens_total += sum(len(_ENCODING.encode(c["text"])) for c in chunks)
    return (
        chunks_total,
        tokens_total,
        tokens_total / 1_000_000 * _EMBEDDING_USD_PER_1M,
    )


def embed_one(doc: Document) -> int:
    """Chunk and embed one document, then mark it embedded.

    Args:
        doc: Document row to process.

    Returns:
        Number of chunks written, or 0 if the document produced no chunks
        (logged — the flag is NOT set, so a re-run retries it).

    Raises:
        RuntimeError: propagated from add_chunks or mark_embedded so the caller
            records the document as failed.
    """
    chunks = chunk_document(doc.cik, doc.filing_year)
    if not chunks:
        logger.error(
            "no_chunks_produced",
            company=doc.company_name,
            ticker=doc.ticker,
            year=doc.filing_year,
        )
        return 0

    written = add_chunks(chunks, replace=True)
    mark_embedded(doc.id)
    return written


def main(
    embed_all: bool = False, reset: bool = False, dry_run: bool = False
) -> None:
    """Embed pending documents into the vector store.

    Args:
        embed_all: Re-embed documents already marked embedded.
        reset: Drop the collection and clear all flags before embedding.
        dry_run: Chunk and report cost without embedding.
    """
    if reset:
        logger.warning("reset_requested", params_hash=chunk_params_hash())
        reset_store()
        cleared = clear_embedded_flags()
        logger.warning("flags_cleared", rows=cleared)
        embed_all = True

    documents = select_targets(embed_all)
    if not documents:
        logger.info("nothing_to_embed", collection_count=collection_count())
        return

    chunks_expected, tokens, usd = estimate_cost(documents)
    logger.info(
        "embed_planned",
        documents=len(documents),
        chunks=chunks_expected,
        tokens=tokens,
        estimated_usd=round(usd, 4),
        params_hash=chunk_params_hash(),
    )

    if dry_run:
        logger.info("dry_run_complete", note="no embedding performed")
        return

    embedded, failed, chunks_written = 0, [], 0
    for index, doc in enumerate(documents, start=1):
        try:
            written = embed_one(doc)
        except Exception as exc:
            logger.error(
                "document_embed_failed",
                company=doc.company_name,
                ticker=doc.ticker,
                year=doc.filing_year,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            failed.append(doc.ticker)
            continue

        if written == 0:
            failed.append(doc.ticker)
            continue

        embedded += 1
        chunks_written += written
        if index % PROGRESS_EVERY == 0:
            logger.info(
                "embed_progress",
                processed=index,
                total=len(documents),
                chunks_written=chunks_written,
            )

    logger.info(
        "embed_complete",
        embedded=embedded,
        failed=len(failed),
        failed_tickers=failed,
        chunks_written=chunks_written,
        chunks_expected=chunks_expected,
        collection_count=collection_count(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed narrative chunks into ChromaDB.")
    parser.add_argument("--all", action="store_true", help="re-embed every document")
    parser.add_argument("--reset", action="store_true", help="drop collection and rebuild")
    parser.add_argument("--dry-run", action="store_true", help="report cost, do not embed")
    args = parser.parse_args()
    main(embed_all=args.all, reset=args.reset, dry_run=args.dry_run)