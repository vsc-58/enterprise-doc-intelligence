"""
src/ingestion/chunker.py
Recursive-character chunking of narrative sections for the vector store.

Reads the artifacts written by scripts/build_narrative_sections.py and splits
each section into retrieval-sized chunks carrying the metadata the RAG path's
Sources block is built from.

Chunks NEVER span a section boundary: each section is split independently, so
every chunk's source_section is exact rather than inferred. This is why the
section slicer runs first — a chunk straddling Item 1A and Item 7 could not be
honestly labelled either.

Chunk ids are deterministic over content AND position. Content alone collides on
boilerplate repeated across filings; position alone silently overwrites and
orphans the tail when a re-chunk produces fewer chunks. Re-chunking at different
parameters is handled by deleting the document's chunks and reinserting, not by
id collision — see vector_store.py.

Token-aware sizing: RecursiveCharacterTextSplitter measures in characters by
default, so from_tiktoken_encoder is used to make the ~500-token target real
rather than a character approximation that drifts with table density.

Dependencies: langchain-text-splitters, tiktoken, src.utils.config,
src.utils.logger.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Separator hierarchy, most to least preferred. Paragraph break first: in Item 1A
# each risk heading and its explanation are separated by blank lines, so
# splitting there keeps a heading attached to the text it introduces far more
# often than a character split would.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Construct the token-aware recursive splitter.

    Returns:
        A splitter sized in tokens under the embedding model's encoding.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.CHUNK_SIZE_TOKENS,
        chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
        separators=_SEPARATORS,
    )


def chunk_id(cik: str, year: int, section: str, index: int, text: str) -> str:
    """Build a deterministic id for one chunk.

    Hashes document identity, section, position and content together. Identical
    inputs always produce the same id, so re-running the pipeline without
    changing anything cannot duplicate a chunk.

    Args:
        cik: Zero-padded 10-digit CIK.
        year: Filing year.
        section: Item label, e.g. "Item 1A".
        index: Position of this chunk within its section.
        text: Chunk text.

    Returns:
        A 32-character hex digest.
    """
    payload = f"{cik}|{year}|{section}|{index}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def chunk_params_hash() -> str:
    """Derive a hash over every parameter that changes chunk output.

    Stored in the Chroma collection metadata so a parameter change is detectable
    rather than depending on someone remembering to bump a version — the same
    reasoning as Phase 3's derived prompt_version. Includes the embedding model,
    because a re-embed with a different model invalidates the store just as
    surely as a re-split.

    Returns:
        A 12-character hex digest.
    """
    payload = "|".join(
        [
            str(settings.CHUNK_SIZE_TOKENS),
            str(settings.CHUNK_OVERLAP_TOKENS),
            "".join(_SEPARATORS),
            settings.EMBEDDING_MODEL,
            "cl100k_base",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_narrative_artifact(path: Path) -> dict[str, Any] | None:
    """Read one narrative section artifact from disk.

    Args:
        path: Path to the artifact JSON.

    Returns:
        The parsed artifact, or None if unreadable or malformed (logged).
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            "artifact_unreadable",
            path=str(path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None


def chunk_artifact(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Split every section in one document's artifact into chunks.

    chunk_index restarts per section, so a chunk is identified by the triple
    (document, section, index). A corpus-wide running index would change for
    every chunk whenever one section's boundaries shifted.

    Args:
        artifact: Parsed narrative artifact with cik, filing_year, company_name
            and a list of sections.

    Returns:
        Chunk dicts, each with id, text, and flat scalar metadata. Chroma
        rejects None and non-scalar metadata values, so every field is a
        populated str or int.
    """
    splitter = _build_splitter()
    cik = artifact["cik"]
    year = int(artifact["filing_year"])
    company = artifact["company_name"]

    chunks: list[dict[str, Any]] = []
    for section in artifact["sections"]:
        label = section["item_label"]
        try:
            pieces = splitter.split_text(section["text"])
        except Exception as exc:
            logger.error(
                "section_split_failed",
                company=company,
                year=year,
                section=label,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue

        kept = 0
        for index, piece in enumerate(pieces):
            if len(piece.strip()) < settings.CHUNK_MIN_CHARS:
                continue
            chunks.append(
                {
                    "id": chunk_id(cik, year, label, index, piece),
                    "text": piece,
                    "metadata": {
                        "source_company": company,
                        "source_ticker": artifact["ticker"],
                        "source_cik": cik,
                        "source_year": year,
                        "source_section": label,
                        "chunk_index": index,
                    },
                }
            )
            kept += 1

        logger.debug(
            "section_chunked",
            company=company,
            year=year,
            section=label,
            produced=len(pieces),
            kept=kept,
        )

    logger.info(
        "document_chunked", company=company, year=year, chunk_count=len(chunks)
    )
    return chunks


def chunk_document(cik: str, year: int) -> list[dict[str, Any]]:
    """Load one document's narrative artifact and chunk it.

    Args:
        cik: Zero-padded 10-digit CIK.
        year: Filing year.

    Returns:
        Chunk dicts, or an empty list if the artifact is missing or unreadable
        (logged — the caller must not flip is_embedded on an empty result).
    """
    path = Path(settings.PROCESSED_DATA_PATH) / f"{cik}_{year}_narrative.json"
    if not path.exists():
        logger.error("artifact_missing", cik=cik, year=year, path=str(path))
        return []

    artifact = load_narrative_artifact(path)
    if artifact is None:
        return []
    return chunk_artifact(artifact)