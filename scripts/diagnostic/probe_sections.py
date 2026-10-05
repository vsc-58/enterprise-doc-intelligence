"""
scripts/probe_sections.py — DIAGNOSTIC (print() allowed, per Phase 2 convention)

Measures edgartools item-boundary health across the full corpus before any
Phase 4 chunking decision is made.

Per document, per item: char count, token count, head/tail boundary text,
containment in the saved raw text, and pairwise overlap between items.

Read-only. No API calls, no writes to db/ or data/eval/. Output JSON lands in
data/processed/ (gitignored).

Dependencies: src.storage.metadata_store, src.ingestion.acquire,
src.utils.config, src.utils.logger, tiktoken, edgartools
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tiktoken
from sqlalchemy import select

from src.ingestion.acquire import ensure_identity, select_10k_for_fiscal_year
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

CANDIDATE_ITEMS: list[str] = [
    "Item 1", "Item 1A", "Item 1B", "Item 1C",
    "Item 2", "Item 3", "Item 4",
    "Item 5", "Item 6", "Item 7", "Item 7A", "Item 8",
    "Item 9", "Item 9A", "Item 9B",
    "Item 10", "Item 11", "Item 12", "Item 13", "Item 14",
    "Item 15", "Item 16",
]

# Items under consideration for the RAG corpus. Overlap is measured within
# this set only — over-capture between items we will never chunk is harmless.
NARRATIVE_CANDIDATES: list[str] = [
    "Item 1", "Item 1A", "Item 2", "Item 3",
    "Item 5", "Item 7", "Item 7A", "Item 8",
]

BOUNDARY_CHARS = 300
OVERLAP_PROBE_CHARS = 400
OUTPUT_PATH = Path("data/processed/section_probe.json")

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the cl100k token count for a string.

    Args:
        text: Text to measure.

    Returns:
        Token count. Empty string returns 0.
    """
    if not text:
        return 0
    return len(_ENCODER.encode(text, disallowed_special=()))


def normalize_for_containment(text: str) -> str:
    """Collapse all whitespace so containment checks survive re-flowing.

    edgartools' item view line-breaks table cells differently from the saved
    filing text (Phase 3 D14: same slice, ~58% higher token count). A raw
    substring test would therefore report false negatives.

    Args:
        text: Text to normalise.

    Returns:
        The text with all whitespace removed.
    """
    return "".join(text.split())


def probe_item(tenk: Any, item_label: str) -> dict[str, Any]:
    """Extract one item from a TenK object and measure it.

    Args:
        tenk: edgartools form-specific object exposing __getitem__ by item label.
        item_label: Item label, e.g. "Item 1A".

    Returns:
        A dict with presence, size, boundary text, and any access error. Never
        raises — an inaccessible item is recorded as absent with its error.
    """
    try:
        text = tenk[item_label]
    except Exception as exc:
        return {
            "present": False,
            "error": f"{type(exc).__name__}: {exc}",
            "chars": 0,
            "tokens": 0,
        }

    if not text or not str(text).strip():
        return {"present": False, "error": None, "chars": 0, "tokens": 0}

    text = str(text)
    return {
        "present": True,
        "error": None,
        "chars": len(text),
        "tokens": count_tokens(text),
        "head": text[:BOUNDARY_CHARS],
        "tail": text[-BOUNDARY_CHARS:],
        "_text": text,
    }


def measure_overlap(
    items: dict[str, dict[str, Any]], labels: list[str]
) -> list[dict[str, Any]]:
    """Detect text shared between item slices.

    Over-capture (Phase 3 D8/D14) means one item's slice can swallow another's.
    Chunking both would embed the same passage twice under two section labels.
    Probes whether a sample drawn from the middle of one slice appears inside
    another.

    Args:
        items: Per-item probe results, keyed by item label.
        labels: Item labels to compare pairwise.

    Returns:
        A list of detected overlaps, each naming the contained and containing
        item. Empty if none found.
    """
    findings: list[dict[str, Any]] = []
    present = [
        (label, normalize_for_containment(items[label]["_text"]))
        for label in labels
        if items.get(label, {}).get("present")
    ]

    for inner_label, inner in present:
        if len(inner) < OVERLAP_PROBE_CHARS * 2:
            continue
        mid = len(inner) // 2
        sample = inner[mid : mid + OVERLAP_PROBE_CHARS]
        for outer_label, outer in present:
            if outer_label == inner_label or len(outer) <= len(inner):
                continue
            if sample in outer:
                findings.append(
                    {
                        "contained": inner_label,
                        "contained_in": outer_label,
                        "contained_chars": len(inner),
                        "container_chars": len(outer),
                    }
                )
    return findings


def probe_document(doc: Document) -> dict[str, Any] | None:
    """Probe every candidate item for one document.

    Args:
        doc: Document row carrying ticker, filing_year, cik, local_path.

    Returns:
        A per-document result dict, or None if the filing or its saved text
        could not be resolved (logged, never raised — one failure must not
        stop the corpus sweep).
    """
    try:
        raw_path = Path(doc.local_path)
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error(
            "raw_text_unreadable",
            ticker=doc.ticker,
            year=doc.filing_year,
            path=doc.local_path,
            error=str(exc),
        )
        return None

    try:
        filing = select_10k_for_fiscal_year(doc.ticker, doc.filing_year)
        if filing is None:
            logger.error(
                "filing_not_resolved", ticker=doc.ticker, year=doc.filing_year
            )
            return None
        tenk = filing.obj()
    except Exception as exc:
        logger.error(
            "filing_object_failed",
            ticker=doc.ticker,
            year=doc.filing_year,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None

    raw_norm = normalize_for_containment(raw_text)
    raw_tokens = count_tokens(raw_text)

    items: dict[str, dict[str, Any]] = {}
    for label in CANDIDATE_ITEMS:
        result = probe_item(tenk, label)
        if result["present"]:
            head_norm = normalize_for_containment(result["head"])[:200]
            result["head_in_raw"] = bool(head_norm) and head_norm in raw_norm
            result["share_of_filing"] = round(
                len(normalize_for_containment(result["_text"])) / max(len(raw_norm), 1),
                4,
            )
            result["collapsed"] = result["chars"] < settings.ITEM8_MIN_CHARS
        items[label] = result

    overlaps = measure_overlap(items, NARRATIVE_CANDIDATES)

    for result in items.values():
        result.pop("_text", None)

    return {
        "ticker": doc.ticker,
        "cik": doc.cik,
        "filing_year": doc.filing_year,
        "company_name": doc.company_name,
        "raw_chars": len(raw_text),
        "raw_tokens": raw_tokens,
        "items": items,
        "overlaps": overlaps,
    }


def print_document_summary(result: dict[str, Any]) -> None:
    """Print a compact per-document table of present items."""
    print(f"\n{'=' * 78}")
    print(
        f"{result['ticker']} {result['filing_year']}  "
        f"({result['raw_tokens']:,} tokens raw)"
    )
    print(f"{'item':<10}{'chars':>10}{'tokens':>10}{'share':>8}{'head_ok':>9}  flag")
    for label, item in result["items"].items():
        if not item["present"]:
            continue
        flag = "COLLAPSED" if item["collapsed"] else ""
        if item["share_of_filing"] > 0.80:
            flag = (flag + " OVERSIZE").strip()
        if not item["head_in_raw"]:
            flag = (flag + " HEAD_NOT_IN_RAW").strip()
        print(
            f"{label:<10}{item['chars']:>10,}{item['tokens']:>10,}"
            f"{item['share_of_filing']:>8.2f}{str(item['head_in_raw']):>9}  {flag}"
        )
    for overlap in result["overlaps"]:
        print(
            f"  OVERLAP: {overlap['contained']} is contained within "
            f"{overlap['contained_in']}"
        )


def print_corpus_summary(results: list[dict[str, Any]]) -> None:
    """Print per-item health across the whole corpus."""
    print(f"\n{'=' * 78}\nCORPUS SUMMARY ({len(results)} documents)")
    print(f"{'item':<10}{'present':>9}{'collapsed':>11}{'median_tok':>12}")
    for label in CANDIDATE_ITEMS:
        present = [r["items"][label] for r in results if r["items"][label]["present"]]
        if not present:
            continue
        collapsed = sum(1 for item in present if item["collapsed"])
        tokens = sorted(item["tokens"] for item in present)
        median = tokens[len(tokens) // 2]
        print(f"{label:<10}{len(present):>9}{collapsed:>11}{median:>12,}")

    overlap_docs = [r["ticker"] for r in results if r["overlaps"]]
    print(f"\ndocuments with item overlap: {len(overlap_docs)} {overlap_docs}")


def main() -> None:
    """Probe every acquired document and write results to data/processed/."""
    ensure_identity()

    with get_session() as session:
        documents = list(session.scalars(select(Document).order_by(Document.ticker)))

    logger.info("probe_started", document_count=len(documents))

    results: list[dict[str, Any]] = []
    for index, doc in enumerate(documents, start=1):
        result = probe_document(doc)
        if result is None:
            continue
        results.append(result)
        print_document_summary(result)
        if index % 5 == 0:
            logger.info("probe_progress", processed=index, total=len(documents))

    print_corpus_summary(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(
        "probe_complete", documents=len(results), output_path=str(OUTPUT_PATH)
    )


if __name__ == "__main__":
    main()