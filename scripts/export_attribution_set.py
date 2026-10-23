"""Build the attribution judge's validation set: real claims plus planted errors.

First of two attribution steps (then run_attribution_eval). The judge's score
is used only if the judge passes this set, under criteria fixed in advance.

Reads data/eval/attribution_set.json (claims chosen and invented-detail seeds
written before any judge output existed) and:
  1. resolves the two citation-swap seed types against the live store —
       wrong_block_same_company: the claim's 10th-nearest uncited chunk from the
         same company and section (near enough to be on topic, far enough not to
         be the source);
       wrong_company_same_topic: the nearest chunk from a different company in
         the same section —
     and writes the chosen chunk keys to data/eval/attribution_seeds.json;
  2. writes the BLIND labelling sheet for the 40 test claims to
     PROCESSED_DATA_PATH/attribution_label_sheet.md: claim, company, cited text,
     an empty label line. No model label is shown, so human labels are
     independent of any model;
  3. writes PROCESSED_DATA_PATH/attribution_review.json: calibration claims and
     all seeds with their text, for calibration labelling and for checking that
     no seed is accidentally supported.

Embedding calls only (one per citation-swap seed, 20). No LLM.

Run:  python -m scripts.export_attribution_set
"""

import json
import sys
from pathlib import Path
from typing import Any

from src.eval.attribution import split_key
from src.query.resolve import load_index
from src.storage.vector_store import (
    RetrievalStatus,
    RetrievedChunk,
    VectorStoreError,
    embed_query,
    get_chunks_by_key,
    query_by_vector,
)
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

SAME_COMPANY_RANK = 10
SEARCH_DEPTH = 30


def _key(chunk: RetrievedChunk) -> str:
    """Chunk key, TICKER|section|index."""
    return f"{chunk.ticker}|{chunk.section}|{chunk.chunk_index}"


def resolve_seed(seed: dict[str, Any]) -> list[str]:
    """Choose the swapped-in citation for one citation-swap seed.

    Args:
        seed: A seed entry with type, company, claim and the base claim's sources.

    Returns:
        A one-element list with the chosen chunk key.

    Raises:
        VectorStoreError: on an embedding or store failure, or when no candidate
            chunk exists.
    """
    ticker, section, _ = split_key(seed["base_sources"][0])
    vector = embed_query(seed["claim"])
    if seed["type"] == "wrong_block_same_company":
        result = query_by_vector(vector, k=SEARCH_DEPTH, ticker=ticker, sections={section})
        candidates = [_key(c) for c in result.chunks if _key(c) not in seed["base_sources"]]
        if result.status is not RetrievalStatus.OK or not candidates:
            raise VectorStoreError(f"{seed['id']}: no same-company candidate")
        return [candidates[min(SAME_COMPANY_RANK, len(candidates)) - 1]]
    result = query_by_vector(vector, k=SEARCH_DEPTH, sections={section})
    candidates = [_key(c) for c in result.chunks if c.ticker != ticker]
    if result.status is not RetrievalStatus.OK or not candidates:
        raise VectorStoreError(f"{seed['id']}: no other-company candidate")
    return [candidates[0]]


def _block_text(keys: list[str], chunks: dict[tuple[str, str, int], RetrievedChunk]) -> str:
    """Render cited blocks for a human reader."""
    parts = []
    for n, key in enumerate(keys, start=1):
        chunk = chunks.get(split_key(key))
        if chunk is None:
            parts.append(f"[{n}] {key}\n(MISSING FROM STORE)")
        else:
            parts.append(f"[{n}] {chunk.company_name} · FY{chunk.fiscal_year} · {chunk.section}"
                         f"\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def main() -> int:
    """Resolve seeds and write the seed file, the blind sheet and the review file.

    Returns:
        Process exit code: 0 on success, 1 on any failure.
    """
    set_path = Path(settings.EVAL_DATA_PATH) / settings.ATTRIBUTION_SET_FILE
    try:
        spec = json.loads(set_path.read_text(encoding="utf-8"))
        index = load_index()
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        logger.error("attribution_export_setup_failed", error_type=type(exc).__name__, error=str(exc))
        return 1
    names = {e.ticker: display_name(e.company_name, e.ticker) for e in index.entries}
    items = {item["id"]: item for item in spec["items"]}

    resolved: dict[str, list[str]] = {}
    for seed in spec["seeds"]:
        base = items[seed["base"]]
        if seed["type"] == "invented_detail":
            resolved[seed["id"]] = base["sources"]
            continue
        try:
            resolved[seed["id"]] = resolve_seed({**seed, "base_sources": base["sources"]})
        except VectorStoreError as exc:
            logger.error("attribution_seed_failed", seed=seed["id"], error=str(exc))
            return 1

    keys = {split_key(k) for item in spec["items"] for k in item["sources"]}
    keys |= {split_key(k) for ks in resolved.values() for k in ks}
    try:
        chunks = get_chunks_by_key(keys)
    except VectorStoreError as exc:
        logger.error("attribution_export_fetch_failed", error=str(exc))
        return 1
    missing = sorted("|".join(map(str, k)) for k in keys - set(chunks))
    if missing:
        logger.warning("attribution_export_missing_chunks", keys=missing)

    seeds_path = Path(settings.EVAL_DATA_PATH) / settings.ATTRIBUTION_SEEDS_FILE
    seeds_path.write_text(json.dumps({
        "note": "Citation for each seed, resolved by scripts/export_attribution_set.py.",
        "same_company_rank": SAME_COMPANY_RANK,
        "sources": resolved,
    }, indent=1), encoding="utf-8")

    out_dir = Path(settings.PROCESSED_DATA_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Attribution labels — blind test set (40 claims)",
        "",
        "For each claim: do the cited excerpts support EVERY detail of the sentence",
        "(company, numbers, dates, causes, qualifiers)? Paraphrase is fine; anything",
        "missing, more specific than the text, or from a different company is not.",
        "",
        "Write S (supported) or N (not supported) after `Label:`. Use only the excerpts.",
        "",
    ]
    for item in spec["items"]:
        if item["split"] != "test":
            continue
        lines += [
            "---", "", f"## {item['id']}", "",
            f"**About:** {names.get(item['company'], item['company'])}", "",
            f"**Claim:** {item['claim']}", "",
            "**Cited excerpts:**", "", "```text", _block_text(item["sources"], chunks), "```", "",
            "Label: ", "",
        ]
    sheet = out_dir / settings.ATTRIBUTION_SHEET_FILE
    sheet.write_text("\n".join(lines), encoding="utf-8")

    review = {
        "calibration": [{**item, "company_name": names.get(item["company"]),
                         "excerpts": _block_text(item["sources"], chunks)}
                        for item in spec["items"] if item["split"] == "calib"],
        "seeds": [{**seed, "sources": resolved[seed["id"]],
                   "company_name": names.get(seed["company"]),
                   "base_claim": items[seed["base"]]["claim"],
                   "excerpts": _block_text(resolved[seed["id"]], chunks)}
                  for seed in spec["seeds"]],
        "test": [{**item, "excerpts": _block_text(item["sources"], chunks)}
                 for item in spec["items"] if item["split"] == "test"],
    }
    review_path = out_dir / settings.ATTRIBUTION_REVIEW_FILE
    review_path.write_text(json.dumps(review, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("attribution_export_written", seeds=str(seeds_path), sheet=str(sheet),
                review=str(review_path), missing_chunks=len(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
