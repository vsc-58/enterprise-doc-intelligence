"""
scripts/build_sections.py
Build section artifacts for the acquired corpus.

For each selected document: read its Document row (for the on-disk text path
and CIK), load the raw filing text, resolve the filing object (for Item 8), run
get_sections, and write data/raw/{cik}_{year}_sections.json.

Targets come from SQLite, not a hardcoded list — the acquired Document row is
the source of truth for what exists (Phase 1 D6). --eval-only / --held-out
filter that set using src.eval.eval_set; they do not define it.

Two measurements are emitted per document because Phase 3 depends on them:
  * assembled input tokens vs MODEL_INPUT_TOKEN_BUDGET — a document over budget
    cannot be extracted at all, and must be caught before any API spend.
  * unchanged vs changed against an existing artifact — the S3 response cache is
    keyed on input text, so an unchanged artifact means a cache hit (no cost)
    and a changed one means the document will be re-billed.

Run: python -m scripts.build_sections [--eval-only | --held-out]
                                      [--tickers AAPL,MSFT]

Dependencies: src.ingestion.acquire, src.ingestion.sections,
src.storage.metadata_store, src.eval.eval_set, src.utils.config,
src.utils.logger.
"""

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from src.eval.eval_set import is_eval_document
from src.ingestion.acquire import ensure_identity, select_10k_for_fiscal_year
from src.ingestion.sections import Section, get_sections
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# (ticker, year, cik, local_path) — plain values, not ORM instances.
DocumentTarget = tuple[str, int, str, str]


def _load_targets(
    scope: str, tickers: list[str] | None
) -> list[DocumentTarget]:
    """
    Read the corpus from SQLite and apply the requested filters.

    Plain tuples are returned rather than Document instances so no attribute is
    accessed after the session closes (a committed session expires its objects
    by default, which would raise DetachedInstanceError at use time).

    Args:
        scope: "all", "eval", or "held_out". Filters on the Phase 2 selection
            set defined in src.eval.eval_set.
        tickers: optional explicit ticker allowlist, applied after scope.

    Returns:
        (ticker, year, cik, local_path) for each matching document, ordered by
        ticker.

    Raises:
        ValueError: if scope is not one of the three accepted values.
    """
    if scope not in {"all", "eval", "held_out"}:
        raise ValueError(f"unknown scope: {scope}")

    with get_session() as session:
        rows = session.execute(
            select(
                Document.ticker,
                Document.filing_year,
                Document.cik,
                Document.local_path,
            ).order_by(Document.ticker)
        ).all()

    targets: list[DocumentTarget] = []
    for ticker, year, cik, local_path in rows:
        in_eval = is_eval_document(ticker, year)
        if scope == "eval" and not in_eval:
            continue
        if scope == "held_out" and in_eval:
            continue
        if tickers is not None and ticker not in tickers:
            continue
        targets.append((ticker, year, cik, local_path))
    return targets


def _build_payload(
    cik: str, year: int, sections: dict[str, Section]
) -> dict:
    """
    Assemble the JSON-serialisable artifact for one document.

    Args:
        cik: canonical 10-digit CIK from the Document row.
        year: fiscal-period-end year.
        sections: the get_sections output.

    Returns:
        The artifact dict, identical in shape to the Phase 2 artifacts.
    """
    return {
        "cik": cik,
        "year": year,
        "sections": {name: sec.model_dump() for name, sec in sections.items()},
    }


def _write_sections(cik: str, year: int, payload: dict) -> tuple[Path, str]:
    """
    Write the artifact, reporting whether it differs from an existing one.

    The comparison is on the serialised bytes because that is what the S3 cache
    key hashes: identical bytes mean the cached response is still valid and the
    document costs nothing to re-extract.

    Args:
        cik: canonical 10-digit CIK.
        year: fiscal-period-end year.
        payload: the artifact dict from _build_payload.

    Returns:
        (path written, change status) where status is "new", "unchanged", or
        "changed".

    Raises:
        OSError: if the file cannot be read or written.
    """
    out_path = Path(settings.RAW_DATA_PATH) / f"{cik}_{year}_sections.json"
    serialised = json.dumps(payload, indent=2)

    if not out_path.exists():
        status = "new"
    elif out_path.read_text(encoding="utf-8") == serialised:
        status = "unchanged"
    else:
        status = "changed"

    out_path.write_text(serialised, encoding="utf-8")
    return out_path, status


def main() -> None:
    """
    Build and persist section artifacts for the selected documents.

    Emits a per-document log line and a batch summary carrying the provenance
    distribution, the over-budget list, and the cache-impact counts.
    """
    parser = argparse.ArgumentParser(description="Build 10-K section artifacts.")
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--eval-only", action="store_true", help="Phase 2 selection set only."
    )
    scope_group.add_argument(
        "--held-out", action="store_true", help="Everything outside the eval set."
    )
    parser.add_argument(
        "--tickers", type=str, default=None, help="Comma-separated ticker filter."
    )
    args = parser.parse_args()

    scope = "eval" if args.eval_only else "held_out" if args.held_out else "all"
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",")]
        if args.tickers
        else None
    )

    ensure_identity()
    targets = _load_targets(scope, tickers)
    logger.info("build_sections_start", scope=scope, documents=len(targets))

    provenance_counts: dict[str, int] = {}
    change_counts: dict[str, int] = {}
    over_budget: list[str] = []
    written, failed = 0, 0

    for ticker, year, cik, local_path in targets:
        try:
            raw_text = Path(local_path).read_text(encoding="utf-8")
            filing = select_10k_for_fiscal_year(ticker, year)
            sections = get_sections(filing, raw_text)

            fs = sections["financial_statements"]
            cover = sections["cover"]
            input_tokens = fs.token_count + cover.token_count
            fits = input_tokens <= settings.MODEL_INPUT_TOKEN_BUDGET
            if not fits:
                over_budget.append(f"{ticker}_{year}")

            payload = _build_payload(cik, year, sections)
            out_path, change_status = _write_sections(cik, year, payload)

            provenance_counts[fs.section_source] = (
                provenance_counts.get(fs.section_source, 0) + 1
            )
            change_counts[change_status] = change_counts.get(change_status, 0) + 1
            written += 1

            log = logger.info if fits else logger.warning
            log(
                "sections_built",
                ticker=ticker,
                year=year,
                cik=cik,
                fs_source=fs.section_source,
                fs_tokens=fs.token_count,
                cover_tokens=cover.token_count,
                input_tokens=input_tokens,
                token_budget=settings.MODEL_INPUT_TOKEN_BUDGET,
                fits_budget=fits,
                artifact=change_status,
                in_eval_set=is_eval_document(ticker, year),
                path=str(out_path),
            )
        except Exception as exc:  # per-item: one failure never stops the batch
            failed += 1
            logger.error(
                "sections_build_failed",
                ticker=ticker,
                year=year,
                cik=cik,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    logger.info(
        "build_sections_summary",
        scope=scope,
        written=written,
        failed=failed,
        provenance=provenance_counts,
        artifacts=change_counts,
        over_budget=over_budget or None,
    )
    if over_budget:
        logger.warning(
            "documents_exceed_token_budget",
            documents=over_budget,
            consequence="extraction will fail on context length; resolve before Phase 3 spend",
        )


if __name__ == "__main__":
    main()