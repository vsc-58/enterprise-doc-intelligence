"""
scripts/diagnose_phase2.py — throwaway diagnostic, not project code.

Answers three questions before Phase 2 code is written:
  1. Do full filings fit in the model context, and what do sections cost?
  2. Does edgartools section detection work on this pinned version, for all 20?
  3. Which documents have complete 5-field ground truth?

Run: python -m scripts.diagnose_phase2

# DIAGNOSTIC — throwaway investigation script, not project code.
# Uses print() deliberately for readable tabular output; project code uses logger.
"""

import json
from pathlib import Path

import tiktoken
from sqlalchemy import select

from src.ingestion.acquire import ensure_identity, select_10k_for_fiscal_year
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

ENCODING = tiktoken.get_encoding("o200k_base")  # gpt-4o-mini family
ITEM_KEYS = ["Item 1", "Item 1A", "Item 7", "Item 8"]
TRUTH_FIELDS = [
    "total_revenue", "net_income", "total_assets",
    "total_liabilities", "operating_cash_flow",
]


def n_tokens(text: str | None) -> int:
    """Count tokens in text, 0 if None/empty."""
    return len(ENCODING.encode(text)) if text else 0


def load_ground_truth() -> dict:
    """Load the Phase 1 answer key."""
    path = Path(settings.EVAL_DATA_PATH) / "ground_truth.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    """Report token budgets, section availability, and truth completeness."""
    ensure_identity()
    truth = load_ground_truth()

    with get_session() as session:
        docs = session.execute(select(Document)).scalars().all()
        rows = [
            (d.id, d.ticker, d.cik, d.filing_year, d.local_path) for d in docs
        ]

    complete, incomplete = [], []

    for doc_id, ticker, cik, year, local_path in rows:
        key = f"{cik}_{year}"

        # --- ground-truth completeness ---
        record = truth.get(key, {})
        nulls = [f for f in TRUTH_FIELDS if record.get(f) is None]
        (incomplete if nulls else complete).append(ticker)

        # --- full-text token count (from the saved raw file) ---
        full_tokens = n_tokens(Path(local_path).read_text(encoding="utf-8"))

        # --- section availability + token counts ---
        try:
            filing = select_10k_for_fiscal_year(ticker, year)
            tenk = filing.obj()
            detected = list(tenk.items) if tenk.items else []
            sections_empty = not bool(tenk.sections)
            per_item = {k: n_tokens(tenk[k]) for k in ITEM_KEYS}
        except Exception as exc:  # diagnostic: never crash the sweep
            logger.error(
                "section_probe_failed",
                ticker=ticker, year=year,
                error_type=type(exc).__name__, error=str(exc),
            )
            continue

        missing_items = [k for k in ITEM_KEYS if per_item[k] == 0]

        logger.info(
            "doc_probe",
            doc_id=doc_id, ticker=ticker, cik=cik, year=year,
            full_tokens=full_tokens,
            sections_empty=sections_empty,
            n_items_detected=len(detected),
            missing_target_items=missing_items or None,
            item1=per_item["Item 1"],
            item1a=per_item["Item 1A"],
            item7=per_item["Item 7"],
            item8=per_item["Item 8"],
            truth_nulls=nulls or None,
        )

    logger.info("summary_truth_complete", tickers=sorted(complete), n=len(complete))
    logger.info("summary_truth_incomplete", tickers=sorted(incomplete), n=len(incomplete))


if __name__ == "__main__":
    main()