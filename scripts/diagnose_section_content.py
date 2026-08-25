"""
scripts/diagnose_sections_content.py — throwaway.

Question: for the filings where edgartools' Item 8 collapsed, is the financial
statement content present in the saved raw text at all?

If YES  -> regex/fallback are viable; the bug is boundary attribution.
If NO   -> those documents are unextractable; both regex and fallback are dead,
           and the eval set must exclude them.

Run: python -m scripts.diagnose_sections_content

# DIAGNOSTIC — throwaway investigation script, not project code.
# Uses print() deliberately for readable tabular output; project code uses logger.
"""

import re
from pathlib import Path

from sqlalchemy import select

from src.storage.metadata_store import Document, get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

BROKEN = {"XOM", "NFLX", "QCOM", "JPM", "INTC"}
CONTROL = {"AAPL", "MSFT"}  # known-good, for comparison

# Anchors that must be present if the statements are really in the text.
ANCHORS = {
    "total_assets_line": r"total\s+assets",
    "operating_cf_line": r"net\s+cash\s+(provided\s+by|from).{0,30}operating",
    "auditor_report": r"report\s+of\s+independent\s+registered\s+public\s+accounting",
    "balance_sheet_hdr": r"consolidated\s+balance\s+sheet",
    "income_stmt_hdr": r"consolidated\s+statements?\s+of\s+(operations|income)",
}


def main() -> None:
    """Report anchor presence and 'Item 8' occurrences in each raw filing."""
    with get_session() as session:
        docs = session.execute(select(Document)).scalars().all()
        rows = [(d.ticker, d.local_path) for d in docs]

    for ticker, local_path in sorted(rows):
        if ticker not in BROKEN | CONTROL:
            continue

        text = Path(local_path).read_text(encoding="utf-8")
        low = text.lower()

        hits = {
            name: len(re.findall(pat, low))
            for name, pat in ANCHORS.items()
        }

        # Where does "Item 8" appear? (first hit is usually the TOC.)
        item8_positions = [
            round(m.start() / len(text), 3)
            for m in re.finditer(r"item\s*8[\.\s\u2014-]", low)
        ]

        logger.info(
            "raw_text_probe",
            ticker=ticker,
            status="BROKEN" if ticker in BROKEN else "control",
            chars=len(text),
            **hits,
            item8_hits=len(item8_positions),
            item8_rel_positions=item8_positions[:6],
        )


if __name__ == "__main__":
    main()