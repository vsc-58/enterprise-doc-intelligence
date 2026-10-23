"""Check GET /export against the database, read independently with plain SQL.

Diagnostic (print() allowed). Checks GET /export against the database, read
INDEPENDENTLY: plain SQL through Python's built-in sqlite3, not the ORM or
latest_successful_records() the export itself uses. Comparing the export with
its own code path would prove nothing.

What it checks:
  - the file: UTF-8 byte-order mark, 35 columns, one row per document;
  - every row x every figure: the CSV value equals the stored value exactly
    (both empty when the figure was not extracted);
  - fiscal_year_end and the cited source line for each figure;
  - and it prints every non-verified quality flag, to compare by eye with the
    expected ones (BAC's four scale_ambiguous fields, GS operating cash flow).

The database is opened READ-ONLY (mode=ro): this script cannot change it.

Usage (API running in another window):
    python -m scripts.diagnostic.verify_export
    python -m scripts.diagnostic.verify_export --url http://localhost:8000

Exit code 0 when everything matches, 1 otherwise.
"""

import argparse
import io
import sqlite3
import sys
from pathlib import Path

import httpx
import pandas as pd

from src.utils.config import settings

METRICS = ["total_revenue", "net_income", "total_assets", "total_liabilities", "operating_cash_flow"]
EXPECTED_COLUMNS = 35
UTF8_BOM = b"\xef\xbb\xbf"

# The same rule as the export's canonical read: per document, the latest successful
# extraction by time, ties broken by id. Written out by hand on purpose.
LATEST_SQL = f"""
SELECT d.ticker, r.fiscal_year_end,
       {", ".join(f"r.{m}, r.{m}_source_line" for m in METRICS)}
FROM documents d
JOIN extracted_records r ON r.document_id = d.id
WHERE r.id = (
    SELECT r2.id FROM extracted_records r2
    WHERE r2.document_id = d.id AND r2.extraction_status = 'success'
    ORDER BY r2.extracted_at DESC, r2.id DESC
    LIMIT 1)
ORDER BY d.ticker
"""


def show(value: object) -> str:
    """A value as plain Python would print it (no numpy wrapper)."""
    item = getattr(value, "item", None)
    return repr(item() if callable(item) else value)


def same(a: object, b: object) -> bool:
    """Equal, treating two missing values as equal."""
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def main() -> int:
    """Fetch the CSV, compare every value with SQLite, and print what differs.

    Returns:
        0 when everything matches, 1 otherwise (the process exit code).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    base_url = parser.parse_args().url.rstrip("/")
    problems: list[str] = []

    try:
        response = httpx.get(f"{base_url}/export", timeout=30)
    except httpx.ConnectError:
        print(f"Cannot reach {base_url}. Is the API running? "
              "Start it in another window: uvicorn src.api.main:app --reload")
        return 1
    if response.status_code != 200:
        print(f"GET /export returned {response.status_code}: {response.text[:200]}")
        return 1
    if not response.content.startswith(UTF8_BOM):
        problems.append("file does not start with a UTF-8 byte-order mark")
    csv = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig",
                      dtype={"cik": str}).set_index("ticker")
    if len(csv.columns) + 1 != EXPECTED_COLUMNS:                 # +1: ticker is the index
        problems.append(f"{len(csv.columns) + 1} columns, expected {EXPECTED_COLUMNS}")

    db_path = Path(settings.SQLITE_DB_PATH).resolve()
    with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as connection:
        db = pd.read_sql_query(LATEST_SQL, connection).set_index("ticker")

    print(f"export: {len(csv)} rows   database: {len(db)} rows   ({db_path})")
    if set(csv.index) != set(db.index):
        problems.append(f"tickers differ: only in CSV {sorted(set(csv.index) - set(db.index))}, "
                        f"only in DB {sorted(set(db.index) - set(csv.index))}")

    compared = 0
    for ticker in sorted(set(csv.index) & set(db.index)):
        for column in ["fiscal_year_end"] + [c for m in METRICS for c in (m, f"{m}_source_line")]:
            compared += 1
            if not same(csv.at[ticker, column], db.at[ticker, column]):
                problems.append(f"{ticker}.{column}: CSV {show(csv.at[ticker, column])} "
                                f"!= DB {show(db.at[ticker, column])}")
    print(f"compared {compared} values")

    print("\nnon-verified quality flags in the export:")
    for ticker in sorted(csv.index):
        for metric in METRICS:
            verdict = csv.at[ticker, f"{metric}_quality"]
            if verdict != "verified":
                print(f"  {ticker:5} {metric:20} {verdict:10} {csv.at[ticker, f'{metric}_quality_reason']}")

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nPASS — every figure, period end and cited line matches the database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
