"""The fixed strategy-selection set of 10 filings, and the held-out complement.

This constant is an experimental fact, not configuration: it records which
documents S3 was SELECTED on. Everything scored against it is in-sample;
everything else in the corpus is held out. It lives here so build_sections.py,
run_eval.py, and the held-out scorer read one definition and cannot drift.

Dependencies: none (deliberately — importing this must not pull in edgartools,
SQLAlchemy, or the LLM stack).
"""

# 10 documents, chosen before the strategy comparison for sector spread and specific hazards
# (Sep/Jun/Jan fiscal year ends, Revenues vs RevenueFromContract concepts,
# ProfitLoss concept, null liabilities, the single fallback document).
# (ticker, fiscal-period-end year).
EVAL_SET: list[tuple[str, int]] = [
    ("AAPL", 2021),
    ("MSFT", 2022),
    ("GOOGL", 2021),
    ("AMZN", 2023),
    ("WMT", 2023),
    ("JNJ", 2021),
    ("NFLX", 2023),
    ("V", 2022),
    ("ADBE", 2022),
    ("INTC", 2023),
]

EVAL_KEYS: frozenset[tuple[str, int]] = frozenset(EVAL_SET)


def is_eval_document(ticker: str, year: int) -> bool:
    """Report whether a document was part of the strategy-selection set.

    Args:
        ticker: the company ticker as stored on the Document row.
        year: fiscal-period-end year as stored on the Document row.

    Returns:
        True if the document is in-sample (S3 was selected using it),
        False if it is held out.

    Raises:
        Nothing.
    """
    return (ticker, year) in EVAL_KEYS