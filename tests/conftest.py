"""Shared pytest fixtures. anyio backend pinned to asyncio; no pytest-asyncio needed."""

import pytest

from src.query.resolve import CorpusEntry, CorpusIndex, build_index


@pytest.fixture
def anyio_backend() -> str:
    """Run anyio-marked async tests on asyncio only."""
    return "asyncio"


@pytest.fixture
def index() -> CorpusIndex:
    """
    A corpus index over hand-written entries.

    Registrant names are the real ones, including the forms that motivated the
    normalisation rules: the slash-suffixed state, the spaced "AMAZON COM INC",
    and the lowercase "salesforce.com, inc.".
    """
    entries = [
        CorpusEntry(ticker="AAPL", cik="0000320193", company_name="Apple Inc.",
                    document_id=1, filing_year=2021, fiscal_year_end="2021-09-25"),
        CorpusEntry(ticker="MSFT", cik="0000789019", company_name="MICROSOFT CORPORATION",
                    document_id=2, filing_year=2022, fiscal_year_end="2022-06-30"),
        CorpusEntry(ticker="AMZN", cik="0001018724", company_name="AMAZON COM INC",
                    document_id=3, filing_year=2023, fiscal_year_end="2023-12-31"),
        CorpusEntry(ticker="GOOGL", cik="0001652044", company_name="Alphabet Inc.",
                    document_id=4, filing_year=2021, fiscal_year_end="2021-12-31"),
        CorpusEntry(ticker="META", cik="0001326801", company_name="Meta Platforms, Inc.",
                    document_id=5, filing_year=2022, fiscal_year_end="2022-12-31"),
        CorpusEntry(ticker="JPM", cik="0000019617", company_name="JPMORGAN CHASE & CO",
                    document_id=6, filing_year=2023, fiscal_year_end="2023-12-31"),
        CorpusEntry(ticker="GS", cik="0000886982", company_name="GOLDMAN SACHS GROUP INC",
                    document_id=7, filing_year=2021, fiscal_year_end="2021-12-31"),
        CorpusEntry(ticker="BAC", cik="0000070858", company_name="BANK OF AMERICA CORP /DE/",
                    document_id=8, filing_year=2022, fiscal_year_end="2022-12-31"),
        CorpusEntry(ticker="TGT", cik="0000027419", company_name="TARGET CORP",
                    document_id=9, filing_year=2021, fiscal_year_end="2021-01-30"),
        CorpusEntry(ticker="CRM", cik="0001108524", company_name="salesforce.com, inc.",
                    document_id=10, filing_year=2021, fiscal_year_end="2022-01-31"),
        CorpusEntry(ticker="TSLA", cik="0001318605", company_name="Tesla, Inc.",
                    document_id=11, filing_year=2022, fiscal_year_end="2022-12-31"),
        CorpusEntry(ticker="QCOM", cik="0000804328", company_name="QUALCOMM INC/DE",
                    document_id=12, filing_year=2021, fiscal_year_end="2021-09-26",
                    is_embedded=False),
    ]
    return build_index(entries)