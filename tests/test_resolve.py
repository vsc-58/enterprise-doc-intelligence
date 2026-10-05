"""Resolution tests. No database, no network, no API key."""

import pytest

from src.query.resolve import CorpusIndex, identifying_tokens, normalise, resolve
from src.query.schemas import ResolutionFailure


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("BANK OF AMERICA CORP /DE/", "bank of america corp"),
        ("QUALCOMM INC/DE", "qualcomm inc"),
        ("salesforce.com, inc.", "salesforce com inc"),
        ("JPMORGAN CHASE & CO", "jpmorgan chase co"),
    ],
)
def test_normalise_strips_state_suffix_and_punctuation(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


def test_generic_tokens_leave_nothing_identifying() -> None:
    assert identifying_tokens("the bank") == frozenset()
    assert identifying_tokens("the company") == frozenset()


@pytest.mark.parametrize(
    "mention, ticker",
    [
        ("Apple", "AAPL"), ("apple inc", "AAPL"), ("AAPL", "AAPL"),
        ("Microsoft", "MSFT"), ("Amazon", "AMZN"), ("amazon.com", "AMZN"),
        ("Google", "GOOGL"), ("Alphabet", "GOOGL"), ("Facebook", "META"),
        ("Meta", "META"), ("BofA", "BAC"), ("Bank of America", "BAC"),
        ("JPMorgan", "JPM"), ("Chase", "JPM"), ("Goldman Sachs", "GS"),
        ("Target", "TGT"), ("Salesforce", "CRM"), ("Tesla", "TSLA"),
    ],
)
def test_known_mentions_resolve(mention: str, ticker: str, index: CorpusIndex) -> None:
    resolution = resolve(mention, index=index)
    assert resolution.entity is not None, resolution.detail
    assert resolution.entity.ticker == ticker


def test_resolution_carries_the_held_year_and_period_end(index: CorpusIndex) -> None:
    entity = resolve("Apple", index=index).entity
    assert entity is not None
    assert (entity.filing_year, entity.fiscal_year_end) == (2021, "2021-09-25")


def test_target_period_end_is_a_year_ahead_of_its_key(index: CorpusIndex) -> None:
    """TGT's filing_year 2021 ends 2021-01-30 — the user's 'fiscal 2020' (D1)."""
    entity = resolve("Target", index=index).entity
    assert entity is not None
    assert entity.filing_year == 2021
    assert entity.fiscal_year_end == "2021-01-30"


def test_unheld_company_is_reported_as_such(index: CorpusIndex) -> None:
    resolution = resolve("Nvidia", index=index)
    assert resolution.failure is ResolutionFailure.NON_CORPUS_COMPANY


def test_vague_mention_is_ambiguous_not_a_guess(index: CorpusIndex) -> None:
    resolution = resolve("the bank", index=index)
    assert resolution.failure is ResolutionFailure.AMBIGUOUS_COMPANY


def test_missing_mention_is_distinguished_from_ambiguity(index: CorpusIndex) -> None:
    assert resolve(None, index=index).failure is ResolutionFailure.NO_COMPANY_MENTION


def test_unheld_year_names_the_year_that_is_held(index: CorpusIndex) -> None:
    resolution = resolve("Tesla", requested_year=2023, index=index)
    assert resolution.failure is ResolutionFailure.YEAR_NOT_HELD
    assert "2022" in (resolution.detail or "")


def test_held_year_resolves(index: CorpusIndex) -> None:
    assert resolve("Tesla", requested_year=2022, index=index).entity is not None


def test_unembedded_company_refused_only_for_narrative(index: CorpusIndex) -> None:
    assert resolve("Qualcomm", index=index).entity is not None
    narrative = resolve("Qualcomm", index=index, require_embedded=True)
    assert narrative.failure is ResolutionFailure.COMPANY_NOT_EMBEDDED