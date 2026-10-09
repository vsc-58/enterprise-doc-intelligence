"""Retriever tests: section policy, numbering, corpus-wide fan-out. No API, no store."""

import pytest

from src.query.intents import sections_for
from src.query.schemas import Intent
from src.rag import retriever
from src.rag.retriever import CompanyRef, format_context
from src.storage.vector_store import (
    RetrievalResult,
    RetrievalStatus,
    RetrievedChunk,
    VectorStoreError,
)

APPLE = CompanyRef(ticker="AAPL", company_name="Apple Inc.", fiscal_year=2021)
META = CompanyRef(ticker="META", company_name="Meta Platforms, Inc.", fiscal_year=2022)
JPM = CompanyRef(ticker="JPM", company_name="JPMORGAN CHASE & CO", fiscal_year=2023)


def _chunk(ticker: str, section: str, index: int, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"  {ticker} {section} text {index}  ", ticker=ticker,
        company_name=ticker, cik="0000000001", fiscal_year=2021,
        section=section, chunk_index=index, distance=distance,
    )


class FakeStore:
    """Records calls; answers OK for every company except JPM."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.searches: list[tuple[str | None, frozenset[str]]] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [0.1, 0.2]

    def search(self, vector, k, ticker=None, sections=None) -> RetrievalResult:
        self.searches.append((ticker, frozenset(sections or ())))
        if ticker == "JPM":
            return RetrievalResult(status=RetrievalStatus.NO_MATCHING_CHUNKS, detail="none")
        section = sorted(sections)[0]
        return RetrievalResult(
            status=RetrievalStatus.OK,
            chunks=[_chunk(ticker, section, i, 0.3 + i / 10) for i in range(k)],
        )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(retriever, "embed_query", fake.embed)
    monkeypatch.setattr(retriever, "query_by_vector", fake.search)
    return fake


def test_aliases_replace_item_1_for_mislabelled_companies() -> None:
    assert sections_for(Intent.COMPANY_OVERVIEW, "META") == {"Item 1A"}
    assert sections_for(Intent.STRATEGY, "QCOM") == {"Item 1A", "Item 7"}
    assert sections_for(Intent.COMPANY_OVERVIEW, "AAPL") == {"Item 1"}
    assert sections_for(Intent.MANAGEMENT_COMMENTARY, "META") == {"Item 7"}


def test_sql_intents_search_no_sections() -> None:
    assert sections_for(Intent.FINANCIAL_METRIC, "AAPL") == frozenset()


def test_single_company_blocks_are_numbered_from_one(store: FakeStore) -> None:
    context = retriever.retrieve_for_company("q", Intent.RISK_FACTORS, APPLE, k=3)
    assert context.status is RetrievalStatus.OK
    assert [b.number for b in context.blocks] == [1, 2, 3]
    assert context.block(2) is context.blocks[1]
    assert context.block(0) is None and context.block(4) is None


def test_single_company_search_uses_aliased_sections(store: FakeStore) -> None:
    context = retriever.retrieve_for_company("q", Intent.COMPANY_OVERVIEW, META, k=2)
    assert store.searches == [("META", frozenset({"Item 1A"}))]
    assert context.sections_searched == {"Item 1A"}


def test_embedding_failure_is_a_failed_context_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(text: str) -> list[float]:
        raise VectorStoreError("Could not embed the query")

    monkeypatch.setattr(retriever, "embed_query", broken)
    context = retriever.retrieve_for_company("q", Intent.RISK_FACTORS, APPLE)
    assert context.status is RetrievalStatus.FAILED
    assert context.blocks == []


def test_corpus_wide_embeds_once_and_searches_each_company(store: FakeStore) -> None:
    contexts = retriever.retrieve_corpus_wide(
        "q", Intent.MARKET_RISK, [APPLE, JPM, META], k=2
    )
    assert store.embed_calls == 1
    assert [t for t, _ in store.searches] == ["AAPL", "JPM", "META"]
    assert [c.company.ticker for c in contexts] == ["AAPL", "JPM", "META"]
    assert contexts[1].status is RetrievalStatus.NO_MATCHING_CHUNKS
    assert all(len(c.blocks) == 2 for c in contexts if c.status is RetrievalStatus.OK)


def test_corpus_wide_blocks_restart_at_one_per_company(store: FakeStore) -> None:
    contexts = retriever.retrieve_corpus_wide("q", Intent.RISK_FACTORS, [APPLE, META], k=2)
    assert [b.number for b in contexts[0].blocks] == [1, 2]
    assert [b.number for b in contexts[1].blocks] == [1, 2]
    assert contexts[0].blocks[0].source_ref() != contexts[1].blocks[0].source_ref()


def test_source_ref_carries_chunk_identity(store: FakeStore) -> None:
    ref = retriever.retrieve_for_company("q", Intent.RISK_FACTORS, APPLE, k=1).blocks[0].source_ref()
    assert (ref.ticker, ref.section, ref.chunk_index) == ("AAPL", "Item 1A", 0)


def test_format_context_heads_each_block_with_number_and_source(store: FakeStore) -> None:
    context = retriever.retrieve_for_company("q", Intent.RISK_FACTORS, APPLE, k=2)
    text = format_context(context)
    assert text.startswith("[1] Apple Inc., FY2021, Item 1A\nAAPL Item 1A text 0")
    assert "\n\n[2] Apple Inc., FY2021, Item 1A\n" in text