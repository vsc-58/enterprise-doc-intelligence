"""Retriever tests: section policy, numbering, corpus-wide fan-out. No API, no store."""

import pytest

from src.query.intents import SECTION_FALLBACKS, fallback_for, sections_for
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
    """A retrieved chunk whose text has padding spaces, which format_context must strip."""
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
        """Record the search; JPM holds nothing, every other company returns k chunks."""
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
    """Route the retriever's embedding and search calls to a FakeStore."""
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
    # JPM's market-risk sections are absent, so its fallback is tried once with
    # the same vector; this fake holds nothing for JPM at all.
    assert [t for t, _ in store.searches] == ["AAPL", "JPM", "JPM", "META"]
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

# --- Section fallback (market risk -> Item 1A) -------------------------------

XOM = CompanyRef(ticker="XOM", company_name="EXXON MOBIL CORP", fiscal_year=2022)


class SectionStore:
    """Answers OK only where a ticker actually holds one of the filtered sections."""

    HELD = {
        "AAPL": {"Item 1", "Item 1A", "Item 7", "Item 7A"},
        "JPM": {"Item 1", "Item 1A"},
        "XOM": {"Item 1"},
    }

    def __init__(self) -> None:
        self.searches: list[tuple[str | None, frozenset[str]]] = []

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]

    def search(self, vector, k, ticker=None, sections=None) -> RetrievalResult:
        """Record the search; return chunks only for sections the company holds (HELD)."""
        wanted = frozenset(sections or ())
        self.searches.append((ticker, wanted))
        held = sorted(wanted & self.HELD.get(ticker, set()))
        if not held:
            return RetrievalResult(status=RetrievalStatus.NO_MATCHING_CHUNKS, detail="none")
        return RetrievalResult(
            status=RetrievalStatus.OK,
            chunks=[_chunk(ticker, held[0], i, 0.3 + i / 10) for i in range(k)],
        )


@pytest.fixture
def sections_store(monkeypatch: pytest.MonkeyPatch) -> SectionStore:
    """Route the retriever's embedding and search calls to a SectionStore."""
    fake = SectionStore()
    monkeypatch.setattr(retriever, "embed_query", fake.embed)
    monkeypatch.setattr(retriever, "query_by_vector", fake.search)
    return fake


def test_only_market_risk_has_a_fallback() -> None:
    assert set(SECTION_FALLBACKS) == {Intent.MARKET_RISK}
    assert fallback_for(Intent.MARKET_RISK, "JPM").sections == {"Item 1A"}
    assert fallback_for(Intent.MANAGEMENT_COMMENTARY, "JPM") is None
    assert fallback_for(Intent.RISK_FACTORS, "JPM") is None


def test_fallback_is_searched_when_the_intent_sections_are_absent(
    sections_store: SectionStore,
) -> None:
    context = retriever.retrieve_for_company("q", Intent.MARKET_RISK, JPM, k=2)
    assert sections_store.searches == [
        ("JPM", frozenset({"Item 7", "Item 7A"})),
        ("JPM", frozenset({"Item 1A"})),
    ]
    assert context.status is RetrievalStatus.OK
    assert context.sections_searched == {"Item 1A"}
    assert context.fallback_from == {"Item 7", "Item 7A"}
    assert {b.chunk.section for b in context.blocks} == {"Item 1A"}


def test_no_fallback_when_the_intent_sections_are_held(sections_store: SectionStore) -> None:
    context = retriever.retrieve_for_company("q", Intent.MARKET_RISK, APPLE, k=2)
    assert len(sections_store.searches) == 1
    assert context.fallback_from == frozenset()


def test_intent_without_a_fallback_still_refuses(sections_store: SectionStore) -> None:
    context = retriever.retrieve_for_company("q", Intent.MANAGEMENT_COMMENTARY, JPM, k=2)
    assert sections_store.searches == [("JPM", frozenset({"Item 7"}))]
    assert context.status is RetrievalStatus.NO_MATCHING_CHUNKS
    assert context.fallback_from == frozenset()


def test_absent_fallback_reports_the_original_absence(sections_store: SectionStore) -> None:
    context = retriever.retrieve_for_company("q", Intent.MARKET_RISK, XOM, k=2)
    assert len(sections_store.searches) == 2
    assert context.status is RetrievalStatus.NO_MATCHING_CHUNKS
    assert context.sections_searched == {"Item 7", "Item 7A"}
    assert context.fallback_from == frozenset()


def test_corpus_wide_applies_the_fallback_per_company(sections_store: SectionStore) -> None:
    contexts = retriever.retrieve_corpus_wide("q", Intent.MARKET_RISK, [APPLE, JPM], k=2)
    assert [c.fallback_from for c in contexts] == [frozenset(), frozenset({"Item 7", "Item 7A"})]
    assert all(c.status is RetrievalStatus.OK for c in contexts)
