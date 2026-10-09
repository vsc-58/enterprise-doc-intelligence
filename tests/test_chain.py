"""Chain tests: citation validation, figure guard wiring, refusals, corpus survey.

No API and no store: retrieval, guard data and the LLM call are replaced.
"""

import asyncio

import pytest

from src.query.schemas import FindingStatus, Intent, NarrativeRefusal
from src.rag import chain
from src.rag.chain import (
    AnswerOutcome,
    DraftAnswer,
    DraftClaim,
    DraftVerdict,
    VerdictOutcome,
    clean_claim,
    refusal_message,
)
from src.rag.figure_guard import PLACEHOLDER
from src.rag.retriever import CompanyContext, CompanyRef, ContextBlock
from src.storage.vector_store import RetrievalStatus, RetrievedChunk

APPLE = CompanyRef(ticker="AAPL", company_name="Apple Inc.", fiscal_year=2021)
INTEL = CompanyRef(ticker="INTC", company_name="INTEL CORP", fiscal_year=2022)
GOOGLE = CompanyRef(ticker="GOOGL", company_name="Alphabet Inc.", fiscal_year=2021)
GUARD = {"AAPL": frozenset({365_817_000_000.0})}


def _context(company: CompanyRef, status: RetrievalStatus, n: int = 3) -> CompanyContext:
    blocks = [
        ContextBlock(number=i, chunk=RetrievedChunk(
            text=f"Passage {i}. Total net sales$365,817.", ticker=company.ticker,
            company_name=company.company_name, cik="1", fiscal_year=company.fiscal_year,
            section="Item 7", chunk_index=i, distance=0.4,
        ))
        for i in range(1, n + 1)
    ] if status is RetrievalStatus.OK else []
    return CompanyContext(company=company, sections_searched=frozenset({"Item 7"}),
                          status=status, blocks=blocks)


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace retrieval, guard data and the LLM; record what the LLM saw."""
    state: dict = {"contexts": {}, "drafts": {}, "seen": [], "active": 0, "peak": 0}

    def one(question, intent, company, k=None):
        return state["contexts"][company.ticker]

    def many(question, intent, companies, k=None):
        return [state["contexts"][c.ticker] for c in companies]

    async def invoke(prompt, schema, variables):
        state["seen"].append(variables)
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        draft = state["drafts"][variables["company"]]
        if isinstance(draft, Exception):
            raise draft
        return draft

    monkeypatch.setattr(chain, "retrieve_for_company", one)
    monkeypatch.setattr(chain, "retrieve_corpus_wide", many)
    monkeypatch.setattr(chain, "load_guard_values", lambda: GUARD)
    monkeypatch.setattr(chain, "_invoke", invoke)
    return state


def _answer(state: dict, draft) -> "chain.NarrativeResult":
    state["contexts"]["AAPL"] = _context(APPLE, RetrievalStatus.OK)
    state["drafts"]["Apple Inc."] = draft
    return asyncio.run(chain.answer_for_company("q", Intent.MANAGEMENT_COMMENTARY, APPLE))


def test_section_absent_refuses_without_calling_the_model(wired: dict) -> None:
    wired["contexts"]["INTC"] = _context(INTEL, RetrievalStatus.NO_MATCHING_CHUNKS)
    result = asyncio.run(chain.answer_for_company("q", Intent.MANAGEMENT_COMMENTARY, INTEL))
    assert result.refusal is NarrativeRefusal.SECTION_ABSENT
    assert wired["seen"] == []
    message = refusal_message(result.refusal, "Intel", result.sections_searched)
    assert "MD&A (Item 7)" in message


def test_retrieval_failure_is_failed(wired: dict) -> None:
    wired["contexts"]["AAPL"] = _context(APPLE, RetrievalStatus.FAILED)
    result = asyncio.run(chain.answer_for_company("q", Intent.RISK_FACTORS, APPLE))
    assert result.refusal is NarrativeRefusal.FAILED


def test_model_never_sees_a_stored_figure(wired: dict) -> None:
    result = _answer(wired, DraftAnswer(outcome=AnswerOutcome.NOT_ADDRESSED))
    context_text = wired["seen"][0]["context"]
    assert "365,817" not in context_text and PLACEHOLDER in context_text
    assert result.figures_masked == 3


def test_invalid_citations_are_dropped_and_counted(wired: dict) -> None:
    result = _answer(wired, DraftAnswer(outcome=AnswerOutcome.ANSWERED, claims=[
        DraftClaim(text="Growth came from iPhone.", passages=[1, 9, 1]),
        DraftClaim(text="Uncited by anything real.", passages=[7]),
    ]))
    assert result.refusal is None
    assert [c.text for c in result.claims] == ["Growth came from iPhone."]
    assert [s.chunk_index for s in result.claims[0].sources] == [1]
    assert result.invalid_citations == 2


def test_figure_in_a_claim_is_redacted_not_dropped(wired: dict) -> None:
    result = _answer(wired, DraftAnswer(outcome=AnswerOutcome.ANSWERED, claims=[
        DraftClaim(text="Net sales rose 33% to $365.8 billion, driven by iPhone.", passages=[2]),
    ]))
    assert result.claims[0].text == f"Net sales rose 33% to {PLACEHOLDER}, driven by iPhone."
    assert result.figures_redacted == 1


def test_no_surviving_claim_is_uncited(wired: dict) -> None:
    result = _answer(wired, DraftAnswer(outcome=AnswerOutcome.ANSWERED, claims=[
        DraftClaim(text="x", passages=[42]),
    ]))
    assert result.refusal is NarrativeRefusal.UNCITED


@pytest.mark.parametrize("outcome,refusal", [
    (AnswerOutcome.NOT_ADDRESSED, NarrativeRefusal.NOT_ADDRESSED),
    (AnswerOutcome.FIGURE_REQUESTED, NarrativeRefusal.FIGURE_REQUESTED),
])
def test_typed_outcomes_map_to_refusals(wired: dict, outcome, refusal) -> None:
    result = _answer(wired, DraftAnswer(outcome=outcome))
    assert result.refusal is refusal
    assert result.sections_searched == {"Item 7"}


def test_llm_error_is_failed_not_raised(wired: dict) -> None:
    assert _answer(wired, RuntimeError("boom")).refusal is NarrativeRefusal.FAILED


def test_survey_reports_every_company_with_its_status(wired: dict) -> None:
    wired["contexts"].update({
        "AAPL": _context(APPLE, RetrievalStatus.OK),
        "INTC": _context(INTEL, RetrievalStatus.NO_MATCHING_CHUNKS),
        "GOOGL": _context(GOOGLE, RetrievalStatus.OK),
    })
    wired["drafts"].update({
        "Apple Inc.": DraftVerdict(outcome=VerdictOutcome.SUPPORTED,
                                   claim="Apple manages interest rate risk.", passages=[1]),
        "Alphabet Inc.": DraftVerdict(outcome=VerdictOutcome.SUPPORTED,
                                      claim="Uncited yes.", passages=[8]),
    })
    result = asyncio.run(chain.survey_companies("q", Intent.MARKET_RISK, [APPLE, INTEL, GOOGLE]))
    assert [(f.ticker, f.status) for f in result.findings] == [
        ("AAPL", FindingStatus.SUPPORTED),
        ("INTC", FindingStatus.NOT_SEARCHABLE),
        ("GOOGL", FindingStatus.NOT_FOUND),  # uncited SUPPORTED is downgraded
    ]
    assert result.invalid_citations == 1


def test_survey_respects_the_concurrency_cap(wired: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chain.settings, "RAG_VERDICT_CONCURRENCY", 2)
    companies = [CompanyRef(ticker=f"T{i}", company_name=f"Co {i}", fiscal_year=2022)
                 for i in range(6)]
    for company in companies:
        wired["contexts"][company.ticker] = _context(company, RetrievalStatus.OK, n=1)
        wired["drafts"][company.company_name] = DraftVerdict(outcome=VerdictOutcome.NOT_FOUND)
    asyncio.run(chain.survey_companies("q", Intent.RISK_FACTORS, companies))
    assert wired["peak"] == 2


def test_survey_with_nothing_searchable_refuses(wired: dict) -> None:
    wired["contexts"]["INTC"] = _context(INTEL, RetrievalStatus.NO_MATCHING_CHUNKS)
    result = asyncio.run(chain.survey_companies("q", Intent.MANAGEMENT_COMMENTARY, [INTEL]))
    assert result.refusal is NarrativeRefusal.SECTION_ABSENT


def test_every_refusal_has_its_own_sentence() -> None:
    sentences = {refusal_message(r, "Intel", frozenset({"Item 7"})) for r in NarrativeRefusal}
    assert len(sentences) == len(NarrativeRefusal)
    assert "MD&A (Item 7)" in refusal_message(
        NarrativeRefusal.SECTION_ABSENT, "Intel", frozenset({"Item 7"}))

@pytest.mark.parametrize("raw,clean", [
    ("Goldman flags conduct risk (passage 1).", "Goldman flags conduct risk."),
    ("Risk from counterparties (passages 2 and 3), notably banks.",
     "Risk from counterparties, notably banks."),
    ("Cyber risk [4] is rising.", "Cyber risk is rising."),
    ("Clearing failures [1, 2].", "Clearing failures."),
    ("Conduct risk can cause losses (1).", "Conduct risk can cause losses."),
    ("Credit concentration (2, 3) is material.", "Credit concentration is material."),
    ("Rates were raised in (2021) twice.", "Rates were raised in (2021) twice."),
    ("Sales fell (10%) in Europe.", "Sales fell (10%) in Europe."),
    ("Rates rose 1.5 percentage points (Item 7A).", "Rates rose 1.5 percentage points (Item 7A)."),
])
def test_clean_claim_strips_only_passage_references(raw: str, clean: str) -> None:
    assert clean_claim(raw) == clean


def test_inline_passage_reference_is_stripped_from_answers(wired: dict) -> None:
    result = _answer(wired, DraftAnswer(outcome=AnswerOutcome.ANSWERED, claims=[
        DraftClaim(text="Growth came from iPhone (passage 1).", passages=[1]),
    ]))
    assert result.claims[0].text == "Growth came from iPhone."