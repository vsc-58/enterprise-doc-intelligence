"""Executor RAG wiring: single-company vs corpus-wide dispatch, outcome mapping.

No API, no stores: resolution, the corpus index and both chain entry points are
replaced.
"""

import asyncio

import pytest

from src.query import executor
from src.query.resolve import CorpusEntry, CorpusIndex
from src.query.schemas import (
    Intent,
    MetricField,
    NarrativeRefusal,
    NarrativeResult,
    Resolution,
    ResolvedEntity,
    Task,
    TaskStatus,
)

INTEL = ResolvedEntity(mention="Intel", ticker="INTC", cik="0000050863",
                       company_name="INTEL CORP", document_id=7, filing_year=2022)


def _entry(ticker: str, embedded: bool = True) -> CorpusEntry:
    return CorpusEntry(ticker=ticker, cik="1", company_name=f"{ticker} INC",
                       document_id=1, filing_year=2022, is_embedded=embedded)


INDEX = CorpusIndex(entries=[_entry("AAPL"), _entry("INTC"), _entry("ZZZ", embedded=False)],
                    tokens_by_ticker={})


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    state: dict = {"single": [], "survey": [], "result": NarrativeResult()}

    async def single(question, intent, company):
        state["single"].append(company)
        return state["result"]

    async def survey(question, intent, companies):
        state["survey"].append([c.ticker for c in companies])
        return state["result"]

    monkeypatch.setattr(executor, "answer_for_company", single)
    monkeypatch.setattr(executor, "survey_companies", survey)
    monkeypatch.setattr(executor, "resolve", lambda *a, **k: Resolution(entity=INTEL))
    return state


def _task(mention: str | None) -> Task:
    return Task(task_id="t1", intent=Intent.MANAGEMENT_COMMENTARY,
                company_mention=mention, question="Why did margins fall?")


def test_named_company_goes_to_the_single_company_chain(calls: dict) -> None:
    outcome = asyncio.run(executor._execute_task(_task("Intel"), INDEX))
    assert [c.ticker for c in calls["single"]] == ["INTC"]
    assert calls["single"][0].fiscal_year == 2022
    assert calls["survey"] == []
    assert outcome.status is TaskStatus.SUCCESS


@pytest.mark.parametrize("mention", [None, "", "   "])
def test_no_company_means_corpus_wide_over_embedded_companies(calls: dict, mention) -> None:
    asyncio.run(executor._execute_task(_task(mention), INDEX))
    assert calls["survey"] == [["AAPL", "INTC"]]
    assert calls["single"] == []


def test_sql_task_without_company_is_not_made_corpus_wide() -> None:
    task = Task(task_id="t1", intent=Intent.FINANCIAL_METRIC, metric=MetricField.NET_INCOME)
    assert executor.is_corpus_wide(task) is False


def test_section_absent_is_refused_with_the_named_section(calls: dict) -> None:
    calls["result"] = NarrativeResult(refusal=NarrativeRefusal.SECTION_ABSENT,
                                      sections_searched=frozenset({"Item 7"}))
    outcome = asyncio.run(executor._execute_task(_task("Intel"), INDEX))
    assert outcome.status is TaskStatus.REFUSED
    assert outcome.message.startswith("The MD&A (Item 7) section of Intel")
    assert outcome.narrative_result.refusal is NarrativeRefusal.SECTION_ABSENT


def test_narrative_failure_is_failed_not_refused(calls: dict) -> None:
    calls["result"] = NarrativeResult(refusal=NarrativeRefusal.FAILED)
    outcome = asyncio.run(executor._execute_task(_task("Intel"), INDEX))
    assert outcome.status is TaskStatus.FAILED


def test_an_exception_outside_the_chain_fails_the_task_not_the_plan(
    calls: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> CorpusIndex:
        raise RuntimeError("db down")

    monkeypatch.setattr(executor, "load_index", broken)
    outcome = asyncio.run(executor._execute_task(_task(None), None))
    assert outcome.status is TaskStatus.FAILED
    assert outcome.narrative_result.refusal is NarrativeRefusal.FAILED

# --- Section fallback (D44) ---------------------------------------------------

_ABSENT = frozenset({"Item 7", "Item 7A"})
_USED = frozenset({"Item 1A"})


def _market_risk_task() -> Task:
    return Task(task_id="t1", intent=Intent.MARKET_RISK, company_mention="Intel",
                question="How does Intel describe its interest rate risk?")


def test_fallback_answer_carries_the_disclosure_as_its_message(calls: dict) -> None:
    from src.query.schemas import CitedClaim, SourceRef
    ref = SourceRef(ticker="INTC", company_name="INTEL CORP", fiscal_year=2022,
                    section="Item 1A", chunk_index=3, distance=0.4)
    calls["result"] = NarrativeResult(
        claims=[CitedClaim(text="Rates affect returns.", sources=[ref])],
        sections_searched=_USED, fallback_from=_ABSENT,
    )
    outcome = asyncio.run(executor._execute_task(_market_risk_task(), INDEX))
    assert outcome.status is TaskStatus.SUCCESS
    assert outcome.message.startswith("The MD&A (Item 7) or Market Risk (Item 7A) section")
    assert "qualitatively" in outcome.message


def test_fallback_not_addressed_states_what_was_searched_then_refuses(calls: dict) -> None:
    calls["result"] = NarrativeResult(refusal=NarrativeRefusal.NOT_ADDRESSED,
                                      sections_searched=_USED, fallback_from=_ABSENT)
    outcome = asyncio.run(executor._execute_task(_market_risk_task(), INDEX))
    assert outcome.status is TaskStatus.REFUSED
    assert "was searched instead." in outcome.message
    assert outcome.message.endswith("do not address this question.")


def test_no_disclosure_without_a_fallback(calls: dict) -> None:
    calls["result"] = NarrativeResult(refusal=NarrativeRefusal.NOT_ADDRESSED,
                                      sections_searched=frozenset({"Item 7"}))
    outcome = asyncio.run(executor._execute_task(_task("Intel"), INDEX))
    assert "instead" not in outcome.message
