# tests/test_retrieval_eval.py
# Retrieval gold validation, ranking metrics, planner-query extraction, and one
# end-to-end run of the script with the retriever and corpus index stubbed out.

import json
from pathlib import Path

import pytest

from scripts import run_retrieval_eval as runner
from src.eval.retrieval_eval import (
    GoldError,
    GoldQuestion,
    PlannerQuery,
    load_gold,
    mean_or_none,
    planner_queries,
    score_ranking,
    unjudged_in_top,
)
from src.eval.query_eval import EvalSet
from src.query.resolve import CorpusEntry, build_index
from src.rag.retriever import CompanyContext, ContextBlock
from src.storage.vector_store import RetrievalStatus, RetrievedChunk
from src.utils.config import settings


def _gq(**overrides: object) -> GoldQuestion:
    """A small gold question: 3 judged chunks, 1 rel, 1 border."""
    base: dict[str, object] = {
        "id": "Q1", "split": "dev", "question": "q", "ticker": "AAA", "intent": "strategy",
        "pool": ["AAA|Item 7|1", "AAA|Item 7|2", "AAA|Item 7|3"],
        "labels": [
            {"key": "AAA|Item 7|1", "relevance": "rel", "evidence": "e"},
            {"key": "AAA|Item 7|2", "relevance": "border", "evidence": "e"},
        ],
    }
    base.update(overrides)
    return GoldQuestion.model_validate(base)


# --- Gold ---------------------------------------------------------------------


def test_relevant_sets_strict_and_lenient() -> None:
    gq = _gq()
    assert gq.relevant(lenient=False) == {"AAA|Item 7|1"}
    assert gq.relevant(lenient=True) == {"AAA|Item 7|1", "AAA|Item 7|2"}
    assert gq.judged() == {"AAA|Item 7|1", "AAA|Item 7|2", "AAA|Item 7|3"}


def test_label_outside_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing from the pool"):
        _gq(labels=[{"key": "AAA|Item 7|9", "relevance": "rel", "evidence": "e"}])


def test_duplicate_label_is_rejected() -> None:
    label = {"key": "AAA|Item 7|1", "relevance": "rel", "evidence": "e"}
    with pytest.raises(ValueError, match="labelled twice"):
        _gq(labels=[label, label])


def test_load_gold_missing_and_malformed(tmp_path: Path) -> None:
    with pytest.raises(GoldError, match="not found"):
        load_gold(tmp_path / "nope.json", expected_sha256="")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldError, match="malformed"):
        load_gold(bad, expected_sha256="")


def test_load_gold_refuses_edited_file(tmp_path: Path) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"status": "verified", "labelled_by": "t", "relevance_rule": "r",
                                "pool_method": "p", "questions": []}), encoding="utf-8")
    with pytest.raises(GoldError, match="changed after verification"):
        load_gold(gold, expected_sha256="0" * 64)


def test_committed_gold_matches_frozen_hash() -> None:
    gold = load_gold()
    assert gold.status == "verified"


# --- Metrics --------------------------------------------------------------------


def test_recall_is_capped_by_k() -> None:
    relevant = frozenset(f"k{i}" for i in range(10))
    ranking = [f"k{i}" for i in range(5)] + ["x1", "x2"]
    score = score_ranking(ranking, relevant, relevant | {"x1", "x2"}, [5, 10])
    assert score.at(5).recall == 1.0           # 5 of min(5, 10)
    assert score.at(10).recall == 0.5          # 5 of min(10, 10)
    assert score.at(5).precision == 1.0
    assert score.first_relevant_rank == 1


def test_miss_hit_and_first_rank() -> None:
    ranking = ["a", "b", "c", "d", "e", "f", "rel"]
    score = score_ranking(ranking, frozenset({"rel"}), frozenset(ranking), [5, 10])
    assert score.at(5).hit is False
    assert score.at(5).recall == 0.0
    assert score.at(10).hit is True
    assert score.first_relevant_rank == 7


def test_no_relevant_chunks_gives_none_not_zero() -> None:
    score = score_ranking(["a", "b"], frozenset(), frozenset({"a", "b"}), [5])
    assert score.at(5).hit is None
    assert score.at(5).recall is None
    assert score.at(5).precision == 0.0       # 2 chunks, none relevant
    assert score.first_relevant_rank is None


def test_precision_uses_chunks_available_and_counts_unjudged() -> None:
    score = score_ranking(["rel", "new"], frozenset({"rel"}), frozenset({"rel"}), [5])
    assert score.at(5).precision == 0.5       # 2 chunks exist, not 5
    assert score.at(5).unjudged == 1
    assert unjudged_in_top(["rel", "new", "z"], frozenset({"rel"}), 2) == ["new"]


def test_bad_cutoff_raises() -> None:
    with pytest.raises(ValueError):
        score_ranking(["a"], frozenset(), frozenset(), [0])
    with pytest.raises(KeyError):
        score_ranking(["a"], frozenset(), frozenset(), [5]).at(10)


# --- Planner queries --------------------------------------------------------------


def test_planner_queries_per_run_with_misses_kept() -> None:
    record = {"runs": [
        {"observed": {"tasks": [{"intent": "strategy", "company": "AAA", "question": "q1"}]}},
        {"observed": {"tasks": [{"intent": "risk_factors", "company": "AAA", "question": "q2"}]}},
        {"observed": None},
        {"observed": {"tasks": [
            {"intent": "strategy", "company": "BBB", "question": "q3"},
            {"intent": "strategy", "company": "AAA", "question": "q4"},
        ]}},
    ]}
    got = planner_queries(record, ["strategy"], "AAA")
    assert [q.question if q else None for q in got] == ["q1", None, None, "q4"]


def test_planner_queries_accept_alternative_intent_strict_first() -> None:
    record = {"runs": [
        {"observed": {"tasks": [{"intent": "risk_factors", "company": "AAA", "question": "q1"}]}},
        {"observed": {"tasks": [
            {"intent": "risk_factors", "company": "AAA", "question": "q2"},
            {"intent": "competition_and_regulation", "company": "AAA", "question": "q3"},
        ]}},
    ]}
    got = planner_queries(record, ["competition_and_regulation", "risk_factors"], "AAA")
    assert got[0] == PlannerQuery(intent="risk_factors", question="q1")
    assert got[1] == PlannerQuery(intent="competition_and_regulation", question="q3")


def test_mean_or_none() -> None:
    assert mean_or_none([None, 1.0, 0.0]) == 0.5
    assert mean_or_none([None]) is None


# --- Script, end to end with stubs ----------------------------------------------------


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(text=f"text {index}", ticker="AAA", company_name="Aaa Inc", cik="1",
                          fiscal_year=2023, section="Item 7", chunk_index=index, distance=0.1 * index)


def test_runner_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gold = {
        "status": "proposed", "labelled_by": "t", "relevance_rule": "r", "pool_method": "p",
        "questions": [_gq(recheck=["AAA|Item 7|3"]).model_dump(mode="json")],
    }
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / settings.RETRIEVAL_GOLD_FILE).write_text(json.dumps(gold), encoding="utf-8")
    routing = {"planner_version": "abcdef123", "questions": [{"id": "Q1", "runs": [
        {"observed": {"tasks": [{"intent": "strategy", "company": "AAA", "question": "rephrased"}]}},
        {"observed": None},
    ]}]}
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(json.dumps(routing), encoding="utf-8")

    monkeypatch.setattr(settings, "EVAL_DATA_PATH", str(tmp_path / "eval"))
    monkeypatch.setattr(settings, "RETRIEVAL_GOLD_SHA256", "")
    monkeypatch.setattr(settings, "PROCESSED_DATA_PATH", str(tmp_path / "processed"))
    monkeypatch.setattr(runner, "load_eval_set", lambda: EvalSet.model_validate({"sha256": "x", "questions": [{
        "id": "Q1", "question": "q", "split": "dev", "source": "t", "stratum": "rag_single",
        "alternatives": [{"plan_type": "single",
                          "tasks": [{"intent": "strategy", "company": "AAA"}]}]}]}))
    monkeypatch.setattr(runner, "load_index", lambda: build_index([CorpusEntry(
        ticker="AAA", cik="1", company_name="Aaa Inc", document_id=1, filing_year=2023)]))

    orders = {"q": [4, 1, 2, 3], "rephrased": [1, 4, 2, 3]}   # chunk 4 is unjudged

    def fake_retrieve(question: str, intent: object, company: object, k: int) -> CompanyContext:
        blocks = [ContextBlock(number=i, chunk=_chunk(c)) for i, c in enumerate(orders[question], 1)]
        return CompanyContext(company=company, sections_searched=frozenset({"Item 7"}),
                              status=RetrievalStatus.OK, blocks=blocks)

    monkeypatch.setattr(runner, "retrieve_for_company", fake_retrieve)

    assert runner.main(routing_path, None) == 0

    results = json.loads((tmp_path / "eval" / settings.RETRIEVAL_RESULTS_FILE).read_text())
    record = results["questions"][0]
    assert record["planner_unmeasured_runs"] == 1
    strict = record["summary"]["strict"]
    assert strict["original"]["first_rank"] == 2.0
    assert strict["planner"]["first_rank"] == 1.0
    assert results["provisional"] is True
    assert results["summary"]["specific"]["strict"]["all"]["n"] == 1

    to_label = json.loads((tmp_path / "processed" / settings.RETRIEVAL_TO_LABEL_FILE).read_text())
    reasons = {(v["key"], v["reason"]) for v in to_label}
    assert ("AAA|Item 7|4", "unjudged in top-k") in reasons
    assert ("AAA|Item 7|3", "judged on partial text") in reasons
    assert next(v for v in to_label if v["key"] == "AAA|Item 7|4")["best_rank"] == 1
    assert (tmp_path / "eval" / "retrieval_results.md").exists()
