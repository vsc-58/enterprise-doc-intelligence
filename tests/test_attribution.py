# tests/test_attribution.py
# Attribution judge (D43): statistics, claim extraction, and judge_claim with the
# model stubbed — no network.

import json
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from src.eval import attribution
from src.eval.attribution import (
    JudgeLabel,
    JudgeVerdict,
    claims_from_execution,
    cohen_kappa,
    flipped,
    judge_claim,
    judge_version,
    majority,
    render_excerpts,
    split_key,
)
from src.storage.vector_store import RetrievedChunk
from src.utils.config import settings

S, N = JudgeLabel.SUPPORTED, JudgeLabel.NOT_SUPPORTED

BLOCK = RetrievedChunk(text="We have $15.0 billion of undrawn  committed lines of credit.",
                       ticker="WMT", company_name="Walmart Inc.", cik="1", fiscal_year=2023,
                       section="Item 7", chunk_index=25, distance=0.0)


class _FakeLLM:
    """Stands in for ChatOpenAI: with_structured_output returns a fixed response."""

    def __init__(self, response: dict) -> None:
        self.response = response

    def with_structured_output(self, schema: type, method: str, include_raw: bool) -> RunnableLambda:
        return RunnableLambda(lambda _: self.response)


def _stub(monkeypatch: pytest.MonkeyPatch, response: dict) -> None:
    monkeypatch.setattr(attribution, "_get_llm", lambda: _FakeLLM(response))


# --- Statistics ------------------------------------------------------------------


def test_kappa_perfect_chance_and_undefined() -> None:
    assert cohen_kappa([S, N, S, N], [S, N, S, N]) == 1.0
    # A judge that always says SUPPORTED: high raw agreement, kappa 0.
    assert cohen_kappa([S, S, S, N], [S, S, S, S]) == 0.0
    assert cohen_kappa([S, S], [S, S]) is None
    assert cohen_kappa([], []) is None
    with pytest.raises(ValueError):
        cohen_kappa([S], [S, N])


def test_majority_tie_is_not_supported_and_failures_ignored() -> None:
    assert majority([S, S, N]) is S
    assert majority([S, N]) is N
    assert majority([S, None, None]) is S
    assert majority([None, None]) is None


def test_flipped_ignores_failed_calls() -> None:
    assert flipped([S, N, S])
    assert not flipped([S, None, S])


# --- Judge call ------------------------------------------------------------------


def test_supported_verdict_checks_the_quoted_span(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, {"parsed": JudgeVerdict(verdict=S, supporting_span="undrawn committed lines",
                                               unsupported_detail=""), "parsing_error": None})
    result = judge_claim("Walmart Inc.", "claim", [BLOCK])
    assert result.label is S and result.span_found is True


def test_invented_quote_is_flagged_not_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, {"parsed": JudgeVerdict(verdict=S, supporting_span="expire in 2026",
                                               unsupported_detail=""), "parsing_error": None})
    result = judge_claim("Walmart Inc.", "claim", [BLOCK])
    assert result.label is S and result.span_found is False


def test_parse_failure_and_exception_return_no_label(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, {"parsed": None, "parsing_error": "bad json"})
    assert judge_claim("W", "c", [BLOCK]).label is None

    def boom() -> None:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(attribution, "_get_llm", boom)
    result = judge_claim("W", "c", [BLOCK])
    assert result.label is None and "RuntimeError" in (result.error or "")


def test_excerpts_carry_company_and_section() -> None:
    assert render_excerpts([BLOCK]).startswith("[1] Walmart Inc. · FY2023 · Item 7\n")


def test_judge_version_tracks_model(monkeypatch: pytest.MonkeyPatch) -> None:
    before = judge_version()
    monkeypatch.setattr(settings, "JUDGE_MODEL", "another-model")
    assert judge_version() != before


# --- Claims ------------------------------------------------------------------------


def test_claims_from_execution_ids_and_kinds() -> None:
    src = {"ticker": "WMT", "section": "Item 7", "chunk_index": 25}
    doc = {"modes": {"end_to_end": {"records": [[
        {"id": "R28", "tasks": [{"task_id": "1", "company": "WMT",
                                 "claims": [{"text": "a", "sources": [src]}], "findings": []}]},
        {"id": "H08", "tasks": [{"task_id": "1", "company": None, "claims": [], "findings": [
            {"ticker": "CRM", "claim": {"text": "b", "sources": [src]}},
            {"ticker": "JPM", "claim": None}]}]},
    ]]}}}
    claims = claims_from_execution(doc)
    assert [(c.id, c.kind, c.company) for c in claims] == [
        ("R28|1|c1", "claim", "WMT"), ("H08|1|CRM", "finding", "CRM")]
    assert claims[0].sources == ["WMT|Item 7|25"]
    assert split_key("WMT|Item 7|25") == ("WMT", "Item 7", 25)


def test_committed_validation_set_is_consistent() -> None:
    spec = json.loads((Path(settings.EVAL_DATA_PATH) / settings.ATTRIBUTION_SET_FILE)
                      .read_text(encoding="utf-8"))
    items = {i["id"]: i for i in spec["items"]}
    assert sum(i["split"] == "test" for i in items.values()) == 40
    assert sum(i["split"] == "calib" for i in items.values()) == 20
    types = [s["type"] for s in spec["seeds"]]
    assert {t: types.count(t) for t in set(types)} == {
        "invented_detail": 10, "wrong_block_same_company": 10, "wrong_company_same_topic": 10}
    for seed in spec["seeds"]:
        base = items[seed["base"]]
        assert base["split"] == "test"
        if seed["type"] == "invented_detail":
            assert seed["claim"] != base["claim"] and seed["sources"] == base["sources"]


# --- Runner, end to end with stubs ----------------------------------------------------


def test_runner_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_attribution_eval as runner
    from src.eval.attribution import JudgeResult
    from src.eval.query_eval import EvalSet
    from src.query.resolve import CorpusEntry, build_index

    key = "WMT|Item 7|25"
    spec = {"items": [
        {"id": "C01", "split": "calib", "claim_id": "R28|1|c1", "question_id": "R28", "kind": "claim",
         "company": "WMT", "claim": "good", "sources": [key]},
        {"id": "T01", "split": "test", "claim_id": "R28|1|c1", "question_id": "R28", "kind": "claim",
         "company": "WMT", "claim": "good", "sources": [key]},
        {"id": "T02", "split": "test", "claim_id": "R28|1|c2", "question_id": "R28", "kind": "claim",
         "company": "WMT", "claim": "bad", "sources": [key]},
    ], "seeds": [{"id": "S3-01", "type": "invented_detail", "base": "T01", "company": "WMT",
                  "claim": "bad seed", "sources": [key]}]}
    (tmp_path / settings.ATTRIBUTION_SET_FILE).write_text(json.dumps(spec))
    (tmp_path / settings.ATTRIBUTION_SEEDS_FILE).write_text(json.dumps({"sources": {"S3-01": [key]}}))
    (tmp_path / settings.ATTRIBUTION_LABELS_FILE).write_text(json.dumps(
        {"test": {"T01": "S", "T02": "N"}, "calib": {"C01": "S"}}))
    src = {"ticker": "WMT", "section": "Item 7", "chunk_index": 25}
    execution = {"modes": {"end_to_end": {"records": [[{"id": "R28", "tasks": [
        {"task_id": "1", "company": "WMT", "findings": [],
         "claims": [{"text": "good", "sources": [src]}, {"text": "bad", "sources": [src]}]}]}]]}}}
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(execution))

    monkeypatch.setattr(settings, "EVAL_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(runner, "load_eval_set", lambda: EvalSet.model_validate({"sha256": "x", "questions": [{
        "id": "R28", "question": "q", "split": "dev", "source": "t", "stratum": "rag_single",
        "alternatives": [{"plan_type": "single", "tasks": [{"intent": "management_commentary",
                                                            "company": "WMT"}]}]}]}))
    monkeypatch.setattr(runner, "load_index", lambda: build_index([CorpusEntry(
        ticker="WMT", cik="1", company_name="Walmart Inc.", document_id=1, filing_year=2023)]))
    monkeypatch.setattr(runner, "get_chunks_by_key", lambda keys: {("WMT", "Item 7", 25): BLOCK})
    monkeypatch.setattr(runner, "judge_claim", lambda company, claim, blocks: JudgeResult(
        label=S if claim == "good" else N, supporting_span="x", span_found=True))

    assert runner.main(execution_path, runs=3, run_number=1) == 0
    result = json.loads((tmp_path / settings.ATTRIBUTION_RESULTS_FILE).read_text())
    assert result["judge_passed"] is True
    assert result["criteria"]["kappa_test_plus_seeds"]["value"] == 1.0
    assert result["headline"]["n"] == 2 and result["headline"]["supported"] == 1
    assert [x["id"] for x in result["not_supported_claims"]] == ["R28|1|c2"]
    assert (tmp_path / "attribution_results.md").exists()


def test_parse_label_sheet() -> None:
    from src.eval.attribution import LabelSheetError, parse_label_sheet

    sheet = "## T01\n\n**Claim:** x\n\nLabel: s\n\n---\n\n## T02\n\nLabel:  N \n"
    assert parse_label_sheet(sheet, ["T01", "T02"]) == {"T01": "S", "T02": "N"}
    with pytest.raises(LabelSheetError, match="T02"):
        parse_label_sheet("## T01\nLabel: S\n## T02\nLabel: \n", ["T01", "T02"])
    with pytest.raises(LabelSheetError, match="S or N"):
        parse_label_sheet("## T01\nLabel: yes\n", ["T01"])
