"""The query-path evaluation scripts refuse to run on an unconfirmed model.

Each script loads its inputs, then checks the served model before any paid
call. These tests replace the inputs with empty stand-ins and the check with a
stub, and prove that a failed check ends the run with exit code 1 before the
planner, the chain or the judge is ever called. No network, no database.
"""

from pathlib import Path
from typing import Any

import pytest

from scripts import run_attribution_eval, run_execution_eval, run_routing_eval
from src.utils.config import settings


def _boom(*args: Any, **kwargs: Any) -> Any:
    """Stand-in for a paid call: reaching it means the guard did not stop the run."""
    raise AssertionError("a paid call ran although the served-model check failed")


def test_routing_eval_refuses_before_any_planner_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(run_routing_eval, "load_eval_set", lambda: object())
    monkeypatch.setattr(run_routing_eval, "load_index", lambda: object())
    monkeypatch.setattr(run_routing_eval, "check_served_or_log",
                        lambda role, configured: seen.append((role, configured)))
    monkeypatch.setattr(run_routing_eval, "plan_query", _boom)
    assert run_routing_eval.main(runs=1, ids=None) == 1
    assert seen == [("query", settings.QUERY_MODEL)]


def test_execution_eval_refuses_before_any_question_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(run_execution_eval, "load_eval_set", lambda: object())
    monkeypatch.setattr(run_execution_eval, "load_oracle_params", lambda: {})
    monkeypatch.setattr(run_execution_eval, "load_index", lambda: object())
    monkeypatch.setattr(run_execution_eval, "check_served_or_log",
                        lambda role, configured: seen.append((role, configured)))
    monkeypatch.setattr(run_execution_eval, "plan_query", _boom)
    monkeypatch.setattr(run_execution_eval, "execute_plan", _boom)
    assert run_execution_eval.main(runs=1, ids=None, modes=["oracle"]) == 1
    assert seen == [("query", settings.QUERY_MODEL)]


def test_attribution_eval_refuses_before_any_judge_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (settings.ATTRIBUTION_SET_FILE, settings.ATTRIBUTION_SEEDS_FILE,
                 settings.ATTRIBUTION_LABELS_FILE):
        (tmp_path / name).write_text('{"sources": {}}', encoding="utf-8")
    execution = tmp_path / "execution.json"
    execution.write_text("{}", encoding="utf-8")
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(settings, "EVAL_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(run_attribution_eval, "load_eval_set",
                        lambda: type("E", (), {"questions": []})())
    monkeypatch.setattr(run_attribution_eval, "load_index", lambda: object())
    monkeypatch.setattr(run_attribution_eval, "claims_from_execution", lambda *a: [])
    monkeypatch.setattr(run_attribution_eval, "check_served_or_log",
                        lambda role, configured: seen.append((role, configured)))
    monkeypatch.setattr(run_attribution_eval, "judge_claim", _boom)
    assert run_attribution_eval.main(execution, runs=1, run_number=1) == 1
    assert seen == [("judge", settings.JUDGE_MODEL)]


def test_attribution_results_are_named_for_the_scored_versions() -> None:
    execution = {"planner_version": "aaaaaaaa1111", "chain_version": "bbbbbbbb2222"}
    first = run_attribution_eval.results_path("cccccccc3333", execution, 1)
    other = run_attribution_eval.results_path("cccccccc3333", execution, 2)
    assert first.name == "attribution_results_cccccccc_aaaaaaaa_bbbbbbbb.json"
    assert other.name == "attribution_results_cccccccc_aaaaaaaa_bbbbbbbb_run2.json"


def test_retrieval_results_are_named_for_the_routing_planner_version() -> None:
    from scripts import run_retrieval_eval

    assert run_retrieval_eval.results_path("abcdef1234", smoke=False).name == (
        "retrieval_results_abcdef12.json")
    assert run_retrieval_eval.results_path(None, smoke=True).name == (
        "retrieval_results_original_smoke.json")
