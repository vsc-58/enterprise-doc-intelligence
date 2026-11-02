"""The prompts in the code are the prompts the reported numbers were measured on.

Every headline number in the README comes from a results file in data/eval/,
and each results file records the version hash of the prompts that produced
it. A version hash covers the prompt text, the model name and the JSON schema
sent with the call, including model docstrings and field descriptions. So any
edit to that prompt surface, even one that looks like documentation, changes
the hash and fails this test.

When a test here fails, the served prompt is no longer the measured one. Either
undo the change, or re-run the evaluation that produced the file and point the
test at the new results.

Reads committed files only: no network, no API call, no database.
"""

import json
from pathlib import Path

import pytest

from src.eval.attribution import judge_version
from src.query.planner import planner_version
from src.rag.chain import chain_version
from src.utils.config import settings

# Results file -> the version fields it records. These are the sources of the
# README's routing, end-to-end, oracle, retrieval and attribution numbers, all
# measured on the pinned query model. The earlier files measured on the
# "gpt-4o-mini" alias stay in data/eval as history and are not pinned here.
MEASURED: dict[str, tuple[str, ...]] = {
    "routing_results_0dacd92a.json": ("planner_version",),
    "execution_results_0dacd92a_79177d36.json": ("planner_version", "chain_version"),
    "retrieval_results_0dacd92a.json": ("planner_version",),
    "attribution_results_95183c61_0dacd92a_79177d36.json": ("judge_version",),
}

# Results file -> (field naming the model the code is configured with now,
# field naming the model that actually answered during the run). Both must equal
# the current setting: a result measured on another snapshot cannot vouch for
# the one being served.
SERVED: dict[str, tuple[str, str, str]] = {
    "routing_results_0dacd92a.json": ("QUERY_MODEL", "model", "served_model"),
    "execution_results_0dacd92a_79177d36.json": ("QUERY_MODEL", "model", "served_model"),
    "retrieval_results_0dacd92a.json": (
        "QUERY_MODEL", "routing_model", "routing_served_model"),
    "attribution_results_95183c61_0dacd92a_79177d36.json": (
        "JUDGE_MODEL", "judge_model", "judge_served_model"),
}

CURRENT = {
    "planner_version": planner_version,
    "chain_version": chain_version,
    "judge_version": judge_version,
}


def _recorded(filename: str) -> dict:
    """Load one committed results file from the evaluation folder."""
    return json.loads(Path(settings.EVAL_DATA_PATH, filename).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("filename", "field"),
    [(filename, field) for filename, fields in MEASURED.items() for field in fields],
)
def test_code_runs_the_measured_prompt_version(filename: str, field: str) -> None:
    recorded = _recorded(filename)[field]
    assert CURRENT[field]() == recorded, (
        f"{field} in the code differs from the one recorded in {filename}: "
        "the numbers from that file no longer describe the served prompt"
    )


@pytest.mark.parametrize("filename", sorted(SERVED))
def test_results_were_measured_on_the_configured_model(filename: str) -> None:
    setting, configured_field, served_field = SERVED[filename]
    recorded = _recorded(filename)
    expected = getattr(settings, setting)
    assert recorded[configured_field] == expected, (
        f"{filename} was run with {configured_field}={recorded[configured_field]!r}, "
        f"but settings.{setting} is {expected!r}")
    assert recorded[served_field] == expected, (
        f"{filename} records the served model {recorded[served_field]!r}, "
        f"not the configured {expected!r}")


def test_attribution_scored_the_measured_execution_run() -> None:
    attribution = _recorded("attribution_results_95183c61_0dacd92a_79177d36.json")
    assert attribution["execution_file"] in MEASURED
    assert attribution["execution_served_model"] == settings.QUERY_MODEL

