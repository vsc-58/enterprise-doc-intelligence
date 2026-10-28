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
# README's routing, end-to-end, oracle, retrieval and attribution numbers.
MEASURED: dict[str, tuple[str, ...]] = {
    "routing_results_d4bc8e36.json": ("planner_version",),
    "execution_results_d4bc8e36_b1d8ee5c.json": ("planner_version", "chain_version"),
    "execution_results_d4bc8e36_b1d8ee5c_oracle.json": ("planner_version", "chain_version"),
    "retrieval_results.json": ("planner_version",),
    "attribution_results.json": ("judge_version",),
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
