"""Query-eval scoring tests: label loading, matching, miss classification, intervals.

No API and no database: plans are built by hand and resolved against the
conftest index.
"""

import json
from pathlib import Path

import pytest

from src.eval.query_eval import (
    AMBIGUOUS,
    NON_CORPUS,
    EvalQuestion,
    EvalSetError,
    MissCategory,
    ObservedPlan,
    ObservedTask,
    canonical_sha256,
    load_eval_set,
    observe,
    resolved_company,
    score_question,
    summarise,
    wilson_interval,
)
from src.query.schemas import Intent, MetricField, PlanType, QueryPlan, Task, UnsupportedReason


def _q(alternatives: list[dict], qid: str = "X1") -> EvalQuestion:
    return EvalQuestion(id=qid, question="q", split="dev", source="test",
                        stratum="s", alternatives=alternatives)


def _obs(plan_type: str, tasks: list[tuple] = (), reason: str | None = None,
         fallback: bool = False) -> ObservedPlan:
    return ObservedPlan(
        plan_type=plan_type, unsupported_reason=reason, fallback=fallback,
        tasks=[ObservedTask(intent=i, mention=c, company=c, metric=m) for i, c, m in tasks],
    )


HYBRID = {"plan_type": "hybrid", "tasks": [
    {"intent": "company_overview", "company": "AMZN"},
    {"intent": "financial_metric", "company": "AMZN", "metric": "net_income"},
]}


# --- Label file ----------------------------------------------------------------


def test_the_committed_label_file_is_the_frozen_one() -> None:
    eval_set = load_eval_set()
    assert len(eval_set.questions) == 109
    assert sum(q.split == "held" for q in eval_set.questions) == 42


def test_hash_ignores_line_endings(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{"a": 1}\n')
    crlf.write_bytes(b'{"a": 1}\r\n')
    assert canonical_sha256(lf) == canonical_sha256(crlf)


def test_edited_labels_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(EvalSetError, match="labels changed after the freeze"):
        load_eval_set(path)


# --- Resolution into the label vocabulary --------------------------------------


@pytest.mark.parametrize("mention,expected", [
    ("amzn", "AMZN"),
    ("Apple", "AAPL"),
    ("Nvidia", NON_CORPUS),
    ("the bank", AMBIGUOUS),
    (None, None),
])
def test_mentions_resolve_to_label_vocabulary(index, mention, expected) -> None:
    assert resolved_company(mention, index) == expected


def test_observe_reduces_a_plan_to_resolved_keys(index) -> None:
    plan = QueryPlan(plan_type=PlanType.HYBRID, rationale="r", tasks=[
        Task(task_id="t1", intent=Intent.COMPANY_OVERVIEW, company_mention="amzn",
             question="What does Amazon do?"),
        Task(task_id="t2", intent=Intent.FINANCIAL_METRIC, company_mention="Amazon",
             metric=MetricField.NET_INCOME),
    ])
    observed = observe(plan, index)
    assert [t.key() for t in observed.tasks] == [
        ("company_overview", "AMZN", None, None, None),
        ("financial_metric", "AMZN", "net_income", None, None),
    ]
    assert not observed.fallback


def test_planner_fallback_is_detected(index) -> None:
    plan = QueryPlan(plan_type=PlanType.CLARIFICATION, clarification_question="?",
                     rationale="planner fallback: schema validation failed")
    assert observe(plan, index).fallback


# --- Matching ------------------------------------------------------------------


def test_hybrid_match_ignores_task_order() -> None:
    observed = _obs("hybrid", [("financial_metric", "AMZN", "net_income"),
                               ("company_overview", "AMZN", None)])
    score = score_question(_q([HYBRID]), observed)
    assert score.strict and score.lenient


def test_alternative_counts_only_for_lenient() -> None:
    question = _q([
        {"plan_type": "single", "tasks": [{"intent": "strategy", "company": "XOM"}]},
        {"plan_type": "single", "tasks": [{"intent": "management_commentary", "company": "XOM"}]},
    ])
    score = score_question(question, _obs("single", [("management_commentary", "XOM", None)]))
    assert not score.strict and score.lenient and score.miss is None


def test_unsupported_must_name_the_right_reason() -> None:
    question = _q([{"plan_type": "unsupported", "unsupported_reason": "cross_year"}])
    assert score_question(question, _obs("unsupported", reason="cross_year")).strict
    miss = score_question(question, _obs("unsupported", reason="aggregation"))
    assert miss.miss is MissCategory.UNSUPPORTED_REASON


def test_fallback_never_counts_even_when_clarification_was_expected() -> None:
    score = score_question(_q([{"plan_type": "clarification"}]),
                           _obs("clarification", fallback=True))
    assert not score.lenient and score.miss is MissCategory.PLANNER_FALLBACK


# --- Miss classification ---------------------------------------------------------


@pytest.mark.parametrize("observed,category", [
    (_obs("out_of_scope"), MissCategory.PLAN_TYPE),
    (_obs("hybrid", [("company_overview", "AMZN", None),
                     ("financial_metric", "AMZN", "net_income"),
                     ("risk_factors", "AMZN", None)]), MissCategory.TASK_COUNT),
    (_obs("hybrid", [("strategy", "AMZN", None),
                     ("financial_metric", "AMZN", "net_income")]), MissCategory.INTENT),
    (_obs("hybrid", [("company_overview", "WMT", None),
                     ("financial_metric", "AMZN", "net_income")]), MissCategory.COMPANY),
    (_obs("hybrid", [("company_overview", "AMZN", None),
                     ("financial_metric", "AMZN", "total_revenue")]), MissCategory.PARAMETER),
])
def test_miss_is_classified_at_the_first_differing_level(observed, category) -> None:
    assert score_question(_q([HYBRID]), observed).miss is category


def test_unresolvable_mention_of_a_held_company_is_flagged_as_resolver() -> None:
    observed = _obs("hybrid", [("company_overview", NON_CORPUS, None),
                               ("financial_metric", "AMZN", "net_income")])
    score = score_question(_q([HYBRID]), observed)
    assert score.miss is MissCategory.COMPANY and score.resolver_suspect


def test_wrong_held_company_is_not_a_resolver_suspect() -> None:
    observed = _obs("hybrid", [("company_overview", "WMT", None),
                               ("financial_metric", "AMZN", "net_income")])
    assert not score_question(_q([HYBRID]), observed).resolver_suspect


# --- Aggregation -----------------------------------------------------------------


def test_wilson_interval_matches_known_values() -> None:
    lo, hi = wilson_interval(15, 17)
    assert round(lo, 2) == 0.66 and round(hi, 2) == 0.97
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_summary_counts_disagreement_separately_from_accuracy() -> None:
    a, b = _q([HYBRID], "A"), _q([HYBRID], "B")
    right = score_question(a, _obs("hybrid", [("company_overview", "AMZN", None),
                                              ("financial_metric", "AMZN", "net_income")]))
    wrong = score_question(a, _obs("out_of_scope"))
    other_wrong = score_question(a, _obs("clarification"))
    scores = {"A": [right, wrong, right], "B": [wrong, other_wrong, wrong]}
    signatures = {"A": ["p1", "p2", "p1"], "B": ["w1", "w2", "w1"]}
    summary = summarise([a, b], scores, signatures)
    assert summary.strict_per_run == [1, 0, 1]
    assert summary.plan_varied == 2      # both changed plans between runs
    assert summary.score_varied == 1     # only A changed correctness
    assert summary.strict_mean == pytest.approx(1 / 3)
