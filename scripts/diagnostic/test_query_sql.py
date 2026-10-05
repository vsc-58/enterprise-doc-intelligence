"""
scripts/test_query_sql.py — DIAGNOSTIC (print() allowed).

End-to-end check of Phase 5A: question -> planner -> resolver -> executor ->
SQL path -> rendered answer. Narrative tasks are expected to refuse; retrieval
arrives in 5B.

Covers the four original worked examples with Apple corrected to FY2021, the
bounded-capability case, a flagged figure, a null-exclusion ranking, a mixed-year
filter, and the three refusal plan types.

Calls the OpenAI API: one planner call per question, roughly $0.004 total.

Usage:
    python -m scripts.test_query_sql

Dependencies: src.query.executor, src.query.planner, src.utils.logger.
"""

from __future__ import annotations

import asyncio

from src.query.planner import plan_query, planner_version
from src.query.executor import execute_plan
from src.query.schemas import PlanType, QueryPlan, TaskOutcome, TaskStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)

CASES: list[tuple[str, str]] = [
    ("What was Apple's total revenue?",
     "FY2021, $365.82 billion — the corpus holds Apple for 2021, not 2022"),
    ("What was Apple's 2022 revenue?",
     "case (b): names 2021 as the year held"),
    ("Which companies had net income above $50 billion?",
     "4 rows across 3 fiscal years, with the differing-periods note"),
    ("Which company had the highest total assets?",
     "JPM; BAC is 1000x low but is not in the top row"),
    ("What were Bank of America's total assets?",
     "$3.05 billion, flagged scale_ambiguous"),
    ("Which company had the lowest operating cash flow?",
     "excludes BAC (null), and says so"),
    ("What was Target's revenue?",
     "FY2021, period ending 2021-01-30"),
    ("What was Nvidia's revenue?",
     "resolver refusal: non_corpus_company"),
    ("What was the bank's net income?",
     "resolver refusal: ambiguous_company"),
    ("What is the average revenue across all companies?",
     "unsupported / aggregation"),
    ("Who audits Apple's financial statements?",
     "unsupported / field_not_queryable"),
    ("What is Apple's current stock price?",
     "out_of_scope"),
    ("What risks did Goldman Sachs flag?",
     "RAG stub refusal until 5B"),
]


def describe_plan(plan: QueryPlan) -> str:
    """
    Render a one-line summary of a plan.

    Args:
        plan: The plan to describe.

    Returns:
        Plan type, with its reason or its intents.
    """
    if plan.plan_type is PlanType.UNSUPPORTED:
        reason = plan.unsupported_reason.value if plan.unsupported_reason else "?"
        return f"{plan.plan_type.value} / {reason}"
    if plan.plan_type in {PlanType.CLARIFICATION, PlanType.OUT_OF_SCOPE}:
        return plan.plan_type.value
    intents = ", ".join(task.intent.value for task in plan.tasks)
    return f"{plan.plan_type.value} [{intents}]"


def describe_outcome(outcome: TaskOutcome) -> str:
    """
    Render one task outcome for display.

    Args:
        outcome: The outcome.

    Returns:
        Status, resolution failure if any, and the answer text indented.
    """
    head = f"  [{outcome.status.value}] {outcome.intent.value}"
    if outcome.resolution_failure:
        head += f" ({outcome.resolution_failure.value})"
    body = "\n".join(
        f"    {line}" for line in (outcome.message or "").splitlines()
    )
    return f"{head}\n{body}" if body else head


def refusal_text(plan: QueryPlan) -> str:
    """
    Render the user-facing text for a plan that carries no tasks.

    Phase 5B moves this into the assembler, where it is shared with the hybrid
    path; it lives here so 5A can be verified end to end.

    Args:
        plan: A clarification, unsupported or out-of-scope plan.

    Returns:
        The text shown to the user.
    """
    if plan.plan_type is PlanType.CLARIFICATION:
        return plan.clarification_question or "Could you rephrase that?"
    if plan.plan_type is PlanType.OUT_OF_SCOPE:
        return (
            "That isn't something a 10-K annual report covers, so I don't hold it."
        )
    reason = plan.unsupported_reason.value if plan.unsupported_reason else "unsupported"
    explanations = {
        "cross_year": (
            "I hold one annual filing per company, so I can't compare a company "
            "against its own earlier years."
        ),
        "aggregation": (
            "I can look up, rank and filter company figures, but I don't compute "
            "aggregates across the corpus."
        ),
        "ratio_or_derived": (
            "I serve the five figures as filed. I don't compute ratios or derived "
            "measures from them."
        ),
        "multi_condition": (
            "I can filter on one condition at a time. Could you ask for one of "
            "them, and I'll follow up with the other?"
        ),
        "field_not_queryable": (
            "That field is extracted and appears in the CSV export, but the query "
            "path doesn't serve it."
        ),
    }
    return explanations.get(reason, "That's outside what this system can answer.")


async def run_case(question: str, expectation: str) -> None:
    """
    Plan and execute one question, printing the result.

    Args:
        question: The question to run.
        expectation: What the run should produce, for eyeball comparison.
    """
    plan = plan_query(question)
    outcomes = await execute_plan(plan)

    print(f"\n{'=' * 78}\nQ: {question}")
    print(f"expect: {expectation}")
    print(f"plan  : {describe_plan(plan)}")

    if not outcomes:
        print(f"  [refused] {refusal_text(plan)}")
        return
    for outcome in outcomes:
        print(describe_outcome(outcome))


async def main() -> int:
    """Entry point. Returns the process exit code."""
    print(f"planner_version: {planner_version()}")

    failures = 0
    for question, expectation in CASES:
        try:
            await run_case(question, expectation)
        except Exception as exc:  # noqa: BLE001 — a probe reports, it doesn't abort
            failures += 1
            print(f"\n{'=' * 78}\nQ: {question}\n  ERROR {type(exc).__name__}: {exc}")
            logger.error("case_failed", question=question, error=str(exc))

    print(f"\n{'=' * 78}\n{len(CASES)} cases, {failures} error(s)")
    logger.info("sql_query_probe_complete", cases=len(CASES), errors=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))