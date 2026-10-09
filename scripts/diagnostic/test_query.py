"""
scripts/diagnostic/test_query.py — DIAGNOSTIC (print() allowed).

End-to-end check of the full query path: question -> planner -> resolver ->
executor -> SQL path and/or narrative chain -> synthesizer -> assembled answer.
Supersedes test_query_sql.py (Phase 5A), whose 13 SQL cases are kept as
regressions, with the last one now answered by the narrative path.

Phase 5B adds: single-company narrative, a section-absent refusal, a figure asked
as narrative, the R38 partial hybrid, a full hybrid whose figure must appear once,
and the corpus-wide headline question.

What to look for in the 5B cases:
  - every narrative sentence carries [n] markers and every [n] is in Sources;
  - R38 prints Intel's figure, then the margin question with the not-captured
    sentence — never a dropped half;
  - the revenue hybrid states revenue exactly once, from the SQL part;
  - the corpus-wide answer ends with the coverage lines.

Calls the OpenAI API: one planner call per question plus the chain calls, about
$0.02 total.

Usage:
    python -m scripts.diagnostic.test_query

Dependencies: src.query.executor, src.query.planner, src.query.synthesizer,
src.utils.logger.
"""

from __future__ import annotations

import asyncio

from src.query.executor import execute_plan
from src.query.planner import plan_query, planner_version
from src.query.schemas import PlanType, QueryPlan
from src.query.synthesizer import assemble
from src.rag.chain import chain_version
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
     "5B: cited risk-factor answer (was the RAG stub)"),
    # --- Phase 5B ---------------------------------------------------------
    ("Why did Apple's net sales grow?",
     "cited MD&A claims, no dollar figure, Sources list"),
    ("Why did Intel's revenue change?",
     "refused: Intel's MD&A (Item 7) not captured"),
    ("What does Meta do?",
     "answered from Item 1A via the alias table (D40)"),
    ("What was Intel's net income, and why did its margins fall?",
     "R38: figure with its flag, then the margin question + not-captured sentence"),
    ("What was Apple's revenue and why did it grow?",
     "hybrid: revenue stated ONCE (SQL part); narrative part has no dollar figure"),
    ("Which companies flagged interest rate risk?",
     "corpus-wide: supported list with markers, then coverage lines (JPM, XOM)"),
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


async def run_case(question: str, expectation: str) -> None:
    """
    Plan, execute and assemble one question, printing the answer.

    Args:
        question: The question to run.
        expectation: What the run should produce, for eyeball comparison.
    """
    plan = plan_query(question)
    outcomes = await execute_plan(plan)
    answer = assemble(plan, outcomes)

    print(f"\n{'=' * 78}\nQ: {question}")
    print(f"expect: {expectation}")
    print(f"plan  : {describe_plan(plan)}")
    print(f"status: {[status.value for status in answer.statuses] or plan.plan_type.value}")
    print("-" * 78)
    print(answer.text)


async def main() -> int:
    """Entry point. Returns the process exit code."""
    print(f"planner_version: {planner_version()}  chain_version: {chain_version()}")

    failures = 0
    for question, expectation in CASES:
        try:
            await run_case(question, expectation)
        except Exception as exc:  # noqa: BLE001 — a probe reports, it doesn't abort
            failures += 1
            print(f"\n{'=' * 78}\nQ: {question}\n  ERROR {type(exc).__name__}: {exc}")
            logger.error("case_failed", question=question, error=str(exc))

    print(f"\n{'=' * 78}\n{len(CASES)} cases, {failures} error(s)")
    logger.info("query_probe_complete", cases=len(CASES), errors=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))