"""
scripts/probe_planner.py — DIAGNOSTIC (print() allowed).

Smoke-tests the planner on one question per plan type before the executor is
built, so a prompt problem surfaces here rather than inside an execution trace.

Not the routing evaluation. These questions overlap the Phase 5C set, so they
are a development aid only — tuning the prompt against them and then reporting
5C numbers from the same questions would be scoring the prompt on its own
training data (D17).

Calls the OpenAI API: ~11 calls, roughly $0.003.

Usage:
    python -m scripts.probe_planner

Dependencies: src.query.planner, src.utils.logger.
"""

from __future__ import annotations

from src.query.planner import plan_query, planner_version
from src.utils.logger import get_logger

logger = get_logger(__name__)

QUESTIONS: list[tuple[str, str]] = [
    ("single/sql", "What was Apple's total revenue?"),
    ("single/sql", "Which companies had net income above $50 billion?"),
    ("single/sql", "According to the MD&A, what was Apple's total net sales?"),
    ("single/rag", "What risks did Goldman Sachs flag?"),
    ("single/rag", "Why did Intel's margins decline?"),
    ("hybrid", "What does Amazon do and what was its net income?"),
    ("hybrid/3", "What does Amazon do, what risks did it flag, and what was its net income?"),
    ("unsupported", "By what percentage did Apple's revenue change from 2020 to 2021?"),
    ("unsupported", "Who audits Apple's financial statements?"),
    ("out_of_scope", "What is Apple's current stock price?"),
    ("clarification", "What was its total revenue?"),
]


def main() -> int:
    """Entry point. Returns the process exit code."""
    print(f"planner_version: {planner_version()}\n")

    for expected, question in QUESTIONS:
        plan = plan_query(question)
        print(f"{question}\n  expected : {expected}\n  plan     : {plan.plan_type.value}")
        if plan.unsupported_reason:
            print(f"  reason   : {plan.unsupported_reason.value}")
        for task in plan.tasks:
            parts = [f"intent={task.intent.value}"]
            if task.company_mention:
                parts.append(f"company={task.company_mention!r}")
            if task.year is not None:
                parts.append(f"year={task.year}")
            if task.metric:
                parts.append(f"metric={task.metric.value}")
            if task.threshold_text:
                parts.append(f"threshold={task.threshold_text!r}")
            if task.direction:
                parts.append(f"direction={task.direction.value}")
            if task.question:
                parts.append(f"q={task.question!r}")
            print(f"  task     : {' '.join(parts)}")
        print(f"  rationale: {plan.rationale}\n")

    logger.info("planner_probe_complete", questions=len(QUESTIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())