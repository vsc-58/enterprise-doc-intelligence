"""
scripts/diagnostic/probe_chain.py — DIAGNOSTIC (print() allowed).

Exercises the narrative chain against the real store and API before it is wired
into the executor (step 6). Five single-company cases, one per path the chain can
take, plus the corpus-wide headline question.

What to look for:
  - Apple MD&A: figures_masked > 0 (the revenue table is in context), claims
    cited, no "[figure withheld]" in claims (redaction is the backstop; firing
    means masking missed a format).
  - Intel MD&A: section_absent with no LLM call.
  - Apple net income as a narrative task: figure_requested.
  - Apple stock price: not_addressed (the Phase 4 junk retrieval at 0.4955).
  - META overview: answered from Item 1A via the alias table (D40).
  - Corpus-wide: one finding per company; JPM and XOM not_searchable for the
    market-risk question; the negative control should be mostly not_found.

Cost: ~45 gpt-4o-mini calls, about $0.02.

Usage:
    python -m scripts.diagnostic.probe_chain
"""

from __future__ import annotations

import asyncio

from src.query.resolve import load_index
from src.query.schemas import Intent, NarrativeResult
from src.rag.chain import answer_for_company, chain_version, refusal_message, survey_companies
from src.rag.retriever import CompanyRef
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

SINGLE: list[tuple[str, str, Intent]] = [
    ("AAPL", "Why did Apple's net sales grow?", Intent.MANAGEMENT_COMMENTARY),
    ("INTC", "Why did Intel's revenue change?", Intent.MANAGEMENT_COMMENTARY),
    ("AAPL", "What was Apple's net income?", Intent.MANAGEMENT_COMMENTARY),
    ("AAPL", "What is Apple's current stock price?", Intent.COMPANY_OVERVIEW),
    ("META", "What does Meta do?", Intent.COMPANY_OVERVIEW),
]
# (question, intent, expectation). The second is a NEGATIVE CONTROL: a topic few
# companies in this corpus discuss. The headline question came back 17/18
# SUPPORTED, which is either correct (interest rate risk is near-universal in
# Item 7A) or a lenient verdict rule saying yes to everything. Only a question
# whose true answer is sparse can tell those apart.
SURVEYS: list[tuple[str, Intent, str]] = [
    ("Which companies flagged interest rate risk?", Intent.MARKET_RISK,
     "expect most SUPPORTED"),
    ("Which companies hold bitcoin or other digital assets?", Intent.RISK_FACTORS,
     "expect FEW SUPPORTED — if most say yes, the verdict rule is too lenient"),
]


def show(result: NarrativeResult, subject: str) -> None:
    """Print a result."""
    print(f"   masked={result.figures_masked} redacted={result.figures_redacted} "
          f"invalid_citations={result.invalid_citations}")
    if result.refusal is not None:
        message = refusal_message(result.refusal, subject, result.sections_searched)
        print(f"   REFUSAL {result.refusal.value}: {message}")
    for claim in result.claims:
        cites = ", ".join(f"{s.section}#{s.chunk_index}" for s in claim.sources)
        print(f"   - {claim.text}  [{cites}]")
    for finding in result.findings:
        line = finding.claim.text if finding.claim else ""
        print(f"   {finding.ticker:<6} {finding.status.value:<15} {line[:110]}")


async def main() -> int:
    """Entry point. Returns the process exit code."""
    print(f"chain_version={chain_version()}")
    companies = {
        e.ticker: CompanyRef(ticker=e.ticker, company_name=e.company_name,
                             fiscal_year=e.filing_year)
        for e in load_index().entries if e.is_embedded
    }

    for ticker, question, intent in SINGLE:
        company = companies[ticker]
        print(f"\n{'=' * 78}\n{ticker} | {intent.value} | {question}")
        result = await answer_for_company(question, intent, company)
        show(result, display_name(company.company_name, company.ticker))

    for question, intent, expectation in SURVEYS:
        print(f"\n{'=' * 78}\nCORPUS-WIDE | {intent.value} | {question}\n   ({expectation})")
        result = await survey_companies(question, intent, list(companies.values()))
        show(result, "the companies in this corpus")
        supported = sum(f.status.value == "supported" for f in result.findings)
        searchable = sum(f.status.value in ("supported", "not_found") for f in result.findings)
        print(f"   SUPPORTED {supported}/{searchable} searchable")

    logger.info("probe_chain_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))