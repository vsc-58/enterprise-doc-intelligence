# enterprise-doc-intelligence
AI pipeline for structured extraction and RAG querying of SEC 10-K filings


## Phase 1 — Acquisition & XBRL Ground Truth

Phase 1 acquires SEC 10-K filings, saves their clean text for downstream
extraction, and captures a verified financial answer key from each filing's XBRL
data. The corpus is 20 filings across 20 companies spanning fiscal years
2021–2023, tracked in SQLite.

`edgartools` fetches each 10-K and its clean text (`data/raw/{cik}_{year}.txt`);
the pipeline records each document in a SQLite tracking table and, separately,
reads five verified financial figures — total revenue, net income, total assets,
total liabilities, operating cash flow — from the filing's XBRL into
`data/eval/ground_truth.json`.

These XBRL figures serve one purpose: they are the **answer key** that scores the
LLM extraction pipeline in Phase 2. They never become the system's output — that
would reduce the project to a wrapper over a library that already did the work.
The system's structured fields come from an LLM reading unstructured text; XBRL
only *measures* how well it does.

Two correctness decisions shaped this phase:

**Strict concept reads, not convenience accessors.** edgartools' `get_revenue()`-
style accessors fuzzy-match XBRL concepts and *substitute a resembling value* when
the exact one is untagged — returning, for Amazon, "total liabilities and
stockholders' equity" (equal to total assets) in place of an untagged
total-liabilities concept, as a populated field that no null check would catch.
Ground truth instead reads each field by its exact XBRL concept, taking only the
consolidated row, and records **null** when a company genuinely didn't tag a
field. An answer key must never hold a confidently-wrong value; an honest null is
excluded from scoring.

**Fiscal-year selection by period-of-report.** A 10-K is selected by the calendar
year in which its fiscal period *ends*, read from `period_of_report` — not
edgartools' `year` filter, which keys on filing year and returns the wrong year's
filing for any December-fiscal-year company.

Result: 20 filings acquired, 0 failures. The answer key's only null values are
genuine — bank revenue (deliberately excluded as non-comparable to an
industrial's total revenue) and companies that don't tag a standalone
total-liabilities line. Full reasoning for every decision is in
`docs/decisions.md`.