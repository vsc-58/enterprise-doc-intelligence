AI pipeline for structured extraction and RAG querying of SEC 10-K filings

## Phase 1 — Acquisition & XBRL Ground Truth

Phase 1 acquires SEC 10-K filings, saves their clean text for downstream extraction, and captures a verified financial answer key from each filing's XBRL data. The corpus is 20 filings across 20 companies spanning fiscal years 2021–2023, tracked in SQLite.

`edgartools` fetches each 10-K and its clean text (`data/raw/{cik}_{year}.txt`); the pipeline records each document in a SQLite tracking table and, separately, reads five verified financial figures — total revenue, net income, total assets, total liabilities, operating cash flow — from the filing's XBRL into `data/eval/ground_truth.json`.

These XBRL figures serve one purpose: they are the answer key that scores the LLM extraction pipeline in Phase 2. They never become the system's output — that would reduce the project to a wrapper over a library that already did the work. The system's structured fields come from an LLM reading unstructured text; XBRL only measures how well it does.

Two correctness decisions shaped this phase:

**Strict concept reads, not convenience accessors.** edgartools' `get_revenue()`-style accessors fuzzy-match XBRL concepts and substitute a resembling value when the exact one is untagged — returning, for Amazon, "total liabilities and stockholders' equity" (equal to total assets) in place of an untagged total-liabilities concept, as a populated field that no null check would catch. Ground truth instead reads each field by its exact XBRL concept, taking only the consolidated row, and records null when a company genuinely didn't tag a field. An answer key must never hold a confidently-wrong value; an honest null is excluded from scoring.

**Fiscal-year selection by period-of-report.** A 10-K is selected by the calendar year in which its fiscal period ends, read from `period_of_report` — not edgartools' `year` filter, which keys on filing year and returns the wrong year's filing for any December-fiscal-year company.

Result: 20 filings acquired, 0 failures. The answer key's only null values are genuine — bank revenue (deliberately excluded as non-comparable to an industrial's total revenue) and companies that don't tag a standalone total-liabilities line. Full reasoning for every decision is in `docs/decisions.md`.

---

## Phase 2 — Extraction Schema, Prompt Strategies & Evaluation

Phase 2 builds the LLM extraction pipeline and measures it. Five prompting strategies were run over ten filings, scored per field against the Phase 1 XBRL answer key, and the winner selected with documented evidence. Full results in `docs/prompt_strategy_results.md`.

**Result: naive prompting scores 0.50 on numeric fields; three explicit rules take it to 1.00. The winning strategy (evidence-based) matches that accuracy and additionally returns a verified source line for every figure, at 1.3% higher cost.**

### Section slicing — a feasibility constraint, not an optimisation

Extraction prompts receive Item 8 (the financial statements) plus a cover-page block, sliced in Python before any API call. This is not a cost optimisation: token-counting the corpus showed PFE's FY2023 filing at 136,215 tokens against GPT-4o-mini's 128,000-token window. Full-document extraction is impossible for that filing, and an instruction inside the prompt cannot reduce what was already paid to put into the prompt.

edgartools exposes item-level access (`tenk["Item 8"]`), but running it across the corpus showed it silently failing on 5 of 20 filings — Item 8 collapsing to a 92-token heading for JPM, 54 for NFLX, while GS over-captured at 119,591 tokens, larger than its entire filing. The slicer therefore accepts a slice only if it passes a size plausibility gate (edgartools' own 26,136-character floor; a 60,000-token ceiling), and otherwise falls back to the full filing text. Every section carries a provenance tag, so the effect of input quality is measurable at scoring time rather than guessed.

### The five strategies

| ID | Name | What varies |
|---|---|---|
| S0 | Naive | Task only — no unit, column, or null-handling rules |
| S1 | Rule-based | Adds three explicit rules |
| S2 | Rules + few-shot | S1 plus two worked examples |
| S3 | Rules + evidence | S1 plus quote-the-source-line-then-normalise |
| S4 | Rules + orientation | S1 plus structural guidance |

The three rules: report figures in **actual dollar units** (the filing prints `$394,328` under an "in millions" header; XBRL truth is in raw dollars); take the **current fiscal year, first data column**; return **null rather than guessing**.

Input text is held identical across all arms — the prompt is the only variable.

### Results

| Field | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| total_revenue | 0.50 | **1.00** | **1.00** | **1.00** | **1.00** |
| net_income | 0.50 | **1.00** | 0.90 | **1.00** | **1.00** |
| total_assets | 0.50 | **1.00** | **1.00** | **1.00** | **1.00** |
| total_liabilities (n=7) | 0.43 | **1.00** | **1.00** | **1.00** | **1.00** |
| operating_cash_flow | 0.50 | **1.00** | **1.00** | **1.00** | **1.00** |
| company_name | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |
| fiscal_year_end | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| Mean relative error | 5.14e-01 | 3.33e-08 | 2.00e-02 | 3.33e-08 | 3.33e-08 |
| Grounding rate | – | – | – | **1.00** | – |
| Cost (10 docs) | $0.0546 | $0.0550 | $0.0552 | $0.0557 | $0.0552 |

Precision equals recall on every cell — the model never asserted a wrong value where it could have returned null. Fields with null ground truth are scored `n_a` and excluded from the denominator.

Three findings:

- **The rules are the entire intervention.** Naive extraction returns roughly half the figures at the wrong scale — the unit trap firing. Three sentences of instruction close it, for $0.0004 across ten documents. Everything layered above the rules is flat.
- **Few-shot examples degraded extraction.** S2 is the only arm that got a value wrong, with a mean relative error 600,000× S1's.
- **Missing instruction and degraded input compound.** On the one fallback document (NFLX, full 52,744-token filing), naive scored 0/5 while every ruled arm scored 5/5. The ruled arms show no fallback penalty at all.

`company_name` caps at 0.80 across every arm including the naive one — invariance across arms means it isn't measuring extraction. The failures are "MICROSOFT CORPORATION" against XBRL's "MICROSOFT CORP". A synonym table would push it to 1.00; it wasn't added, because fuzzy string matching is what corrupted the ground truth in Phase 1.

### The harness caught two bugs in itself

**The answer key was wrong.** The first complete run showed `total_revenue` at 0.90 — identically across every strategy, failing on the same document. Uniformity was the tell: a prompt problem varies by prompt, a ground-truth problem does not. Walmart's income statement prints net sales 605,881 + membership and other income 5,408 = total revenues 611,289; the answer key had captured the component, because the Phase 1 concept-priority list preferred `RevenueFromContractWithCustomerExcludingAssessedTax` over `us-gaap:Revenues` and Walmart tags both. The Phase 1 rule was "true synonyms only" — which holds for filers tagging one concept, and fails silently for a filer tagging both, where the two are subset and superset. Fixed in the code and the key regenerated with a verified diff, never hand-edited: an answer key adjusted because the model disagreed is no longer independent. Three documents corrected.

**The grounding check was miscalibrated.** It first read 0.96. Both failures were on values that scored correct — the model had written `Total assets $85,501 $82,896` where the source text holds `total assets$85,501 $82,896`. The text extractor runs labels into their values, so a faithful quote could never contain that space; the model had normalised the mangled spacing back to readable form. The check was flagging legibility as fabrication. Loosened to ignore whitespace and currency symbols while still catching invented labels or absent digits, it reads 1.00.

### Why the evidence strategy wins

S1, S3 and S4 are indistinguishable on accuracy, so the tie breaks on operational value. S3 returns a verbatim source line per figure, all of which were verified present in the source. That enables an automatic grounding check at write time — a citation absent from the source means the value was fabricated, and the record is flagged rather than trusted. It makes human review economical (one quoted line instead of a 100-page filing), and it carries cell-level lineage from an unstructured source into the structured store. Ten filings cannot establish that S1 will keep scoring 1.00 on the next hundred; S3 makes each individual answer checkable regardless.

Total cost of the five-arm bake-off: approximately $0.28.

### Evaluation artifacts

| Path | Contents |
|---|---|
| `data/eval/ground_truth.json` | XBRL answer key per document |
| `data/eval/results.json` | Per-field precision/recall/F1 with denominators, provenance split, per-document scores |
| `data/eval/raw_outputs/{strategy}/` | Every model response, cached by content hash — the evidence behind the table |
| `docs/prompt_strategy_results.md` | Full write-up: method, results, limitations |

Cached outputs are keyed by a hash of strategy, model, prompt template and input text, so a changed prompt or re-sliced input invalidates the cache automatically. Scoring re-runs read from disk at zero API cost — which is how the ground-truth fix and the grounding recalibration were re-scored without paying for a single additional call.
