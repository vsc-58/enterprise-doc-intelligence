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

## Phase 3 — Full Extraction Run

**Status:** Complete. 20/20 documents extracted, 0 failures. Numeric F1 1.00 in-sample, **0.90 held-out**. $0.10 billed ($0.05 replayed from cache).

Ran the Phase 2 winner (S3, rules + evidence) across the full corpus, wrote the results to SQLite with per-field citation lineage, and then scored the half of the corpus that was never used to select the strategy.

### What was built

| File | What it does |
|---|---|
| `src/eval/grounding.py` | **New.** Both citation checks, importable by the evaluator and the pipeline without a circular import |
| `src/eval/eval_set.py` | **New.** The Phase 2 selection set as one constant; defines the in-sample / held-out split |
| `src/storage/metadata_store.py` | `ExtractedRecord`, `ExtractionStatus`, `create_extracted_record()`, `latest_successful_records()` |
| `src/extraction/extractor.py` | `run_extraction_pipeline()` — grounding at write time, per-item error handling |
| `src/ingestion/sections.py` | Item 8 token ceiling replaced by an assembled-input budget check |
| `scripts/run_extraction.py` | **New.** Extracts every document where `is_extracted` is False |
| `scripts/run_heldout_eval.py` | **New.** Out-of-sample scoring, written to a separate results file |

### The headline result: 1.00 in-sample, 0.90 held-out

The evaluation set S3 was chosen on is ten documents. The other ten have XBRL ground truth and were never used for selection — so scoring them is a free out-of-sample measurement. That is the advantage of an automatic answer key: the eval-set size was a cost choice, not a labelling constraint, so the unused half is a real held-out set rather than a set nobody could afford to label.

| Field | In-sample | Held-out | n (held-out) |
|---|---|---|---|
| total_revenue | 1.00 | 0.875 | 8 |
| net_income | 1.00 | 1.00 | 9 |
| total_assets | 1.00 | 0.90 | 10 |
| total_liabilities | 1.00 | 0.889 | 9 |
| operating_cash_flow | 1.00 | 0.842 | 10 |
| fiscal_year_end | 1.00 | 1.00 | 10 |
| company_name | 0.80 | 0.50 | 10 *(see below — not an extraction result)* |

Four wrong values and one missing, across 46 scored numeric fields.

**I committed in writing, before seeing these numbers, that the prompt would not be revised in response to them.** Changing it now would turn the held-out set into a second dev set and leave no out-of-sample claim at all; recovering one would need freshly acquired documents. Deciding beforehand is what makes it a commitment rather than a rationalisation after the fact.

### Failure analysis

#### Three of the four wrong values are the unit-scale trap — on banks

This project names unit scale as its top silent failure: XBRL ground truth is in raw dollars, the filing text prints scaled values, and every extraction prompt instructs the model to return actual dollars. On the ten documents I selected the strategy on, that rule never failed. On the held-out ten it failed three times, both documents being banks.

| Document | Field | Extracted | Truth | Off by |
|---|---|---|---|---|
| BAC 2022 | total_assets | 3,051,375,000 | 3,051,375,000,000 | ×1,000 |
| BAC 2022 | total_liabilities | 2,778,178,000 | 2,778,178,000,000 | ×1,000 |
| GS 2021 | operating_cash_flow | 921,000 | 921,000,000 | ×1,000 |

Digits identical every time — the model read the figure correctly and scaled it wrong. The cause is the header:

- BAC: `Dollars in millions, except per share information; shares in thousands` — two scales in one header, and the model took the thousands.
- GS: `$ in millions` — terse and unpunctuated, unlike the `(In millions, except share amounts)` form the eval-10 all used.

Every eval-10 header was a conventional single-scale phrasing, so the actual-dollar-units rule had only ever been tested against the easy case. **That is the argument for a held-out set in one example.**

A prompt revision for multi-scale headers is a legitimate follow-up — as a new experiment against a new held-out set, documented as unvalidated until that set exists.

#### The fourth is a concept disagreement, not a misreading

XOM returned 398,675M ("Sales and other operating revenue") where the key holds 413,680M ("Total revenues and other income"). The printed income statement confirms the key: 413,680 is the total. The model took the line that most literally says *revenue*. Recorded as wrong; the key was not changed.

#### The one missing value is the right behaviour

BAC's operating cash flow is negative (−6,327M). The model returned null rather than a wrong figure. A `missing` costs recall only; a `wrong` costs precision *and* recall. In financial extraction an admitted gap beats a confident error, and the metric design says so explicitly.

#### company_name 0.50 is string normalisation, not extraction

All five mismatches are EDGAR registrant names versus what the filing cover actually says:

| Extracted (cover page) | Ground truth (EDGAR) |
|---|---|
| Bank of America Corporation | BANK OF AMERICA CORP /DE/ |
| The Goldman Sachs Group, Inc. | GOLDMAN SACHS GROUP INC |
| QUALCOMM Incorporated | QUALCOMM INC/DE |
| Exxon Mobil Corporation | EXXON MOBIL CORP |
| salesforce.com, inc. | Salesforce, Inc. |

The `/DE/` suffix is an EDGAR state-of-incorporation marker that appears in no filing. Phase 2 reached the same conclusion at 0.80; the held-out half is bank- and energy-heavy, where legal names diverge hardest from EDGAR's abbreviated form.

**Salesforce is the sharpest case: the model is right and the answer key is stale.** The FY2021 filing was made under `salesforce.com, inc.`; the company renamed in 2022, and EDGAR's registrant name is current, not as-filed. Recorded, not hand-edited — an answer key adjusted because it disagreed with the model is no longer independent.

### Citation lineage: 8 defects in 100 fields, on values that were all correct

S3 won the Phase 2 bake-off on operational grounds, not score — S1 and S4 tied it at 1.00. The tiebreaker was that S3 returns a verbatim source line per figure, which makes lineage checkable at write time. Phase 3 is where that paid.

Two independent checks, both stored per field:

- **`evidence_present`** — is the cited line real? Catches fabricated citations.
- **`evidence_consistent`** — does the cited line contain the value? Catches misattribution.

Across the 20-document corpus they found eight defects in 100 fields, in three patterns. **Every reported value was correct.** None of these is visible in a precision/recall table.

**Derived value cited to the adjacent row (4 — AMZN, INTC, TGT, WMT).** These filers have no standalone total-liabilities line in XBRL, which is why their ground truth is null. The model computes assets minus equity — verifiable on Target: 51,248 − 30,940 = 20,308 — and cites "Total liabilities and stockholders' equity," a row containing neither figure. Phase 1 refused to derive this value for the answer key because it would compare two computations rather than check extraction against a filed fact. The model derives it anyway. These fields score `n_a`, so no accuracy metric could ever have shown it.

**Concept straddle (2 — INTC, TSLA).** INTC returned net income 1,689 (`us-gaap:NetIncomeLoss`) and quoted the line printing 1,675 (`ProfitLoss`). TSLA returned 12,556, quoted 12,587. Both scored *correct*: the model picks the right concept and cites its neighbour.

**Row completed from another statement line (2 — JPM, PFE).** JPM cited `Total liabilities 3,547,515 3,373,411 3,743,567`. The first two figures are correct FY2023/FY2022 liabilities; the source row ends after them with a footnote marker. Searching the filing, 3,743,567 occurs three times, every occurrence in a *total assets* context — it is FY2021 total assets. The model completed a three-year table row with a figure from a different line. PFE cited the operating-activities section *header* (`Net cash provided by/(used in) operating activities:`) with the totals appended from further down the statement.

Reading JPM's value alone, the extraction is flawless — 3,547,515 matches ground truth exactly. Only the citation reveals it.

#### What the instruments cannot see

`evidence_consistent` tries multiple scale multipliers when matching a value against its line, because the filing prints millions and the model reports dollars. It therefore matched BAC's 3,051,375,000 against the printed 3,051,375 and **passed a figure that was wrong by a factor of 1,000.** It is structurally blind to unit errors.

| Check | Catches | Blind to |
|---|---|---|
| `evidence_present` | synthesised or fabricated quotes | wrong values inside real lines |
| `evidence_consistent` | miscitation | unit scale |
| XBRL ground truth | value errors including scale | only 5 fields, only where XBRL tags exist |

Three instruments, three disjoint blind spots. Knowing which is which is the point.

### Two design changes forced by what the data showed

**The Item 8 token ceiling was removed.** Building sections across the full corpus fired the 60,000-token ceiling for the first time — three times (PFE, GS, BAC), and PFE's fallback exceeded the model's context window entirely, so the document could not be extracted at all.

Boundary inspection showed **all three slices were genuine Item 8**: correct starts (auditor's report or statement index) and tails inside the notes, with no Item 9A, signatures, or Part III. The evidence that had set the ceiling turned out to be a measurement artifact — Phase 2 recorded GS's Item 8 as "larger than its whole filing" (119,591 tokens vs 102,626), but the two come from different edgartools text paths and the item view line-breaks every table cell, tokenising ~58% higher. In characters the slice is 491,175 against a filing of 587,294: a proper subset, as it must be.

Three fires, zero true positives, on evidence comparing two things that were never comparable. The ceiling was replaced by the constraint that actually matters — does the assembled input fit the context window. The character floor is kept: it has four confirmed true positives where the Item 8 anchor lands on the table-of-contents entry.

Corpus provenance after the change: 15 `edgartools_item8`, 4 `fallback_fulltext`, 1 `fallback_fulltext_oversize`.

**`business_description` and `primary_risk_factors` are no longer written to the store.** With cover + Item 8 as the input, 15 of 20 documents contain no Item 1 or Item 1A at all — yet the fields came back populated on 5 of 9 clean-Item-8 documents. What the model found was Item 8 note content. For Apple, `primary_risk_factors` returned "Concentrations in the Available Sources of Supply of Materials and Product," "Legal proceedings and claims that have arisen in the ordinary course of business," and "Uncertain tax positions" — note headings from Note 1, the contingencies note, and the tax note. Real text, correctly located, **wrong field**. Apple's actual Item 1A top risks are supply-chain disruption, competition, and dependence on third-party manufacturing.

Neither field is scored and neither is grounding-checked, so this would have gone into the export CSV unmarked. Both fields remain in the schema — removing them would change the tool definition sent to the model and silently invalidate every cached response — and are dropped at the write. Item 1 / Item 1A slicing belongs in Phase 4, where the same text is already being chunked.

### Storage design

`extracted_records` holds **one row per attempt**, not per document, so a failure and its eventual fix both survive for diagnosis. `latest_successful_records()` is the single canonical read — the alternative is three consumers each implementing "latest" slightly differently.

Three statuses: `success`, `extraction_failed` (the call completed and was billed but the output failed validation — the row is written so it is diagnosable), `technical_failed` (the call never completed). Only `success` sets `Document.is_extracted`, so both failure kinds are picked up by a re-run. This mirrors the Phase 2 cache rule: a timeout is not a result, a validation failure is.

Evidence is stored twice on purpose. `evidence_json` is the verbatim audit record and survives a change to the field set; the flattened `*_source_line` columns are the readable, filterable, exportable projection — a JSON blob is opaque in DB Browser and unreadable in the CSV a reviewer opens.

`prompt_version` is derived from the prompt template's hash rather than hand-maintained, so it cannot drift from the prompt it names, and it joins straight to the cached raw output that produced the record.

### Run

```bash
python -m scripts.build_sections          # all 20, or --eval-only / --held-out
python -m scripts.run_extraction --dry-run
python -m scripts.run_extraction
python -m scripts.run_heldout_eval
```

### Cost

| | |
|---|---|
| Billed this phase | $0.1045 (688,281 tokens) |
| Replayed from cache | $0.0523 — the eval-10, whose prompt, model and sliced input were unchanged |
| Project total to date | under $0.40 |

### Limitations

- n=10 per split. Enough to separate 1.00 from 0.90; not enough to attach a confidence interval to either.
- The three unit failures are two documents. The pattern is clear but the sample is not.
- Pure-play bank revenue is excluded from the answer key (no comparable "total revenue" concept), so held-out denominators are smaller than in-sample.
- `evidence_consistent` cannot detect unit errors, by construction.
- `evidence_present` proves a line exists, not that it supports the value attached to it.
- A ground-truth value that is wrong in the same way the model is wrong would score correct and stay invisible. The harness catches disagreement, not shared error.
