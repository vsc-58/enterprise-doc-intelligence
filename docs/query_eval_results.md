# Query-Path Evaluation Results

Status: complete (2026-10-15). Covers routing, execution, retrieval (section 7)
and attribution support (section 8).

All numbers are means over **3 runs**: temperature 0 is not deterministic, and
the same question can get a different plan from one run to the next.
95% intervals are Wilson intervals with **n = questions**: repeated runs measure
the model's variability on the same questions and do not narrow the interval.
**Held-out is report-only**: no prompt was revised in response to a held-out
result, and every revision is tied to a named dev miss. Revising on held-out
results would turn them into a second dev set and leave no out-of-sample claim.

---

## 1. Headline

| Measure | Held-out | Dev |
|---|---|---|
| Routing, v1 — **clean out-of-sample** | **82.5%** [68–91] | 79.6% [68–88] |
| Routing, v3 (final) | 88.9% [76–95] | 90.0% [81–95] |
| End-to-end correct outcome (v3, adjudicated) | 89.7% [77–96] | 93.5% [85–97] |
| Downstream with gold plan (oracle, all 91 executable, adjudicated) | 100% [90–100] (raw 91.7%) | 100% [94–100] |
| Deterministic refusals (section absent, year not held, non-corpus, ambiguous, flagged figure, fallback) | 100% | 100% |
| Figure-guard redactions (masking misses), 9 runs | 0 | 0 |
| Invalid citations, 9 runs | 1 in total, detected by the citation check | |
| Retrieval: relevant chunk in the top 5 (strict, n = 18 specific questions) | 100% [68–100] | 100% [72–100] |
| Retrieval: capped recall@5 / precision@5 (strict) | 79% / 50% | 59% / 46% |
| **Attribution: claims supported by their cited chunks** (validated judge, 322 claims; 93.2% overall) | **91.1%** [85–95] | **94.5%** [90–97] |

Reading it: the downstream system is exact when given the right plan; the
remaining end-to-end errors are almost all routing. Retrieval always puts a
relevant chunk in front of the chain, but often not all of them (Walmart's
liquidity text ranks 9–18, section 7.4). About 7% of answer sentences claim more
than their cited text says — mostly added inference, rarely invention (section
8.3). v1's held-out routing number is the only one untouched by tuning; v2 and
v3 held-out numbers are reported beside it, not instead of it (section 4.4).

---

## 2. Eval Set

`data/eval/query_eval_set.json` — 109 questions, frozen at commit `a15727e`,
SHA-256 `c3af4d55…603e` (line endings normalised). Harnesses refuse to score
against any other hash.

| Source | Dev | Held | Notes |
|---|---|---|---|
| Earlier development questions, relabelled to the 9 intents | 33 | 17 | 5 held questions that had appeared verbatim in earlier probes moved to dev |
| Written by a separate LLM, user phrasing (typos, tickers, vague) | 25 | 25 | Stratified split, fixed seed |
| Hard cases written with system knowledge | 9 | 0 | Dev only |

Each row carries a strict label, accepted alternatives (lenient scoring), the
expected outcome per task (derived from `data/eval/section_inventory.json`, not
intuition: a label set by intuition measures the labeller, not the system), and
a narrative-set flag (23 questions) for retrieval and
attribution. Contamination controls: questions seen in earlier probes moved to
dev; the prompt example that was eval question R11 removed; every prompt
revision scanned against held-out questions for shared 3-word phrases.

---

## 3. Method

**Routing** (`scripts/run_routing_eval.py`): planner only. Each plan is reduced
to plan type, unsupported reason, and per task (intent, company resolved to a
ticker, metric, year, direction); task order ignored. Strict = matches the first
label; lenient = matches any accepted alternative. Each miss is classified at
the first level that differs. A mention that fails to resolve for a held
company is flagged as a resolver gap rather than charged to the planner.

**Execution** (`scripts/run_execution_eval.py`), two modes:
- *oracle* — the label's strict plan executed directly: downstream with routing
  assumed perfect;
- *end to end* — the planner's plan: what a user receives.

Every task is reduced to an outcome the code emits (resolver failure,
NarrativeRefusal, data-quality verdict, section fallback) and compared with the
label. A question is correct only if every task matches (and, end to end, it
routed correctly). The gap between modes is the cost of routing errors.

---

## 4. Routing

### 4.1 Versions

| planner_version | Held strict | Held lenient | Dev strict | Dev lenient | Plan varied (of 109) |
|---|---|---|---|---|---|
| v1 `0ed3e740` | 82.5% | 87.3% | 79.6% | 84.1% | 5 |
| v2 `f5252664` | 88.1% | 92.9% | 87.6% | 90.5% | 2 |
| v3 `d4bc8e36` (final) | 88.9% | 92.9% | 90.0% | 93.0% | 5 |

Few questions change plan between runs: the misses are systematic, not noise.

### 4.2 v1 miss patterns

| | Pattern | Dev evidence | User impact |
|---|---|---|---|
| P1 | "Which companies…" narrative planned unsupported / out of scope | N32, N33, H01 | Wrong refusal; the corpus-wide survey mostly unreachable |
| P2 | Figures stated in text (R&D, dividends) → field_not_queryable | N41 | Wrong refusal |
| P3 | "compare X and Y" → aggregation | N15 | Wrong refusal |
| P4 | Ranking within named companies → one ranking per company | N10 | **Wrong answer**: the corpus-wide winner, three times |
| P5 | "last year" → a year, or cross_year | N08 | Wrong refusal |
| P6 | "net sales" → net_income | R11 | **Wrong figure** |

P4 was also fixed structurally (`executor.scoped_to_company`): a ranking or
filter that names a company runs as that company's lookup. It is the only
pattern detectable from the plan's shape alone, so it is the only one enforced in
code; P6 ("net sales" → net_income) is a well-formed plan that only the prompt
and the eval can catch.

### 4.3 Paired comparison — fixed vs broken

| Step | Fixed | Broken |
|---|---|---|
| v1 → v2 held | N01, N13, N34, N35, N42 | N14, N50 |
| v1 → v2 dev | R11, N08, N10, N15, N29, N32, N33, H08 | R07, N49, R05 (1 run) |
| v2 → v3 | R05, R07, N14, R50 | N01, N32, N35 (some runs) |

v2's regressions were caused by its own wording ("never financial_ranking",
"not a clarification"): prohibitions generalise beyond their target. v3
restated both as positive discriminators (a filter needs a threshold; a question
about companies in general differs from one about a specific, unnamed company). A
stopping rule declared before the v3 run made it final: each further iteration on
dev, with held-out misses visible, would erode the held-out claim further.

### 4.4 What can be claimed

- Clean out-of-sample routing: **82.5% held-out, interval 68–91%**.
- v1 → v3 on held-out: 4 fixed (N13, N34, N35, N42) against 1 broken (N50).
  At n = 42 this is **not** statistically significant (≥4 of 5 changes in one
  direction by chance ≈ 19%). The gain is consistent with the dev evidence but
  not proven by the held-out set, and held-out misses had been seen.
- Dev v1 → v3: 9 fixed, 1 broken — in-sample, since dev is where tuning happened.

### 4.5 Remaining routing misses (documented, not chased)

| Question | v3 behaviour | User impact |
|---|---|---|
| N49, N50 ("that bank", "its strategy") | Planned as a 20-company survey | **Wrong answer** + ~20 calls; plan is indistinguishable from a legitimate survey |
| N41, N43 (R&D, dividends) | field_not_queryable | Wrong refusal; three prompt attempts failed — the rule that narrative answers may quote figures other than the five stored ones needs a schema signal or a stronger planner |
| N01 ("last year") | Year 2022 in some runs | Recoverable refusal |
| N32, N35 | Survey split into 2–3 tasks | Duplicate surveys, double cost |
| H01 (bitcoin) | company_overview survey | Searches Item 1; Tesla's holding (Item 7) missed |

---

## 5. Execution Outcomes

### 5.1 Oracle vs end to end on identical questions

The oracle was rerun after each narrative task of a multi-task label was given
its own question (6.2), so hybrids are now included.

| Questions | Split | Oracle raw (per run) | Oracle adjudicated | End to end (per run, adjudicated) |
|---|---|---|---|---|
| All executable | held (36) | 33 / 33 / 33 | 36 / 36 / 36 | 32 / 33 / 33 |
| | dev (55) | 55 / 55 / 55 | 55 / 55 / 55 | 53 / 52 / 52 |

The three raw held misses are label errors, not system errors: R35 and N14
(6.1) and N20, labelled "answered" although no relevant chunk exists in the
corpus (section 7.5). Every question that passes the oracle but fails end to end
is a routing miss already listed in 4.5: N01, N32, N35, N41, N43, H01.

### 5.2 Outcomes decided in code — all exact

| Expected outcome | Oracle task-runs matched |
|---|---|
| section_absent (MD&A or market risk not captured) | 15 / 15 |
| year_not_held | 12 / 12 |
| non_corpus_company | 12 / 12 |
| answered_flagged (disputed figure disclosed) | 24 / 24 |
| fallback_answered (market risk answered from Item 1A when Items 7 and 7A were not captured; R31 and H02) | 6 / 6 |
| ambiguous_company (end to end) | 3 / 3 |

### 5.3 Figure guard and citations

| Run | Oracle masked | Oracle redacted | E2E masked | E2E redacted | Invalid citations |
|---|---|---|---|---|---|
| 1 | 31 | 0 | 22 | 0 | 0 |
| 2 | 31 | 0 | 19 | 0 | 0 |
| 3 | 31 | 0 | 19 | 0 | 0 |
| Oracle rerun (per-task questions), runs 1–3 | 28 / 28 / 28 | 0 | — | — | 0 / 1 / 0 |

The redaction backstop never fired: masking caught every stored figure before
the model saw it. One citation in nine runs pointed at a block number that did
not exist; the citation check detected and counted it. First occurrence, logged.

### 5.4 Corpus-wide surveys (oracle, run 1)

| Question | Supported | Not found | Not searchable | Fallback |
|---|---|---|---|---|
| H08 interest-rate risk (headline demo question) | 19 | 1 | 0 | JPM, XOM via Item 1A |
| H05 cybersecurity risk | 19 | 1 | 0 | — |
| N34 climate / environmental regulation | 17 | 3 | 0 | — |
| N32 supply chain | 16 | 4 | 0 | — |
| N35 AI risk | 13 | 7 | 0 | — |
| N33 inflation | 12 | 8 | 0 | — |
| H01 bitcoin holdings (known answer) | 1 (TSLA) | 16 | 3 | — |

H01 is the recall probe: with the gold intent the survey finds Tesla's holding
in Item 7 and returns not-found for Goldman (facilitation, not holding). N35's
13/20 for AI risk in FY2021–23 filings may reflect lenient verdicts (generic
technology risk counted as AI); the judge validation in section 7 tests this.

---

## 6. Adjudications and Instrument Issues

Labels are frozen and never edited. Where audit found a label wrong, the raw
score is reported as primary and the adjudicated score beside it.

### 6.1 Label errors

| ID | Label said | Correct | Why |
|---|---|---|---|
| N29 | single task only | two same-intent tasks also acceptable | Accepted for N22, not here — inconsistent labelling |
| N20 | answered | not_addressed (honest refusal) | Intel's two strategy chunks are a 10-K cross-reference index; no relevant chunk exists (7.5) |
| N44 | ratio_or_derived | field_not_queryable defensible | Debt-to-equity needs equity, which is not stored |
| R35 | BAC revenue answered | answered_flagged | BAC's four checkable fields split 2–2 between two scales, so all four are flagged scale-ambiguous; revenue is disputed |
| N14 | answered | answered_flagged | The lowest row is Intel, whose net income is disputed |

### 6.2 Harness artifact — oracle hybrids (fixed)

The first oracle passed the whole question to each narrative task, so a hybrid
("Describe Tesla's business and give its total revenue") sent the figure clause
to the chain, which correctly refused it. Five oracle misses were this artifact.
The fix gives each narrative task of the 13 multi-task labels its own clause, kept in
`oracle_params.json` beside the frozen labels; a missing clause now raises. After
the fix every hybrid passes the oracle.

### 6.3 Genuine downstream misses

- **N20** (Intel turnaround strategy): resolved as a label error (6.1). Both
  of Intel's eligible chunks are the 10-K cross-reference index, so the refusal
  is the correct answer; the defect is the corpus (Intel's Item 1 and MD&A text
  were not captured).
- **N41** (Apple R&D): the chain returned figure_requested in 1 of 3 runs,
  treating R&D as a reserved figure. The rule that narrative answers may quote
  figures other than the five stored ones is fragile at both the planner and
  the chain.

---

## 7. Retrieval

Scorer: `scripts/run_retrieval_eval.py`; gold: `data/eval/retrieval_gold.json`
(frozen by SHA-256; the scorer refuses an edited file); results:
`data/eval/retrieval_results.{json,md}`.

### 7.1 Why chunk-level gold

Retrieval filters on company and section, so "right company, right
section" is ~100% by construction and measures the filter. What can fail is which
chunks of the right section reach the 5 the chain reads. That needs chunk-level
relevance labels.

### 7.2 Gold

- 24 questions: the 23-question narrative set plus N20 as an audit.
- Pool per question: the retriever's top 10 ∪ keyword hits (regexes written from
  the question before reading retrieved text, capped at 25). Pooled-but-unlabelled
  = not relevant; outside the pool = unjudged.
- Rule fixed before labelling: **rel** = a reader could write at least one sentence
  answering the question from this chunk alone, for this company; **border** = on
  topic but generic, a summary bullet, hypothetical or partial.
- Labels proposed with an evidence phrase each; every label reviewed by the
  developer (one change, R18). Two further rounds labelled every chunk that entered
  any top 10 unjudged, until none remained. 173 rel/border labels; 110 rel.

### 7.3 Results (strict = rel only; planner query = what production embeds)

| Questions | n | Relevant in top 5 | Capped recall@5 | Recall@10 | Precision@5 | First relevant rank |
|---|---|---|---|---|---|---|
| Specific, held-out | 8 | 100% [68–100] | 79% | 83% | 50% | 1.5 |
| Specific, dev | 10 | 100% [72–100] | 59% | 70% | 46% | 1.4 |
| Specific, all | 18 | 100% [82–100] | 68% | 76% | 48% | 1.4 |
| Broad (precision only) | 5 | — | — | — | 84% (92% lenient) | 1.0 |

Capped recall = relevant in top k / min(k, R): a question with 17 relevant chunks
cannot fit more than 5 in a top 5, and plain recall would call a perfect top 5 29%.
Recall is an upper bound (unpooled relevant chunks are unknown).

**Planner rewording vs the user's words**: identical rankings on 17 of 18 specific
questions; N19 improved (+20 points recall@5). Rewording neither helps nor hurts.
The one cost came from intent, not wording: the planner routes R24 (Pfizer
regulatory risk) to risk_factors, an accepted label, which searches Item 1A only
and makes 8 of its 19 relevant chunks unreachable (recall@10 70% → 50%).

Held-out scores higher than dev because the two hardest questions (R28, N29,
both Walmart MD&A) happen to be in dev; with n = 8 and 10 the gap means little.

### 7.4 Weakest questions, and the diagnosed cause

| Question | Recall@5 | Why |
|---|---|---|
| R28 Walmart liquidity | 20% | Liquidity chunks rank 9, 15, 17, 18 behind general MD&A |
| N19 Salesforce growth strategy | 20% (40% planner) | Strategy spread thinly across Item 1 |
| N29 Walmart inventory & capital needs | 40% | Same section, same crowding as R28 |
| R19 Netflix content investment | 50% | First relevant chunk at rank 5 |

A controlled test on R28 and N29 (dev) changed one side at a time:

| Condition | R28 recall@5 | N29 recall@5 |
|---|---|---|
| Baseline | 20% | 40% |
| Subsection heading prefixed to every chunk | **80%** | **60%** |
| Question rewritten in the filing's words | 60% | 40% |

Chunks deep inside "Liquidity and Capital Resources" no longer carry that heading.
Heading loss is the larger effect; vocabulary is partial. Headings were assigned by
a crude heuristic, so the effect is directional. The fix — contextual chunk headers
from the parser — needs a full re-embed and is logged as a next step, not built.

### 7.5 N20 audit

Both of Intel's eligible strategy chunks are the 10-K cross-reference index
("Item 1. Business … Pages 3–9"). No relevant chunk exists, so not_addressed is
the correct answer and the label is wrong (6.1).

### 7.6 Limitations

- n = 18 specific questions: a 100% top-5 hit rate has a lower bound of 82%.
- R24 (59 keyword hits) and R25 (34) were capped at 25 labelled: their recall is a
  weak upper bound; precision is the informative number there.
- Similarity is not salience: broad risk questions reach ~100% precision because
  every Item 1A chunk names a risk. Whether the top 5 holds the *main* risks is not
  measured (N25's rank 1 is J&J's COVID vaccine-programme uncertainty).

---

## 8. Attribution Support

Judge: `src/eval/attribution.py`; run: `scripts/run_attribution_eval.py`;
results: `data/eval/attribution_results.{json,md}`; labels:
`data/eval/attribution_labels.json`. The validation design and its pass criteria
were fixed before the judge ran, and amended once, before its first run (8.2).

### 8.1 What is measured

For each sentence of a narrative answer and the chunks it cites: do the chunks
support **every** detail (company, figures, dates, causes, qualifiers)? Binary
verdict, supporting span quoted. The judge (gpt-6-luna, function calling,
reasoning off) is a different model from the system (gpt-4o-mini) to reduce
self-preference. It sees the claim, the company it is about and the cited text —
not the question, so it rules on support, not relevance.

### 8.2 Judge validation — passed every criterion

Validation set: 20 calibration claims, 40 test claims (random, seed 43, ≤ 3 per
question) and 30 seeded mismatches built from test claims: 10 with an invented
detail written by hand, 10 citing a wrong chunk of the same company, 10 citing the
nearest chunk of a different company. All 30 seeds were read against their text to
confirm none was accidentally supported.

| Criterion (committed before any judge output) | Result | Bar |
|---|---|---|
| Seeded mismatches caught, overall | **30 / 30** | ≥ 90% |
| — invented detail / wrong block / wrong company | 10/10 · 10/10 · 10/10 | ≥ 80% each |
| Cohen's kappa vs reference, 40 test + 30 seeds | **0.97** (1.00 after adjudication) | ≥ 0.70 |
| Label flips across 3 runs (validation) | 3.3% | ≤ 5% |

Reference labels for the test claims were written by Claude — a different model
family from the judge, from claim and excerpts only — at the developer's request,
not by a human. The single judge–reference disagreement (T02, Tesla:
"continuous innovation in EV technology" is not in the text) was adjudicated by the
developer in the judge's favour, giving 40 / 40 agreement on real claims.

Why kappa, and why over test + seeds: 39 of 40 real claims are supported, so a judge
that always said "supported" would score 97.5% raw agreement while catching
nothing. Kappa discounts chance agreement. At that prevalence it is also unstable:
the one disagreement alone gives a test-only kappa of 0.66 at 97.5% agreement.
Adding the 30 known-bad seeds gives both classes enough items. This amendment was
made before the first judge run, for that statistical reason, and is recorded
rather than hidden: changing a criterion after seeing results would invalidate it.

Calibration (not gating): 17 / 20 agreement. The judge is stricter on wording than
the reference — it rejected "substantial" for the filing's "significant" (C20) —
so the headline below leans conservative.

### 8.3 Results — 322 claims, end-to-end run 1

| Group | Claims | Supported |
|---|---|---|
| **All** | 322 | **93.2%** [90–95] |
| Held-out questions | 123 | 91.1% [85–95] |
| Dev questions | 199 | 94.5% [90–97] |
| Single-company answers | 172 | 94.2% |
| Corpus-wide survey findings | 150 | 92.0% |

Intervals use n = claims; claims cluster within questions, so true uncertainty is
wider. 2 of 324 claims got no verdict (all three calls failed).

**What the 22 unsupported claims are** — mostly added inference, rarely invention:

| Pattern | Count | Example |
|---|---|---|
| Inference the text does not make | 10 | "…which reflects its focus on improving inventory management" (N29); "as part of its growth strategy" (N40) |
| Over-specific or narrowed detail | 6 | "particularly those used in AI applications" (N35 QCOM); "advertising business specifically" (N27) |
| Wrong attribution within the text | 2 | Netflix's operating-margin drivers reported as revenue drivers; company-wide FX impact reported as streaming-only (N28) |
| No support at all | 1 | Amazon "inflation may impact demand" — cited text never mentions it (N33) |
| Borderline: wording strictness or split runs | 3 | "significantly affect market value" (H08 META); H05 ADBE and H08 TGT split 1–2 across runs |

The N35 "AI risk" survey suspicion (5.4) is confirmed in one case: Bank of America's
text is about regulators' compliance focus on AI, not AI causing harm.

### 8.4 Caveats

- Reference labels are a second model's, not a human's; two LLMs can share a blind
  spot. The seeds, wrong by construction, are the check that depends on neither.
- One real unsupported claim among 40 test claims: the judge's recall on *real*
  failures is barely measured and rests on the seeds.
- Flip rate on the 322 scored claims was 5.6%, just above the 5% bar that was set
  for the validation set; majority over 3 runs is the label.
- For 104 of 300 SUPPORTED verdicts the judge's quoted span is not verbatim in the
  text (paraphrased or reformatted quotes). Recorded, not used for verdicts.

---

## 9. Known Limitations and Next Steps

| Item | Status |
|---|---|
| Chunks lose their subsection heading (R28, N29) | Diagnosed (7.4); fix = contextual chunk headers, needs re-embed |
| Unnamed-company questions planned as surveys (N49, N50) | Wrong answers; needs a structural guard, not a prompt rule |
| R&D and dividend questions refused (N41, N43) | Non-stored-figure rule fragile (6.3); needs a schema signal |
| ~7% of answer sentences over-interpret their source | Measured; candidate fix = run the judge as a post-generation check |
| Corpus gaps: Intel/JPM/XOM MD&A, Tesla Item 7A, Intel Item 1 is an index | Logged; acquisition-layer fixes |
| Planner on a stronger model | Not run; would share the judge's model |

---

## 10. Reproduce

```bash
python -m scripts.diagnostic.dump_section_inventory          # section inventory
python -m scripts.run_routing_eval                           # routing, 3 runs
python -m scripts.run_execution_eval                         # oracle + end to end, 3 runs
python -m scripts.run_retrieval_eval --routing data/eval/routing_results_d4bc8e36.json
python -m scripts.diagnostic.probe_heading_context           # heading-loss test (7.4)
python -m scripts.diagnostic.probe_judge                     # judge structured-output probe
python -m scripts.export_attribution_set                     # seeds + label sheet
python -m scripts.run_attribution_eval --execution data/eval/execution_results_d4bc8e36_b1d8ee5c.json
pytest                                                       # unit tests (no network, no API key)
```

Evidence in `data/eval/`: `query_eval_set.json`, `section_inventory.json`,
`oracle_params.json`, `routing_results_{0ed3e740,f5252664,d4bc8e36}.{json,md}`,
`execution_results_d4bc8e36_b1d8ee5c{,_oracle}.{json,md}`, `retrieval_gold.json`,
`retrieval_results.{json,md}`, `attribution_set.json`, `attribution_seeds.json`,
`attribution_labels.json`, `attribution_results.{json,md}`.
