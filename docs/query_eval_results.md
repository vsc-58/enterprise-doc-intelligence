# Query-Path Evaluation — Phase 5C Results

Status: routing and execution complete (2026-10-13). Retrieval hit-rate and
attribution support (D43) pending — section 7.

All numbers are means over **3 runs** (temperature 0 is not deterministic, D35).
95% intervals are Wilson intervals with **n = questions**: repeated runs measure
the model's variability on the same questions and do not narrow the interval.
**Held-out is report-only** (D17 reasoning): no prompt was revised in response to
a held-out result, and every revision is tied to a named dev miss.

---

## 1. Headline

| Measure | Held-out | Dev |
|---|---|---|
| Routing, v1 — **clean out-of-sample** | **82.5%** [68–91] | 79.6% [68–88] |
| Routing, v3 (final) | 88.9% [76–95] | 90.0% [81–95] |
| End-to-end correct outcome (v3, adjudicated) | 89.7% [77–96] | 93.5% [85–97] |
| Downstream with gold plan (oracle, non-hybrid, adjudicated) | 96.6% [83–99] | 99.3% [91–100] |
| Deterministic refusals (section absent, year not held, non-corpus, ambiguous, flagged figure, fallback) | 100% | 100% |
| Figure-guard redactions (masking misses), 6 runs | 0 | 0 |
| Invalid citations, 6 runs | 0 | 0 |

Reading it: the downstream system is near-exact when given the right plan; the
remaining end-to-end errors are almost all routing. v1's held-out number is the
only one untouched by tuning; v2 and v3 held-out numbers are reported beside it,
not instead of it (section 4.4).

---

## 2. Eval Set

`data/eval/query_eval_set.json` — 109 questions, frozen at commit `a15727e`,
SHA-256 `c3af4d55…603e` (line endings normalised). Harnesses refuse to score
against any other hash.

| Source | Dev | Held | Notes |
|---|---|---|---|
| Original 5A/5B set, relabelled to 9 intents (D38/D40) | 33 | 17 | 5 held questions used verbatim in 5A/5B probes moved to dev |
| Written by a separate LLM, user phrasing (typos, tickers, vague) | 25 | 25 | Stratified split, fixed seed |
| Hard cases written with system knowledge | 9 | 0 | Dev only |

Each row carries a strict label, accepted alternatives (lenient scoring), the
expected outcome per task (derived from `data/eval/section_inventory.json`, not
intuition — D24), and a narrative-set flag (23 questions) for retrieval and
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
NarrativeRefusal, D29 quality verdict, D44 fallback) and compared with the
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

### 4.2 v1 miss patterns (D46)

| | Pattern | Dev evidence | User impact |
|---|---|---|---|
| P1 | "Which companies…" narrative planned unsupported / out of scope | N32, N33, H01 | Wrong refusal; the D39 survey mostly unreachable |
| P2 | Figures stated in text (R&D, dividends) → field_not_queryable | N41 | Wrong refusal |
| P3 | "compare X and Y" → aggregation | N15 | Wrong refusal |
| P4 | Ranking within named companies → one ranking per company | N10 | **Wrong answer**: the corpus-wide winner, three times |
| P5 | "last year" → a year, or cross_year | N08 | Wrong refusal |
| P6 | "net sales" → net_income | R11 | **Wrong figure** |

P4 was also fixed structurally (`executor.scoped_to_company`): a ranking or
filter that names a company runs as that company's lookup. It is the only
pattern detectable from the plan's shape alone (D46).

### 4.3 Paired comparison — fixed vs broken

| Step | Fixed | Broken |
|---|---|---|
| v1 → v2 held | N01, N13, N34, N35, N42 | N14, N50 |
| v1 → v2 dev | R11, N08, N10, N15, N29, N32, N33, H08 | R07, N49, R05 (1 run) |
| v2 → v3 | R05, R07, N14, R50 | N01, N32, N35 (some runs) |

v2's regressions were caused by its own wording ("never financial_ranking",
"not a clarification"): prohibitions generalise beyond their target. v3
restated both as positive discriminators (D47). A stopping rule declared before
the v3 run made it final.

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
| N41, N43 (R&D, dividends) | field_not_queryable | Wrong refusal; three prompt attempts failed — D45 needs a schema signal or a stronger planner |
| N01 ("last year") | Year 2022 in some runs | Recoverable refusal |
| N32, N35 | Survey split into 2–3 tasks | Duplicate surveys, double cost |
| H01 (bitcoin) | company_overview survey | Searches Item 1; Tesla's holding (Item 7) missed |

---

## 5. Execution Outcomes

### 5.1 Oracle vs end to end on identical questions (adjudicated)

| Questions | Split | Oracle (per run) | End to end (per run) |
|---|---|---|---|
| Executable, non-hybrid | held (29) | 28 / 28 / 28 | 25 / 26 / 26 |
| | dev (45) | 45 / 44 / 45 | 43 / 42 / 42 |
| Hybrid | held (7) | see 6.2 | 7 / 7 / 7 |
| | dev (10) | see 6.2 | 10 / 10 / 10 |

Every non-hybrid question that passes the oracle but fails end to end is a
routing miss already listed in 4.5: N01, N32, N35, N41, N43, H01.

### 5.2 Outcomes decided in code — all exact

| Expected outcome | Oracle task-runs matched |
|---|---|
| section_absent (MD&A or market risk not captured) | 15 / 15 |
| year_not_held | 12 / 12 |
| non_corpus_company | 12 / 12 |
| answered_flagged (D29 disputed figure disclosed) | 24 / 24 |
| fallback_answered (D44, R31 and H02) | 6 / 6 |
| ambiguous_company (end to end) | 3 / 3 |

### 5.3 Figure guard and citations (D41)

| Run | Oracle masked | Oracle redacted | E2E masked | E2E redacted | Invalid citations |
|---|---|---|---|---|---|
| 1 | 31 | 0 | 22 | 0 | 0 |
| 2 | 31 | 0 | 19 | 0 | 0 |
| 3 | 31 | 0 | 19 | 0 | 0 |

The redaction backstop never fired: masking caught every stored figure before
the model saw it. Every citation pointed at a retrieved block.

### 5.4 Corpus-wide surveys (oracle, run 1)

| Question | Supported | Not found | Not searchable | Fallback |
|---|---|---|---|---|
| H08 interest-rate risk (5B headline) | 19 | 1 | 0 | JPM, XOM via Item 1A |
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
| N44 | ratio_or_derived | field_not_queryable defensible | Debt-to-equity needs equity, which is not stored |
| R35 | BAC revenue answered | answered_flagged | BAC's four fields are in the 2–2 scale tie (5A); revenue is disputed |
| N14 | answered | answered_flagged | The lowest row is Intel, whose net income is disputed |

### 6.2 Harness artifact — oracle hybrids

The oracle passes the original question to each narrative task. For a hybrid
("Describe Tesla's business and give its total revenue"), the narrative task
then receives a question that also asks for a figure, and the chain refuses.
Five oracle misses (R33, N37, N39, N40, R35's narrative half) are this
artifact; all five pass end to end, where the planner splits the question.
Oracle numbers above therefore exclude hybrids. Fix before the attribution run:
per-task questions for hybrids, or hybrids excluded from oracle.

### 6.3 Genuine downstream misses

- **N20** (Intel turnaround strategy): not_addressed in every run, both modes.
  Intel's strategy pool is 2 chunks (no Item 7). Probably an honest refusal;
  to be verified against the chunks during retrieval labelling.
- **N41** (Apple R&D): the chain returned figure_requested in 1 of 3 runs,
  treating R&D as a reserved figure. D45's policy is fragile at both the
  planner and the chain.

---

## 7. Pending

- **Retrieval hit-rate**: chunk-level gold for the 23 narrative-set questions
  (keyword + top-10 pooling, model-proposed and human-verified), recall@k.
  Ticker + section hit-rate is ~100% by construction under filtered retrieval
  and is reported only as a sanity counter.
- **Attribution support (D43)**: validated judge (gpt-6-luna) over every claim.
  Pass criteria fixed in D43 before any judge output: seeded catch rate ≥ 0.90
  overall and ≥ 0.80 per type, Cohen's kappa ≥ 0.70 against blind human labels,
  ≤ 5% label flips across runs. Run 1 produced 157 single-company claims and 97
  supported survey findings (oracle).

---

## 8. Reproduce

```bash
python -m scripts.diagnostic.dump_section_inventory   # section inventory
python -m scripts.run_routing_eval                    # routing, 3 runs
python -m scripts.run_execution_eval                  # oracle + end to end, 3 runs
pytest tests                                          # 157 unit tests
```

Evidence files in `data/eval/`: `query_eval_set.json`, `section_inventory.json`,
`oracle_params.json`, `routing_results_{0ed3e740,f5252664,d4bc8e36}.{json,md}`,
`execution_results_d4bc8e36_b1d8ee5c.{json,md}`. Decisions D43–D47.
