# Evaluation evidence

Every number in the main README comes from a file in this folder. Files are
never overwritten: a re-run with a different prompt, chain or judge version
writes a new file, named by the first eight characters of those versions
(`0dacd92a` = planner, `79177d36` = answer chain, `95183c61` = judge). Older
files stay as evidence of what earlier versions scored.

**Current** means the files behind the README's numbers, measured on the
pinned model `gpt-4o-mini-2024-07-18`. **History** means earlier runs, kept so
every change can be traced.

## Extraction (figures from filing text)

| File | Status | What it holds |
|---|---|---|
| `ground_truth.json` | Current | The answer key: five figures per filing, read from XBRL by exact concept |
| `ground_truth_PREV.json` | History | The answer key before the revenue-concept fix ([prompt strategy results](../../docs/prompt_strategy_results.md), section 5.1) |
| `results.json` | Current | Five prompt strategies on the ten selection filings: per-field precision, recall, F1 |
| `held_out_results.json` | Current | The chosen strategy on the ten held-out filings |
| `raw_outputs/S0`–`S4/` | Current | Every model response, cached by content hash, so re-scoring costs nothing |
| `grounding_parity_baseline.json` | Tooling | Baseline for `scripts/diagnostic/verify_numbers_parity.py`, which proved a code move changed no verdict |

## Query path (questions and answers)

| File | Status | What it holds |
|---|---|---|
| `query_eval_set.json` | Current | 109 questions (67 dev, 42 held-out) with frozen labels (checked by SHA-256 before scoring) |
| `oracle_params.json` | Current | Thresholds and per-task questions for the gold-plan ("oracle") runs |
| `section_inventory.json` | Current | Which sections each filing holds; the source of expected refusals |
| `routing_results_0dacd92a.*` | Current | Planner accuracy, final prompt, pinned model, 3 runs |
| `routing_results_0ed3e740.*` | History | First prompt version: the clean out-of-sample 82.5% |
| `routing_results_f5252664.*`, `routing_results_d4bc8e36.*` | History | Second and third prompt versions, October, model alias |
| `execution_results_0dacd92a_79177d36.*` | Current | Oracle and end-to-end execution, pinned model |
| `execution_results_d4bc8e36_b1d8ee5c*.*` | History | The same in October, model alias |
| `retrieval_gold.json` | Current | Chunk-level relevance labels (frozen by SHA-256) |
| `retrieval_results_0dacd92a.*` | Current | Hit and recall at 5 and 10 on the planner's queries |
| `retrieval_results.*` | History | The same in October |
| `attribution_set.json`, `attribution_seeds.json`, `attribution_labels.json` | Current | Judge validation: real claims, 30 planted errors, reference labels |
| `attribution_results_95183c61_0dacd92a_79177d36.*` | Current | Judge validation and the 91.4% attribution score |
| `attribution_results.*` | History | The same in October |

`.json` files are the full results; `.md` files are the readable summaries.
Methods, caveats and every adjudicated label are in
[docs/query_eval_results.md](../../docs/query_eval_results.md).
