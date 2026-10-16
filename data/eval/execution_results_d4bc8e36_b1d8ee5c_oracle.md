# Execution eval

planner `d4bc8e36` · chain `b1d8ee5c` · 3 runs · eval set `c3af4d558c0c`

Question correct = every task's outcome matches its label (end to end: and routed correctly). Intervals use n = questions. Held-out is report-only.

## oracle

| split | n | mean | 95% CI | per run | varied |
|---|---|---|---|---|---|
| held | 36 | 91.7% | 78%–97% | 33/33/33 | 0 |
| dev | 55 | 100.0% | 93%–100% | 55/55/55 | 0 |
| all | 91 | 96.7% | 91%–99% | 88/88/88 | 0 |

| expected outcome | task-runs | matched | rate |
|---|---|---|---|
| answered | 258 | 249 | 96.5% |
| answered_flagged | 24 | 24 | 100.0% |
| fallback_answered | 6 | 6 | 100.0% |
| non_corpus_company | 12 | 12 | 100.0% |
| section_absent | 15 | 15 | 100.0% |
| year_not_held | 12 | 12 | 100.0% |

Guard per run: run 1: masked 28, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2 · run 2: masked 28, redacted 0, invalid citations 1, fallback answers 2, fallback findings 2 · run 3: masked 28, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2

Redactions (masking misses): none

### Misses

| ID | split | run | routed | task outcomes (expected → got) | error |
|---|---|---|---|---|---|
| R35 | held | 1 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 1 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 1 | yes | strategy:INTC answered→not_addressed |  |
| R35 | held | 2 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 2 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 2 | yes | strategy:INTC answered→not_addressed |  |
| R35 | held | 3 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 3 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 3 | yes | strategy:INTC answered→not_addressed |  |
