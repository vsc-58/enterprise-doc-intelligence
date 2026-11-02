# Execution eval

planner `0dacd92a` · chain `79177d36` · model `gpt-4o-mini-2024-07-18` · 3 runs · eval set `c3af4d558c0c`

Question correct = every task's outcome matches its label (end to end: and routed correctly). Intervals use n = questions. Held-out is report-only.

## oracle

| split | n | mean | 95% CI | per run | varied |
|---|---|---|---|---|---|
| held | 36 | 91.7% | 78%–97% | 33/33/33 | 0 |
| dev | 55 | 98.2% | 90%–100% | 54/54/54 | 0 |
| all | 91 | 95.6% | 89%–98% | 87/87/87 | 0 |

| expected outcome | task-runs | matched | rate |
|---|---|---|---|
| answered | 258 | 246 | 95.3% |
| answered_flagged | 24 | 24 | 100.0% |
| fallback_answered | 6 | 6 | 100.0% |
| non_corpus_company | 12 | 12 | 100.0% |
| section_absent | 15 | 15 | 100.0% |
| year_not_held | 12 | 12 | 100.0% |

Guard per run: run 1: masked 28, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2 · run 2: masked 28, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2 · run 3: masked 28, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2

Redactions (masking misses): none

### Misses

| ID | split | run | routed | task outcomes (expected → got) | error |
|---|---|---|---|---|---|
| R35 | held | 1 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 1 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 1 | yes | strategy:INTC answered→not_addressed |  |
| H04 | dev | 1 | yes | company_overview:INTC answered→not_addressed |  |
| R35 | held | 2 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 2 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 2 | yes | strategy:INTC answered→not_addressed |  |
| H04 | dev | 2 | yes | company_overview:INTC answered→not_addressed |  |
| R35 | held | 3 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 3 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 3 | yes | strategy:INTC answered→not_addressed |  |
| H04 | dev | 3 | yes | company_overview:INTC answered→not_addressed |  |

## end_to_end

| split | n | mean | 95% CI | per run | varied |
|---|---|---|---|---|---|
| held | 42 | 83.3% | 69%–92% | 36/35/34 | 3 |
| dev | 67 | 89.6% | 80%–95% | 60/61/59 | 4 |
| all | 109 | 87.2% | 80%–92% | 96/96/93 | 7 |

| expected outcome | task-runs | matched | rate |
|---|---|---|---|
| ambiguous_company | 3 | 3 | 100.0% |
| answered | 240 | 228 | 95.0% |
| answered_flagged | 24 | 24 | 100.0% |
| fallback_answered | 6 | 6 | 100.0% |
| non_corpus_company | 11 | 11 | 100.0% |
| section_absent | 12 | 12 | 100.0% |
| year_not_held | 12 | 12 | 100.0% |

Guard per run: run 1: masked 19, redacted 0, invalid citations 1, fallback answers 2, fallback findings 2 · run 2: masked 23, redacted 0, invalid citations 1, fallback answers 2, fallback findings 2 · run 3: masked 22, redacted 0, invalid citations 0, fallback answers 2, fallback findings 0

Redactions (masking misses): none

### Misses

| ID | split | run | routed | task outcomes (expected → got) | error |
|---|---|---|---|---|---|
| R35 | held | 1 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 1 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 1 | yes | strategy:INTC answered→not_addressed |  |
| N32 | dev | 1 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N33 | dev | 1 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N35 | held | 1 | no | risk_factors:None None→answered; risk_factors:None None→answered |  |
| N41 | dev | 1 | no |  |  |
| N43 | held | 1 | no |  |  |
| N44 | dev | 1 | no |  |  |
| N50 | held | 1 | no | strategy:None None→answered |  |
| H01 | dev | 1 | no | company_overview:None None→answered; risk_factors:None None→answered |  |
| H04 | dev | 1 | yes | company_overview:INTC answered→not_addressed |  |
| H05 | dev | 1 | no | risk_factors:None None→answered; risk_factors:None None→answered; risk_factors:None None→answered |  |
| R35 | held | 2 | yes | financial_metric:BAC answered→answered_flagged |  |
| N01 | held | 2 | no | financial_metric:NFLX None→year_not_held |  |
| N14 | held | 2 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 2 | yes | strategy:INTC answered→not_addressed |  |
| N32 | dev | 2 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N35 | held | 2 | no | risk_factors:None None→answered; risk_factors:None None→answered |  |
| N41 | dev | 2 | no |  |  |
| N43 | held | 2 | no |  |  |
| N44 | dev | 2 | no |  |  |
| N49 | dev | 2 | no | risk_factors:None None→answered |  |
| N50 | held | 2 | no | strategy:None None→answered |  |
| H01 | dev | 2 | no | company_overview:None None→answered; risk_factors:None None→answered; management_commentary:None None→answered |  |
| H04 | dev | 2 | yes | company_overview:INTC answered→not_addressed |  |
| R35 | held | 3 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 3 | yes | financial_ranking:None answered→answered_flagged |  |
| N18 | held | 3 | no | company_overview:NON_CORPUS None→non_corpus_company; company_overview:NON_CORPUS None→non_corpus_company |  |
| N20 | held | 3 | yes | strategy:INTC answered→not_addressed |  |
| N32 | dev | 3 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N34 | held | 3 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N35 | held | 3 | no | risk_factors:None None→answered; risk_factors:None None→answered; risk_factors:None None→answered |  |
| N41 | dev | 3 | no |  |  |
| N43 | held | 3 | no |  |  |
| N44 | dev | 3 | no |  |  |
| N49 | dev | 3 | no | risk_factors:None None→answered |  |
| N50 | held | 3 | no | strategy:None None→answered |  |
| H01 | dev | 3 | no | company_overview:None None→answered; risk_factors:None None→answered; management_commentary:None None→answered |  |
| H04 | dev | 3 | yes | company_overview:INTC answered→not_addressed |  |
| H05 | dev | 3 | no | risk_factors:None None→answered; risk_factors:None None→answered; risk_factors:None None→answered |  |
| H08 | dev | 3 | no | risk_factors:None None→answered |  |
