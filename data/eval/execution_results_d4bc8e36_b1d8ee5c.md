# Execution eval

planner `d4bc8e36` · chain `b1d8ee5c` · 3 runs · eval set `c3af4d558c0c`

Question correct = every task's outcome matches its label (end to end: and routed correctly). Intervals use n = questions. Held-out is report-only.

## oracle

| split | n | mean | 95% CI | per run | varied |
|---|---|---|---|---|---|
| held | 36 | 90.7% | 77%–97% | 33/32/33 | 1 |
| dev | 55 | 94.5% | 85%–98% | 53/51/52 | 2 |
| all | 91 | 93.0% | 86%–97% | 86/83/85 | 3 |

| expected outcome | task-runs | matched | rate |
|---|---|---|---|
| answered | 258 | 236 | 91.5% |
| answered_flagged | 24 | 24 | 100.0% |
| fallback_answered | 6 | 6 | 100.0% |
| non_corpus_company | 12 | 12 | 100.0% |
| section_absent | 15 | 15 | 100.0% |
| year_not_held | 12 | 12 | 100.0% |

Guard per run: run 1: masked 31, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2 · run 2: masked 31, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2 · run 3: masked 31, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2

Redactions (masking misses): none

### Misses

| ID | split | run | routed | task outcomes (expected → got) | error |
|---|---|---|---|---|---|
| R35 | held | 1 | yes | financial_metric:BAC answered→answered_flagged; risk_factors:BAC answered→not_addressed |  |
| N14 | held | 1 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 1 | yes | strategy:INTC answered→not_addressed |  |
| N39 | dev | 1 | yes | company_overview:GOOGL answered→not_addressed |  |
| N40 | dev | 1 | yes | strategy:WMT answered→not_addressed |  |
| R33 | dev | 2 | yes | company_overview:TSLA answered→not_addressed |  |
| R35 | held | 2 | yes | financial_metric:BAC answered→answered_flagged; risk_factors:BAC answered→not_addressed |  |
| N14 | held | 2 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 2 | yes | strategy:INTC answered→not_addressed |  |
| N37 | held | 2 | yes | risk_factors:TSLA answered→not_addressed |  |
| N39 | dev | 2 | yes | company_overview:GOOGL answered→not_addressed |  |
| N40 | dev | 2 | yes | strategy:WMT answered→not_addressed |  |
| N41 | dev | 2 | yes | management_commentary:AAPL answered→figure_requested |  |
| R33 | dev | 3 | yes | company_overview:TSLA answered→not_addressed |  |
| R35 | held | 3 | yes | financial_metric:BAC answered→answered_flagged; risk_factors:BAC answered→not_addressed |  |
| N14 | held | 3 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 3 | yes | strategy:INTC answered→not_addressed |  |
| N39 | dev | 3 | yes | company_overview:GOOGL answered→not_addressed |  |
| N40 | dev | 3 | yes | strategy:WMT answered→not_addressed |  |

## end_to_end

| split | n | mean | 95% CI | per run | varied |
|---|---|---|---|---|---|
| held | 42 | 84.9% | 71%–93% | 35/36/36 | 1 |
| dev | 67 | 93.5% | 85%–97% | 63/63/62 | 2 |
| all | 109 | 90.2% | 83%–94% | 98/99/98 | 3 |

| expected outcome | task-runs | matched | rate |
|---|---|---|---|
| ambiguous_company | 3 | 3 | 100.0% |
| answered | 246 | 237 | 96.3% |
| answered_flagged | 24 | 24 | 100.0% |
| fallback_answered | 6 | 6 | 100.0% |
| non_corpus_company | 12 | 12 | 100.0% |
| section_absent | 12 | 12 | 100.0% |
| year_not_held | 12 | 12 | 100.0% |

Guard per run: run 1: masked 22, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2 · run 2: masked 19, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2 · run 3: masked 19, redacted 0, invalid citations 0, fallback answers 2, fallback findings 2

Redactions (masking misses): none

### Misses

| ID | split | run | routed | task outcomes (expected → got) | error |
|---|---|---|---|---|---|
| R35 | held | 1 | yes | financial_metric:BAC answered→answered_flagged |  |
| N01 | held | 1 | no | financial_metric:NFLX None→year_not_held |  |
| N14 | held | 1 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 1 | yes | strategy:INTC answered→not_addressed |  |
| N32 | dev | 1 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N35 | held | 1 | no | risk_factors:None None→answered; risk_factors:NON_CORPUS None→non_corpus_company |  |
| N43 | held | 1 | no |  |  |
| N44 | dev | 1 | no |  |  |
| N49 | dev | 1 | no | risk_factors:None None→answered |  |
| N50 | held | 1 | no | strategy:None None→answered |  |
| H01 | dev | 1 | no | company_overview:None None→answered; risk_factors:None None→answered |  |
| R35 | held | 2 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 2 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 2 | yes | strategy:INTC answered→not_addressed |  |
| N32 | dev | 2 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N35 | held | 2 | no | risk_factors:None None→answered; risk_factors:None None→answered; risk_factors:None None→answered |  |
| N41 | dev | 2 | no |  |  |
| N43 | held | 2 | no |  |  |
| N44 | dev | 2 | no |  |  |
| N50 | held | 2 | no | strategy:None None→answered |  |
| H01 | dev | 2 | no | company_overview:None None→answered; risk_factors:None None→answered |  |
| R35 | held | 3 | yes | financial_metric:BAC answered→answered_flagged |  |
| N14 | held | 3 | yes | financial_ranking:None answered→answered_flagged |  |
| N20 | held | 3 | yes | strategy:INTC answered→not_addressed |  |
| N32 | dev | 3 | no | risk_factors:None None→answered; competition_and_regulation:None None→answered |  |
| N35 | held | 3 | no | risk_factors:None None→answered; risk_factors:NON_CORPUS None→non_corpus_company |  |
| N41 | dev | 3 | no |  |  |
| N43 | held | 3 | no |  |  |
| N44 | dev | 3 | no |  |  |
| N49 | dev | 3 | no | risk_factors:None None→answered |  |
| N50 | held | 3 | no | strategy:None None→answered |  |
| H01 | dev | 3 | no | company_overview:None None→answered; risk_factors:None None→answered |  |
