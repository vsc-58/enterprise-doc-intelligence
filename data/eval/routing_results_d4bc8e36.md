# Routing eval

planner_version `d4bc8e369907a2638eba7c3a43a7ea65` · model `gpt-4o-mini` · 3 runs · eval set `c3af4d558c0c` (freeze a15727e)

Held-out is report-only. Intervals: n = questions, not question×runs.

| split | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| held | 42 | 88.9% | 76%–95% | 92.9% | 81%–98% | 38/36/38 | 3 | 3 |
| dev | 67 | 90.0% | 81%–95% | 93.0% | 84%–97% | 60/60/61 | 2 | 1 |
| all | 109 | 89.6% | 82%–94% | 93.0% | 87%–96% | 98/96/99 | 5 | 4 |

| held stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 1 | 0.0% | 0%–79% | 0.0% | 0%–79% | 0/0/0 | 0 | 0 |
| corpus_wide | 2 | 83.3% | 24%–99% | 83.3% | 24%–99% | 2/1/2 | 1 | 1 |
| hybrid | 6 | 100.0% | 61%–100% | 100.0% | 61%–100% | 6/6/6 | 0 | 0 |
| out_of_scope | 3 | 100.0% | 44%–100% | 100.0% | 44%–100% | 3/3/3 | 0 | 0 |
| rag_single | 17 | 84.3% | 61%–95% | 94.1% | 73%–99% | 14/14/15 | 1 | 1 |
| sql_multi | 1 | 100.0% | 21%–100% | 100.0% | 21%–100% | 1/1/1 | 0 | 0 |
| sql_single | 10 | 93.3% | 63%–99% | 93.3% | 63%–99% | 10/9/9 | 1 | 1 |
| unsupported | 2 | 100.0% | 34%–100% | 100.0% | 34%–100% | 2/2/2 | 0 | 0 |

| dev stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 4 | 50.0% | 15%–85% | 75.0% | 30%–95% | 2/2/2 | 0 | 0 |
| corpus_wide | 5 | 66.7% | 28%–91% | 66.7% | 28%–91% | 3/3/4 | 2 | 1 |
| hybrid | 7 | 100.0% | 65%–100% | 100.0% | 65%–100% | 7/7/7 | 0 | 0 |
| out_of_scope | 4 | 100.0% | 51%–100% | 100.0% | 51%–100% | 4/4/4 | 0 | 0 |
| rag_single | 25 | 92.0% | 75%–98% | 96.0% | 80%–99% | 23/23/23 | 0 | 0 |
| sql_multi | 3 | 100.0% | 44%–100% | 100.0% | 44%–100% | 3/3/3 | 0 | 0 |
| sql_single | 15 | 100.0% | 80%–100% | 100.0% | 80%–100% | 15/15/15 | 0 | 0 |
| unsupported | 4 | 75.0% | 30%–95% | 75.0% | 30%–95% | 3/3/3 | 0 | 0 |

Plan varied across runs: N01, N26, N32, N35, H01

## Misses (lenient), per run

| ID | split | run | category | resolver? | planner produced | strict label |
|---|---|---|---|---|---|---|
| N01 | held | 2 | parameter |  | `single|-|financial_metric:NFLX:total_revenue:2022:-` | `single` |
| N01 | held | 3 | parameter |  | `single|-|financial_metric:NFLX:total_revenue:2022:-` | `single` |
| N32 | dev | 1 | plan_type |  | `hybrid|-|management_commentary:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N32 | dev | 2 | plan_type |  | `hybrid|-|competition_and_regulation:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N35 | held | 2 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N41 | dev | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N41 | dev | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N41 | dev | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N44 | dev | 1 | unsupported_reason |  | `unsupported|field_not_queryable|` | `unsupported` |
| N44 | dev | 2 | unsupported_reason |  | `unsupported|field_not_queryable|` | `unsupported` |
| N44 | dev | 3 | unsupported_reason |  | `unsupported|field_not_queryable|` | `unsupported` |
| N49 | dev | 1 | company |  | `single|-|risk_factors:-:-:-:-` | `clarification` |
| N49 | dev | 2 | company |  | `single|-|risk_factors:-:-:-:-` | `clarification` |
| N49 | dev | 3 | company |  | `single|-|risk_factors:-:-:-:-` | `clarification` |
| N50 | held | 1 | plan_type |  | `single|-|strategy:-:-:-:-` | `clarification` |
| N50 | held | 2 | plan_type |  | `single|-|strategy:-:-:-:-` | `clarification` |
| N50 | held | 3 | plan_type |  | `single|-|strategy:-:-:-:-` | `clarification` |
| H01 | dev | 1 | intent |  | `single|-|company_overview:-:-:-:-` | `single` |
| H01 | dev | 2 | intent |  | `single|-|company_overview:-:-:-:-` | `single` |
| H01 | dev | 3 | plan_type |  | `hybrid|-|company_overview:-:-:-:-,risk_factors:-:-:-:-` | `single` |
