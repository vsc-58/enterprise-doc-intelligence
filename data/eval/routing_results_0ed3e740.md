# Routing eval

planner_version `0ed3e740ddc022a3d9574c0a488647f7` · model `gpt-4o-mini` · 3 runs · eval set `c3af4d558c0c` (freeze a15727e)

Held-out is report-only. Intervals: n = questions, not question×runs.

| split | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| held | 42 | 82.5% | 68%–91% | 87.3% | 74%–94% | 34/34/36 | 3 | 2 |
| dev | 67 | 79.6% | 68%–88% | 84.1% | 74%–91% | 54/54/52 | 2 | 2 |
| all | 109 | 80.7% | 72%–87% | 85.3% | 77%–91% | 88/88/88 | 5 | 4 |

| held stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 1 | 100.0% | 21%–100% | 100.0% | 21%–100% | 1/1/1 | 0 | 0 |
| corpus_wide | 2 | 0.0% | 0%–66% | 0.0% | 0%–66% | 0/0/0 | 1 | 0 |
| hybrid | 6 | 100.0% | 61%–100% | 100.0% | 61%–100% | 6/6/6 | 0 | 0 |
| out_of_scope | 3 | 100.0% | 44%–100% | 100.0% | 44%–100% | 3/3/3 | 0 | 0 |
| rag_single | 17 | 78.4% | 55%–92% | 90.2% | 68%–98% | 13/13/14 | 1 | 1 |
| sql_multi | 1 | 0.0% | 0%–79% | 0.0% | 0%–79% | 0/0/0 | 0 | 0 |
| sql_single | 10 | 93.3% | 63%–99% | 93.3% | 63%–99% | 9/9/10 | 1 | 1 |
| unsupported | 2 | 100.0% | 34%–100% | 100.0% | 34%–100% | 2/2/2 | 0 | 0 |

| dev stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 4 | 50.0% | 15%–85% | 75.0% | 30%–95% | 2/2/2 | 0 | 0 |
| corpus_wide | 5 | 33.3% | 9%–72% | 33.3% | 9%–72% | 2/2/1 | 1 | 1 |
| hybrid | 7 | 100.0% | 65%–100% | 100.0% | 65%–100% | 7/7/7 | 0 | 0 |
| out_of_scope | 4 | 100.0% | 51%–100% | 100.0% | 51%–100% | 4/4/4 | 0 | 0 |
| rag_single | 25 | 84.0% | 65%–94% | 92.0% | 75%–98% | 21/21/21 | 0 | 0 |
| sql_multi | 3 | 33.3% | 6%–79% | 33.3% | 6%–79% | 1/1/1 | 0 | 0 |
| sql_single | 15 | 86.7% | 62%–96% | 86.7% | 62%–96% | 13/13/13 | 0 | 0 |
| unsupported | 4 | 91.7% | 43%–99% | 91.7% | 43%–99% | 4/4/3 | 1 | 1 |

Plan varied across runs: N01, N35, N42, N44, H08

## Misses (lenient), per run

| ID | split | run | category | resolver? | planner produced | strict label |
|---|---|---|---|---|---|---|
| R11 | dev | 1 | parameter |  | `single|-|financial_metric:AAPL:net_income:-:-` | `single` |
| R11 | dev | 2 | parameter |  | `single|-|financial_metric:AAPL:net_income:-:-` | `single` |
| R11 | dev | 3 | parameter |  | `single|-|financial_metric:AAPL:net_income:-:-` | `single` |
| R50 | dev | 1 | plan_type |  | `single|-|financial_metric:-:total_revenue:-:-` | `clarification` |
| R50 | dev | 2 | plan_type |  | `single|-|financial_metric:-:total_revenue:-:-` | `clarification` |
| R50 | dev | 3 | plan_type |  | `single|-|financial_metric:-:total_revenue:-:-` | `clarification` |
| N01 | held | 1 | parameter |  | `single|-|financial_metric:NFLX:total_revenue:2022:-` | `single` |
| N01 | held | 2 | parameter |  | `single|-|financial_metric:NFLX:total_revenue:2022:-` | `single` |
| N08 | dev | 1 | plan_type |  | `unsupported|cross_year|` | `single` |
| N08 | dev | 2 | plan_type |  | `unsupported|cross_year|` | `single` |
| N08 | dev | 3 | plan_type |  | `unsupported|cross_year|` | `single` |
| N10 | dev | 1 | intent |  | `hybrid|-|financial_ranking:AAPL:total_revenue:-:highest,financial_ranking:META:total_revenue:-:highest,financial_ranking:MSFT:total_revenue:-:highest` | `hybrid` |
| N10 | dev | 2 | intent |  | `hybrid|-|financial_ranking:AAPL:total_revenue:-:highest,financial_ranking:META:total_revenue:-:highest,financial_ranking:MSFT:total_revenue:-:highest` | `hybrid` |
| N10 | dev | 3 | intent |  | `hybrid|-|financial_ranking:AAPL:total_revenue:-:highest,financial_ranking:META:total_revenue:-:highest,financial_ranking:MSFT:total_revenue:-:highest` | `hybrid` |
| N13 | held | 1 | plan_type |  | `unsupported|aggregation|` | `hybrid` |
| N13 | held | 2 | plan_type |  | `unsupported|aggregation|` | `hybrid` |
| N13 | held | 3 | plan_type |  | `unsupported|aggregation|` | `hybrid` |
| N15 | dev | 1 | plan_type |  | `unsupported|aggregation|` | `hybrid` |
| N15 | dev | 2 | plan_type |  | `unsupported|aggregation|` | `hybrid` |
| N15 | dev | 3 | plan_type |  | `unsupported|aggregation|` | `hybrid` |
| N29 | dev | 1 | plan_type |  | `hybrid|-|management_commentary:WMT:-:-:-,management_commentary:WMT:-:-:-` | `single` |
| N29 | dev | 2 | plan_type |  | `hybrid|-|management_commentary:WMT:-:-:-,management_commentary:WMT:-:-:-` | `single` |
| N29 | dev | 3 | plan_type |  | `hybrid|-|management_commentary:WMT:-:-:-,management_commentary:WMT:-:-:-` | `single` |
| N32 | dev | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N32 | dev | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N32 | dev | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N33 | dev | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N33 | dev | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N33 | dev | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N34 | held | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N34 | held | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N34 | held | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N35 | held | 1 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N35 | held | 2 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N35 | held | 3 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N41 | dev | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N41 | dev | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N41 | dev | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N42 | held | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N42 | held | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N44 | dev | 3 | unsupported_reason |  | `unsupported|field_not_queryable|` | `unsupported` |
| H01 | dev | 1 | plan_type |  | `out_of_scope|-|` | `single` |
| H01 | dev | 2 | plan_type |  | `out_of_scope|-|` | `single` |
| H01 | dev | 3 | plan_type |  | `out_of_scope|-|` | `single` |
| H08 | dev | 3 | plan_type |  | `unsupported|aggregation|` | `single` |
