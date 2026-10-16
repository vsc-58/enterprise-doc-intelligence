# Routing eval

planner_version `f525266478672b4f74d093622823d52e` · model `gpt-4o-mini` · 3 runs · eval set `c3af4d558c0c` (freeze a15727e)

Held-out is report-only. Intervals: n = questions, not question×runs.

| split | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| held | 42 | 88.1% | 75%–95% | 92.9% | 81%–98% | 37/37/37 | 0 | 0 |
| dev | 67 | 87.6% | 78%–93% | 90.5% | 81%–96% | 59/58/59 | 2 | 1 |
| all | 109 | 87.8% | 80%–93% | 91.4% | 85%–95% | 96/95/96 | 2 | 1 |

| held stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 1 | 0.0% | 0%–79% | 0.0% | 0%–79% | 0/0/0 | 0 | 0 |
| corpus_wide | 2 | 100.0% | 34%–100% | 100.0% | 34%–100% | 2/2/2 | 0 | 0 |
| hybrid | 6 | 100.0% | 61%–100% | 100.0% | 61%–100% | 6/6/6 | 0 | 0 |
| out_of_scope | 3 | 100.0% | 44%–100% | 100.0% | 44%–100% | 3/3/3 | 0 | 0 |
| rag_single | 17 | 82.4% | 59%–94% | 94.1% | 73%–99% | 14/14/14 | 0 | 0 |
| sql_multi | 1 | 100.0% | 21%–100% | 100.0% | 21%–100% | 1/1/1 | 0 | 0 |
| sql_single | 10 | 90.0% | 60%–98% | 90.0% | 60%–98% | 9/9/9 | 0 | 0 |
| unsupported | 2 | 100.0% | 34%–100% | 100.0% | 34%–100% | 2/2/2 | 0 | 0 |

| dev stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 4 | 25.0% | 5%–70% | 50.0% | 15%–85% | 1/1/1 | 0 | 0 |
| corpus_wide | 5 | 80.0% | 38%–96% | 80.0% | 38%–96% | 4/4/4 | 0 | 0 |
| hybrid | 7 | 100.0% | 65%–100% | 100.0% | 65%–100% | 7/7/7 | 0 | 0 |
| out_of_scope | 4 | 100.0% | 51%–100% | 100.0% | 51%–100% | 4/4/4 | 0 | 0 |
| rag_single | 25 | 92.0% | 75%–98% | 96.0% | 80%–99% | 23/23/23 | 0 | 0 |
| sql_multi | 3 | 100.0% | 44%–100% | 100.0% | 44%–100% | 3/3/3 | 0 | 0 |
| sql_single | 15 | 91.1% | 67%–98% | 91.1% | 67%–98% | 14/13/14 | 2 | 1 |
| unsupported | 4 | 75.0% | 30%–95% | 75.0% | 30%–95% | 3/3/3 | 0 | 0 |

Plan varied across runs: R05, R07

## Misses (lenient), per run

| ID | split | run | category | resolver? | planner produced | strict label |
|---|---|---|---|---|---|---|
| R05 | dev | 2 | intent |  | `single|-|financial_metric:-:total_assets:-:highest` | `single` |
| R07 | dev | 1 | intent |  | `single|-|financial_filter:-:operating_cash_flow:-:-` | `single` |
| R07 | dev | 2 | intent |  | `single|-|financial_filter:-:operating_cash_flow:-:lowest` | `single` |
| R07 | dev | 3 | intent |  | `single|-|financial_filter:-:operating_cash_flow:-:-` | `single` |
| R50 | dev | 1 | plan_type |  | `single|-|financial_metric:-:total_revenue:-:-` | `clarification` |
| R50 | dev | 2 | plan_type |  | `single|-|financial_metric:-:total_revenue:-:-` | `clarification` |
| R50 | dev | 3 | plan_type |  | `single|-|financial_metric:-:total_revenue:-:-` | `clarification` |
| N14 | held | 1 | intent |  | `single|-|financial_filter:-:net_income:-:lowest` | `single` |
| N14 | held | 2 | intent |  | `single|-|financial_filter:-:net_income:-:lowest` | `single` |
| N14 | held | 3 | intent |  | `single|-|financial_filter:-:net_income:-:lowest` | `single` |
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
| H01 | dev | 3 | intent |  | `single|-|company_overview:-:-:-:-` | `single` |
