# Routing eval

planner_version `0dacd92a6441fe5ae19dcca321e6cdf2` · model `gpt-4o-mini-2024-07-18` (served `gpt-4o-mini-2024-07-18`) · 3 runs · eval set `c3af4d558c0c` (freeze a15727e)

Held-out is report-only. Intervals: n = questions, not question×runs.

| split | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| held | 42 | 87.3% | 74%–94% | 91.3% | 79%–97% | 37/37/36 | 4 | 2 |
| dev | 67 | 89.6% | 80%–95% | 92.5% | 84%–97% | 60/59/61 | 3 | 2 |
| all | 109 | 88.7% | 81%–93% | 92.0% | 85%–96% | 97/96/97 | 7 | 4 |

| held stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 1 | 0.0% | 0%–79% | 0.0% | 0%–79% | 0/0/0 | 0 | 0 |
| corpus_wide | 2 | 50.0% | 9%–91% | 50.0% | 9%–91% | 1/1/1 | 1 | 0 |
| hybrid | 6 | 100.0% | 61%–100% | 100.0% | 61%–100% | 6/6/6 | 0 | 0 |
| out_of_scope | 3 | 100.0% | 44%–100% | 100.0% | 44%–100% | 3/3/3 | 1 | 0 |
| rag_single | 17 | 80.4% | 57%–93% | 90.2% | 68%–98% | 14/14/13 | 2 | 2 |
| sql_multi | 1 | 100.0% | 21%–100% | 100.0% | 21%–100% | 1/1/1 | 0 | 0 |
| sql_single | 10 | 100.0% | 72%–100% | 100.0% | 72%–100% | 10/10/10 | 0 | 0 |
| unsupported | 2 | 100.0% | 34%–100% | 100.0% | 34%–100% | 2/2/2 | 0 | 0 |

| dev stratum | n | strict mean | strict 95% CI | lenient mean | lenient 95% CI | per run (strict) | plan varied | score varied |
|---|---|---|---|---|---|---|---|---|
| clarification | 4 | 75.0% | 30%–95% | 100.0% | 51%–100% | 3/3/3 | 0 | 0 |
| corpus_wide | 5 | 40.0% | 12%–77% | 40.0% | 12%–77% | 2/1/3 | 2 | 2 |
| hybrid | 7 | 100.0% | 65%–100% | 100.0% | 65%–100% | 7/7/7 | 0 | 0 |
| out_of_scope | 4 | 100.0% | 51%–100% | 100.0% | 51%–100% | 4/4/4 | 1 | 0 |
| rag_single | 25 | 92.0% | 75%–98% | 96.0% | 80%–99% | 23/23/23 | 0 | 0 |
| sql_multi | 3 | 100.0% | 44%–100% | 100.0% | 44%–100% | 3/3/3 | 0 | 0 |
| sql_single | 15 | 100.0% | 80%–100% | 100.0% | 80%–100% | 15/15/15 | 0 | 0 |
| unsupported | 4 | 75.0% | 30%–95% | 75.0% | 30%–95% | 3/3/3 | 0 | 0 |

Plan varied across runs: N18, N26, N33, N35, N45, N46, H05

## Misses (lenient), per run

| ID | split | run | category | resolver? | planner produced | strict label |
|---|---|---|---|---|---|---|
| N18 | held | 1 | plan_type |  | `hybrid|-|company_overview:NON_CORPUS:-:-:-,company_overview:NON_CORPUS:-:-:-` | `single` |
| N18 | held | 3 | plan_type |  | `hybrid|-|company_overview:NON_CORPUS:-:-:-,company_overview:NON_CORPUS:-:-:-` | `single` |
| N32 | dev | 1 | plan_type |  | `hybrid|-|competition_and_regulation:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N32 | dev | 2 | plan_type |  | `hybrid|-|competition_and_regulation:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N32 | dev | 3 | plan_type |  | `hybrid|-|competition_and_regulation:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N33 | dev | 1 | plan_type |  | `hybrid|-|competition_and_regulation:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N33 | dev | 2 | plan_type |  | `hybrid|-|competition_and_regulation:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N35 | held | 1 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:NON_CORPUS:-:-:-` | `single` |
| N35 | held | 2 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N35 | held | 3 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| N41 | dev | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N41 | dev | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N41 | dev | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 1 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 2 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N43 | held | 3 | plan_type |  | `unsupported|field_not_queryable|` | `single` |
| N44 | dev | 1 | unsupported_reason |  | `unsupported|field_not_queryable|` | `unsupported` |
| N44 | dev | 2 | unsupported_reason |  | `unsupported|field_not_queryable|` | `unsupported` |
| N44 | dev | 3 | unsupported_reason |  | `unsupported|field_not_queryable|` | `unsupported` |
| N50 | held | 1 | plan_type |  | `single|-|strategy:-:-:-:-` | `clarification` |
| N50 | held | 2 | plan_type |  | `single|-|strategy:-:-:-:-` | `clarification` |
| N50 | held | 3 | plan_type |  | `single|-|strategy:-:-:-:-` | `clarification` |
| H01 | dev | 1 | plan_type |  | `hybrid|-|company_overview:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| H01 | dev | 2 | plan_type |  | `hybrid|-|company_overview:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| H01 | dev | 3 | plan_type |  | `hybrid|-|company_overview:-:-:-:-,risk_factors:-:-:-:-` | `single` |
| H05 | dev | 2 | plan_type |  | `hybrid|-|risk_factors:-:-:-:-,risk_factors:-:-:-:-,risk_factors:-:-:-:-` | `single` |
