# Retrieval eval

gold `verified` · planner `d4bc8e36` · embeddings `text-embedding-3-small` · k = [5, 10]

- recall@k is capped: relevant in top k / min(k, R).
- Recall is an upper bound: relevant chunks outside the labelled pool are unknown.
- Precision counts unjudged chunks as not relevant: a lower bound while any remain.
- Planner scores average a question's runs first; each question counts once.
- Wilson 95% intervals on hit@k use n = questions.

## Specific questions — strict

| split | source | n | hit@5 | recall@5 | prec@5 | hit@10 | recall@10 | prec@10 | first rank |
|---|---|---|---|---|---|---|---|---|---|
| held | original | 8 | 100% [68%–100%] | 79% | 50% | 100% [68%–100%] | 86% | 38% | 1.5 |
| held | planner | 8 | 100% [68%–100%] | 79% | 50% | 100% [68%–100%] | 83% | 35% | 1.5 |
| dev | original | 10 | 100% [72%–100%] | 57% | 44% | 100% [72%–100%] | 68% | 32% | 1.4 |
| dev | planner | 10 | 100% [72%–100%] | 59% | 46% | 100% [72%–100%] | 70% | 33% | 1.4 |
| all | original | 18 | 100% [82%–100%] | 67% | 47% | 100% [82%–100%] | 76% | 35% | 1.4 |
| all | planner | 18 | 100% [82%–100%] | 68% | 48% | 100% [82%–100%] | 76% | 34% | 1.4 |

## Specific questions — lenient

| split | source | n | hit@5 | recall@5 | prec@5 | hit@10 | recall@10 | prec@10 | first rank |
|---|---|---|---|---|---|---|---|---|---|
| held | original | 8 | 100% [68%–100%] | 67% | 60% | 100% [68%–100%] | 78% | 52% | 1.0 |
| held | planner | 8 | 100% [68%–100%] | 64% | 58% | 100% [68%–100%] | 78% | 52% | 1.0 |
| dev | original | 10 | 100% [72%–100%] | 73% | 66% | 100% [72%–100%] | 71% | 50% | 1.0 |
| dev | planner | 10 | 100% [72%–100%] | 75% | 68% | 100% [72%–100%] | 74% | 52% | 1.0 |
| all | original | 18 | 100% [82%–100%] | 70% | 63% | 100% [82%–100%] | 74% | 51% | 1.0 |
| all | planner | 18 | 100% [82%–100%] | 70% | 64% | 100% [82%–100%] | 76% | 52% | 1.0 |

## Broad questions — precision

| mode | source | n | prec@5 | prec@10 |
|---|---|---|---|---|
| strict | original | 5 | 84% | 84% |
| strict | planner | 5 | 84% | 84% |
| lenient | original | 5 | 92% | 94% |
| lenient | planner | 5 | 92% | 94% |

## Planner rewording vs original (strict recall@5)

- better: N19
- worse: none
- same: 17 questions

**Audit N20**: no relevant chunk exists in the corpus — a refusal is the correct answer (judged 2, retrieval status ok).

## Per question (strict)

| ID | split | R | source | recall@5 | recall@10 | prec@5 | first rank | unjudged | note |
|---|---|---|---|---|---|---|---|---|---|
| R14 | dev | 4 | original | 50% | 100% | 40% | 1.0 | 0 | broad |
| R14 | dev | 4 | planner | 50% | 100% | 40% | 1.0 | 0 | broad |
| R16 | dev | 1 | original | 100% | 100% | 20% | 1.0 | 0 |  |
| R16 | dev | 1 | planner | 100% | 100% | 20% | 1.0 | 0 |  |
| R17 | held | 8 | original | 80% | 100% | 80% | 1.0 | 0 | broad |
| R17 | held | 8 | planner | 80% | 100% | 80% | 1.0 | 0 | broad |
| R18 | dev | 4 | original | 50% | 50% | 40% | 2.0 | 0 |  |
| R18 | dev | 4 | planner | 50% | 50% | 40% | 2.0 | 0 |  |
| R19 | held | 2 | original | 50% | 100% | 20% | 5.0 | 0 |  |
| R19 | held | 2 | planner | 50% | 100% | 20% | 5.0 | 0 |  |
| R21 | held | 10 | original | 60% | 60% | 60% | 1.0 | 0 |  |
| R21 | held | 10 | planner | 60% | 60% | 60% | 1.0 | 0 |  |
| R22 | dev | 10 | original | 100% | 100% | 100% | 1.0 | 0 | broad |
| R22 | dev | 10 | planner | 100% | 100% | 100% | 1.0 | 0 | broad |
| R23 | dev | 10 | original | 100% | 100% | 100% | 1.0 | 0 | broad |
| R23 | dev | 10 | planner | 100% | 100% | 100% | 1.0 | 0 | broad |
| R24 | held | 19 | original | 80% | 70% | 80% | 1.0 | 0 |  |
| R24 | held | 19 | planner | 80% | 50% | 80% | 1.0 | 0 | planner intent risk_factors |
| R25 | dev | 9 | original | 60% | 44% | 60% | 3.0 | 0 |  |
| R25 | dev | 9 | planner | 60% | 44% | 60% | 3.0 | 0 |  |
| R26 | dev | 5 | original | 80% | 100% | 80% | 1.0 | 0 |  |
| R26 | dev | 5 | planner | 80% | 100% | 80% | 1.0 | 0 |  |
| R28 | dev | 5 | original | 20% | 40% | 20% | 1.0 | 0 |  |
| R28 | dev | 5 | planner | 20% | 40% | 20% | 1.0 | 0 |  |
| R30 | dev | 3 | original | 67% | 100% | 40% | 1.0 | 0 |  |
| R30 | dev | 3 | planner | 67% | 100% | 40% | 1.0 | 0 |  |
| R31 | held | 2 | original | 100% | 100% | 40% | 1.0 | 0 | fallback |
| R31 | held | 2 | planner | 100% | 100% | 40% | 1.0 | 0 | fallback |
| N16 | held | 9 | original | 80% | 56% | 80% | 1.0 | 0 |  |
| N16 | held | 9 | planner | 80% | 56% | 80% | 1.0 | 0 |  |
| N19 | dev | 5 | original | 20% | 60% | 20% | 1.0 | 0 |  |
| N19 | dev | 5 | planner | 40% | 80% | 40% | 1.0 | 0 |  |
| N20 | held | 0 | original | – | – | 0% | – | 0 | audit |
| N20 | held | 0 | planner | – | – | 0% | – | 0 | audit |
| N21 | held | 1 | original | 100% | 100% | 20% | 1.0 | 0 |  |
| N21 | held | 1 | planner | 100% | 100% | 20% | 1.0 | 0 |  |
| N22 | dev | 17 | original | 60% | 60% | 60% | 1.0 | 0 |  |
| N22 | dev | 17 | planner | 60% | 60% | 60% | 1.0 | 0 |  |
| N25 | dev | 12 | original | 100% | 100% | 100% | 1.0 | 0 | broad |
| N25 | dev | 12 | planner | 100% | 100% | 100% | 1.0 | 0 | broad |
| N27 | held | 5 | original | 60% | 100% | 60% | 1.0 | 0 |  |
| N27 | held | 5 | planner | 60% | 100% | 60% | 1.0 | 0 |  |
| N28 | held | 2 | original | 100% | 100% | 40% | 1.0 | 0 |  |
| N28 | held | 2 | planner | 100% | 100% | 40% | 1.0 | 0 |  |
| N29 | dev | 6 | original | 40% | 50% | 40% | 2.0 | 0 |  |
| N29 | dev | 6 | planner | 40% | 50% | 40% | 2.0 | 0 |  |
| N30 | dev | 4 | original | 75% | 75% | 60% | 1.0 | 0 |  |
| N30 | dev | 4 | planner | 75% | 75% | 60% | 1.0 | 0 |  |
