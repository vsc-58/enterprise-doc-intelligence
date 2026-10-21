# API Contract — Phase 6

What the API accepts, what it returns, and which status code means what.
Written before any code (Phase 6, step 2). Steps 3–9 build to this document;
if the code and this document disagree, one of them is wrong.

Sources followed: the code on `main`, then `Phase_5c_summary.md` and
`decisions.md`, then `02_project_phases.md`. Where `02_project_phases.md`
differs (route_taken, `/ingest`, `/extract`, `/query` filters), this document
wins and the reason is in D54.

---

## 1. Scope

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Answer one question through the existing query path |
| `/documents` | GET | List the filings the corpus holds |
| `/export` | GET | `extracted_records` as CSV, with D29 quality columns |
| `/health` | GET | Readiness: database, vector store, corpus index |
| `/docs`, `/openapi.json` | GET | Swagger UI and the OpenAPI schema (FastAPI built-in) |

**Not in the API (D54):** `/ingest`, `/extract`. Acquisition, extraction and
embedding stay as the existing scripts. Reasons: they call SEC and paid
OpenAI; they write to Chroma, whose local store is single-process, while
`/query` reads it; and they run for minutes, which would need 202 + background
jobs. The API is **read-only**.

**No path version prefix.** One client (Swagger) and no external consumers.
Behaviour versions are reported per response (`planner_version`,
`chain_version`). Adding `/v1` later is one `APIRouter(prefix=...)` line.

**One worker.** Chroma's local store is single-process, and the corpus index
and store caches live per process.

---

## 2. The query path the API calls (unchanged)

```
QueryRequest (validated by Pydantic — 422 if malformed)
  → await asyncio.to_thread(plan_query, question)        1 LLM call
  → await execute_plan(plan, index)                       tasks concurrent
  → assemble(plan, outcomes)                              pure Python
  → QueryResponse built from plan + outcomes + answer
```

All inside `asyncio.timeout(settings.QUERY_TIMEOUT_SECONDS)`.
`index` comes from `app.state.index` (built once in the lifespan) via
`Depends(get_index)`.

`AssembledAnswer` alone is not enough: it has `text`, `sources`, `statuses`.
Figures with quality flags live in `TaskOutcome.metric_result`, refusal reasons
in `TaskOutcome.resolution_failure` and `TaskOutcome.narrative_result.refusal`,
and the plan type in `QueryPlan`. The route reads all three.

---

## 3. `POST /query`

### Request

```json
{"question": "What was Apple's total revenue in fiscal 2021?"}
```

| Field | Type | Rule |
|---|---|---|
| `question` | string | Whitespace stripped, then 1 ≤ length ≤ `QUERY_MAX_CHARS` (500) |

- **Extra fields are rejected** (`extra="forbid"`). `{"question": ..., "company": "AAPL"}`
  is a 422, so the dropped filter (D54) fails loudly instead of being ignored.
- The longest question in the 109-question eval set is 83 characters, so 500
  leaves room while bounding cost per call.

### Response — 200

```json
{
  "question": "What was Apple's total revenue in fiscal 2021?",
  "answer": "Apple Inc.'s total revenue for fiscal year 2021 (period ending 2021-09-25) was $365.82 billion.",
  "plan_type": "single",
  "unsupported_reason": null,
  "tasks": [
    {
      "task_id": "t1",
      "intent": "financial_metric",
      "tool": "sql",
      "status": "success",
      "refusal_reason": null,
      "figures": [
        {
          "ticker": "AAPL",
          "company_name": "Apple Inc.",
          "metric": "total_revenue",
          "value": 365817000000.0,
          "fiscal_year": 2021,
          "fiscal_year_end": "2021-09-25",
          "quality": {"verdict": "verified", "reason": "consistent", "note": null}
        }
      ],
      "flagged_not_shown": [],
      "excluded_null_count": 0,
      "mixed_years": false,
      "findings": []
    }
  ],
  "sources": [],
  "planner_version": "d4bc8e36…",
  "chain_version": "b1d8ee5c…",
  "request_id": "6f1c2a9e0b7d4e1f",
  "latency_ms": 1840
}
```

| Field | Source | Notes |
|---|---|---|
| `answer` | `AssembledAnswer.text` | Everything the user reads, Sources list included. Unchanged. |
| `plan_type` | `QueryPlan.plan_type` | single, hybrid, clarification, unsupported, out_of_scope |
| `unsupported_reason` | `QueryPlan.unsupported_reason` | Only for `unsupported` |
| `tasks` | plan tasks + outcomes, plan order | Empty for clarification / unsupported / out_of_scope |
| `tasks[].status` | `TaskOutcome.status` | success, refused, failed |
| `tasks[].refusal_reason` | see §3.1 | `null` when status is success |
| `tasks[].figures` | `metric_result.rows` | SQL tasks only. `quality` always present (D29) |
| `tasks[].flagged_not_shown` | `metric_result.flagged_not_shown` | Rankings/filters only (D53) |
| `tasks[].excluded_null_count`, `mixed_years` | `metric_result` | Rankings/filters (D33) |
| `tasks[].findings` | `narrative_result.findings` | Corpus-wide RAG only: `{ticker, status}` per company |
| `sources` | `AssembledAnswer.sources` | `{number, ticker, company_name, fiscal_year, section, chunk_index, label}` |
| `planner_version`, `chain_version` | `planner_version()`, `chain_version()` | Which prompts produced this answer |
| `request_id`, `latency_ms` | middleware | Also in the `X-Request-ID` response header |

**Never in the response:** `QueryPlan.rationale` (docstring: never shown to
the user), `MetricResult.sql_preview`, retrieved chunk text, any XBRL value.

`quality.note` is `QualityFlag.note()` — the same sentence the answer text and
the CSV use, so the three never drift apart.

### 3.1 `refusal_reason` — one field, three sources

| Situation | Value | From |
|---|---|---|
| Company not in corpus | `non_corpus_company` | `ResolutionFailure` |
| Year not held | `year_not_held` | `ResolutionFailure` |
| Ambiguous company / no company named / not embedded | `ambiguous_company`, `no_company_mention`, `company_not_embedded` | `ResolutionFailure` |
| Section absent, not addressed, figure requested, uncited | `section_absent`, `not_addressed`, `figure_requested`, `uncited` | `NarrativeRefusal` |
| SQL filter threshold unreadable ("fifty billion dollars") | `threshold_unparsed` | API mapping: SQL task, refused, no resolver failure, intent `financial_filter` |
| Any other refused SQL task without a resolver failure | `sql_not_runnable` | API mapping (defensive; not reachable today) |

Mapped in the API layer, so no Phase 5 code changes (D54). A `failed` task has
`refusal_reason: null`: failure is not a refusal.

### Response — refusal (still 200)

```json
{
  "question": "What was Apple's revenue in 2022?",
  "answer": "I hold one Apple Inc. filing, for fiscal year 2021. I don't have 2022.",
  "plan_type": "single",
  "tasks": [{"task_id": "t1", "intent": "financial_metric", "tool": "sql",
             "status": "refused", "refusal_reason": "year_not_held",
             "figures": [], "flagged_not_shown": [], "excluded_null_count": 0,
             "mixed_years": false, "findings": []}],
  "sources": [], "...": "..."
}
```

---

## 4. `GET /documents`

Read from the corpus index in `app.state` — no database call per request.

```json
[
  {"ticker": "AAPL", "company_name": "Apple Inc.", "cik": "0000320193",
   "fiscal_year": 2021, "fiscal_year_end": "2021-09-25",
   "is_extracted": true, "is_embedded": true}
]
```

`company_name` passes through `display_name()`, as in answers. Ordered by
ticker. No query parameters: 20 rows.

---

## 5. `GET /export`

- `Content-Type: text/csv; charset=utf-8`,
  `Content-Disposition: attachment; filename="extracted_records.csv"`.
- One row per document: **latest successful extraction only**
  (`latest_successful_records()`, D31).
- Built with pandas; a plain `Response`, not streaming — 20 rows.
- **Never reads `ground_truth.json` or XBRL** (edgartools boundary rule).

| Columns | Notes |
|---|---|
| `ticker, company_name, cik, fiscal_year, fiscal_year_end, auditor_name` | identity |
| for each of the 5 metrics: `<m>`, `<m>_quality`, `<m>_quality_reason`, `<m>_quality_note`, `<m>_source_line` | value in actual dollars + D29 flag + its sentence + the cited line |
| `extraction_strategy_used, prompt_version, model_name, extracted_at` | provenance |

Test: the column list is asserted in `tests/test_api.py`; the values are
checked by hand against DB Browser (step 10).

---

## 6. `GET /health`

No LLM call. Checks run in a thread.

```json
200 {"status": "ok",
     "checks": {"database": "ok", "vector_store": "ok", "index": "ok"},
     "documents": 20, "chunks": 1750,          (example values)
     "planner_version": "…", "chain_version": "…"}

503 {"status": "not_ready",
     "checks": {"database": "ok", "vector_store": "unavailable", "index": "ok"},
     "documents": 20, "chunks": null, "...": "..."}
```

- `database`: `SELECT 1` through the existing engine.
- `vector_store`: `collection_count()`.
- `index`: the lifespan built a non-empty corpus index.

**Startup never crashes on a missing store (D54).** If `load_index()` raises
(empty database) or the store cannot open, the lifespan logs the error and
starts anyway; `/health` returns 503 naming the failed part, and `/query`
returns 503 `not_ready`. On Hugging Face this shows *why* in a browser instead
of a crash loop.

---

## 7. Status codes

| Code | When | Body |
|---|---|---|
| 200 | Any answer: success, refusal, clarification, unsupported, out_of_scope, partial (some tasks failed) | `QueryResponse` |
| 404 | Unknown path | FastAPI default |
| 422 | Body malformed: missing/empty/too long `question`, extra fields, not JSON | FastAPI's validation format (kept: Swagger understands it) |
| 429 | Rate limit or daily budget exceeded — **off by default** (§9) | error envelope |
| 500 | Unexpected exception | error envelope, `internal_error` |
| 503 | Not ready (§6); OpenAI unreachable/timeout/rate-limited/auth on the planner call; every task in the plan `failed` | error envelope: `not_ready`, `llm_unavailable`, `service_degraded` |
| 504 | `QUERY_TIMEOUT_SECONDS` exceeded | error envelope, `timeout` |

**Refused is not failed.** `refused` means the system worked and the corpus
cannot answer (year not held, section absent, …) — a correct answer. `failed`
means something broke. Only `failed` can make a response 5xx.

| Task statuses in the plan | Status | Why |
|---|---|---|
| all `success` | 200 | answered |
| all `refused` | 200 | each part's own refusal sentence, e.g. year not held + section absent |
| `success` + `refused` / `failed` (any mix with one success) | 200 | partial answer, each part explained (5A design) |
| `refused` + `failed` | 200 | the refused part is a complete answer; the failed part says it could not be answered; `tasks[].status` tells a client which is which |
| all `failed` | 503 `service_degraded` | nothing was answered and nothing is known about the corpus; a retry may work |

No generic "not answerable from this corpus" summary line is added on top:
each refused part already carries its own specific sentence (typed refusals,
Python-written), and a generic line would hide which reason applied.

**422 is checked before readiness.** A malformed body is a 422 even while the
server is not ready: validation costs nothing, readiness may touch the disk, and
"try again in 5 s" would be wrong advice for a request that can never succeed.
(FastAPI runs dependencies before body validation, so `/query` checks readiness
inside the handler — found running the server in step 6.)

**OpenAI down → 503, not 502.** 502 says "an upstream gave a bad response";
the user-facing meaning (try again later) is 503's. Stated default.

### Error envelope (every non-422 error)

```json
{"error": "timeout",
 "message": "The question took too long to answer. Try a narrower question.",
 "request_id": "a83b0c1d22e94f70"}
```

`message` is fixed text written in our code. Exception text, stack traces,
paths, SQL and keys go to the logs only.

---

## 8. Timeouts (D54)

| Setting | Value | Applies to |
|---|---|---|
| `QUERY_TIMEOUT_SECONDS` | 30 | Whole `/query` request → 504 |
| `OPENAI_TIMEOUT_SECONDS` | 10 | Each OpenAI call: planner, chain answer, each verdict, embeddings |
| `OPENAI_MAX_RETRIES` | 1 | Same clients |

Rule: `OPENAI_TIMEOUT_SECONDS × (OPENAI_MAX_RETRIES + 1) < QUERY_TIMEOUT_SECONDS`
(10 × 2 = 20 < 30). A hung planner call — the one that runs in a thread and
cannot be cancelled — finishes before the request times out, and a retry still
fits. Async chain calls are cancelled by the request timeout.

Consequences to know:
- Touches `planner._get_llm()`, `chain._get_llm()` and `vector_store._embeddings()`.
  Version hashes do not include these parameters, so `planner_version` and
  `chain_version` are unchanged.
- Offline scripts use the same clients. `embed_documents` sends batches of 100
  chunks; if a batch ever needs longer, run it with `OPENAI_TIMEOUT_SECONDS=60`
  in the environment.
- A verdict call that times out marks that one company `failed` ("Could not
  be checked: GS"); the survey continues.

---

## 9. Settings added in Phase 6

| Setting | Default | Purpose |
|---|---|---|
| `QUERY_MAX_CHARS` | 500 | Request validation |
| `QUERY_TIMEOUT_SECONDS` | 30 | §8 |
| `OPENAI_TIMEOUT_SECONDS` | 10 | §8 |
| `OPENAI_MAX_RETRIES` | 1 | §8 |
| `READINESS_RETRY_SECONDS` | 5 | Minimum gap between readiness re-tries; `Retry-After` on 503 |
| `CORS_ALLOW_ORIGINS` | `[]` | No cross-origin browser access; Swagger is same-origin |
| `RATE_LIMIT_PER_MINUTE` | `None` (off) | Per client IP, `/query` only → 429. Phase 8 switches on |
| `DAILY_QUERY_BUDGET` | `None` (off) | Max `/query` calls per UTC day → 429. Phase 8 switches on |

### Usage limits (D55, `src/api/limits.py`)

- **What is counted:** `/query` requests that are about to spend money — checked
  after body validation and readiness, immediately before the planner. A 422 or
  a not-ready 503 is not counted. A request refused by one limit is not counted
  against the other. `/health`, `/documents` and `/export` are never limited.
- **Why requests, not tokens:** a request's cost has a ceiling (≤ 3 tasks,
  ≤ `QUERY_MAX_CHARS`, the costliest kind about $0.004), so a budget of N
  requests is a dollar cap of about N × $0.004. Count tokens if that ceiling
  ever goes.
- **Per client:** sliding 60-second window per client IP. `Retry-After` is the
  seconds until that client's oldest counted request leaves the window.
- **Daily:** all clients together, per UTC day. `Retry-After` is the seconds
  until UTC midnight.
- **Client identity** is the address the server sees. Behind a proxy (the
  Hugging Face Space) run uvicorn with `--proxy-headers --forwarded-allow-ips`
  so it is the real client's; to confirm in Phase 8 with two devices.
  `X-Forwarded-For` is never parsed in code: any client can send it.
- In-memory counters — correct only because there is one worker (§1) — and
  reset on restart.
- The OpenAI dashboard spend limit stays the final guard (it also covers a
  leaked key, which nothing in this code can).

---

## 10. Files (steps 3–9)

```
src/api/models.py          request/response models (this document, §3–§6)
src/api/main.py            app, lifespan, middleware, exception handlers, CORS
src/api/context.py         request-id header name, request timer
src/api/errors.py          API error types and fixed messages
src/api/readiness.py       which data parts are usable; warm-up and re-try
src/api/dependencies.py    per-endpoint readiness checks, limiter access
src/api/limits.py          per-client rate limit and daily budget (§9)
src/api/mapping.py         plan + outcomes + answer → QueryResponse (pure, tested)
src/api/routes/query.py
src/api/routes/documents.py
src/api/routes/export.py
src/api/routes/health.py
tests/test_api.py          TestClient, stubs, no network or database
```

`mapping.py` is separate so the largest piece of logic (§3, §3.1) is a pure
function tested without FastAPI.
