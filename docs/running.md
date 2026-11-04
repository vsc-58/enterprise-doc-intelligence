# Running the system

How to set it up, build the corpus, run the API (directly, in Docker, or on
Google Cloud Run), run the tests and reproduce every evaluation number. The
[README](../README.md) has the overview and the results.

- [Setup](#setup)
- [Build the corpus](#build-the-corpus)
- [Run the API](#run-the-api)
- [Run with Docker](#run-with-docker)
- [Deploy to Google Cloud Run](#deploy-to-google-cloud-run)
- [Tests](#tests)
- [Reproduce the evaluation](#reproduce-the-evaluation)

## Setup

Requires Python 3.11 and an OpenAI API key. Building the corpus also needs
network access to SEC EDGAR.

```bash
git clone https://github.com/vsc-58/enterprise-doc-intelligence.git
cd enterprise-doc-intelligence
python -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env               # then set OPENAI_API_KEY and EDGAR_IDENTITY
```

`EDGAR_IDENTITY` is the name and email SEC asks automated clients to send
(for example `Jane Doe jane@example.com`). Every other setting has a default in
`src/utils/config.py` and can be overridden in `.env`.

The chat model is set by two settings: `QUERY_MODEL` (planner and answers,
default the dated snapshot `gpt-4o-mini-2024-07-18`) and `EXTRACTION_MODEL`
(default `gpt-4o-mini`). They replace the older `OPENAI_MODEL`; a `.env` that
still sets `OPENAI_MODEL` stops at startup with a message saying which to use.

## Build the corpus

Run from the repository root, in this order. Acquisition, extraction and
embedding skip work already done; the two slicing steps rebuild their files.

```bash
python -m scripts.acquire_filings            # 20 10-K filings + XBRL answer key (SEC)
python -m scripts.build_sections             # cover + Item 8 slices for extraction
python -m scripts.run_extraction --dry-run   # what would be extracted, no spend
python -m scripts.run_extraction             # LLM extraction into SQLite
python -m scripts.build_narrative_sections   # Items 1, 1A, 7, 7A (SEC)
python -m scripts.embed_documents --dry-run  # chunk count and cost, no spend
python -m scripts.embed_documents            # embed into ChromaDB
```

Measured cost of a full build: about $0.16 for extraction and $0.02 for
embeddings. `python -m scripts.acquire_filings 2` acquires only the first two
filings, for a quick check.

## Run the API

```bash
uvicorn src.api.main:app --reload            # http://localhost:8000
```

- `http://localhost:8000/` — a demo page: ask a question, see the answer with
  each figure's quality mark and numbered sources.
- `http://localhost:8000/docs` — Swagger UI with example requests.
- Run **one worker**: the Chroma store is single-process and the usage counters
  are in memory.
- Without a built corpus the server still starts; `/health` returns 503 naming
  the missing part.

| Endpoint | What it does |
|---|---|
| `POST /query` | `{"question": "..."}` → answer text, plan type, per-task status and refusal reason, each figure with its quality flag, numbered sources, planner and chain versions |
| `GET /documents` | The 20 filings held, with ticker, name and fiscal year |
| `GET /export` | `extracted_records` as CSV: each figure beside its quality flag, reason, note and cited line |
| `GET /health` | Readiness of the database, vector store, corpus index and figure guard; 503 names the part that is down |

Status codes: a refusal is a correct answer about the corpus, so it is a 200
with a typed `refusal_reason`. 422 is malformed input; 429 a usage limit (off
locally, on in the hosted demo); 503 not ready, OpenAI unavailable, or
`llm_quota_exhausted` when the demo's OpenAI credit is used up; 504 a timeout
after 30 s. The full contract is [api_contract.md](api_contract.md).

Example (`POST /query`):

```json
{"question": "What was Apple's total revenue in fiscal 2021?"}
```

returns, among other fields:

```json
{
  "answer": "Apple Inc.'s total revenue for fiscal year 2021 (period ending 2021-09-25) was $365.82 billion.",
  "plan_type": "single",
  "tasks": [{"intent": "financial_metric", "tool": "sql", "status": "success",
             "figures": [{"ticker": "AAPL", "value": 365817000000.0,
                          "quality": {"verdict": "verified", "reason": "consistent", "note": null}}]}]
}
```

Asking for a year the corpus does not hold returns a 200 refusal: "I hold one
Apple Inc. filing, for fiscal year 2021. I don't have 2022."

## Run with Docker

Needs Docker with Compose, a `.env` file (as in Setup) and the built corpus in
`db/`. The image holds the code and the pinned requirements; the corpus is
mounted from `./db`, not copied into the image.

```bash
docker compose up --build -d          # build the image and start the API
docker compose logs -f app            # "api_started ... ready=True" when the corpus loaded
curl http://localhost:8000/health     # 200 with document and chunk counts
docker compose down                   # stop
```

- Same endpoints as above, on `http://localhost:8000`.
- One uvicorn worker, as in "Run the API", for the same two reasons.
- `./db` is mounted read-write. The API only reads the corpus, but Chroma writes
  to its own files whenever it opens a store, so a read-only mount does not
  work. The container runs as UID 1000, which must be able to write `./db`.
- Back up `db/` before the first run on a new machine: the store is then
  opened, and written, by a different operating system.
- Stop the container before running any pipeline script on the host: the Chroma
  store supports one process at a time.
- Without a corpus the container still starts; `/health` returns 503 naming the
  missing parts, and the container reports itself unhealthy.
- The image is 1.44 GB unpacked (333 MB compressed in the registry), most of it
  the dependencies of edgartools and chromadb (pyarrow, onnxruntime,
  kubernetes). The API imports neither pipeline; a smaller image would need a
  second, API-only requirements file.

## Deploy to Google Cloud Run

One script builds the local image, builds the hosted image on top of it (adding
the corpus), checks what the image contains, pushes it, deploys it and compares
the live `/health` versions with the image's:

```bash
bash scripts/deploy_cloud_run.sh all     # or one step: build, check, push, deploy, verify
```

It needs a Google Cloud project with Cloud Run, Artifact Registry and Secret
Manager, a runtime service account allowed to read the two secrets, and
`gcloud auth configure-docker` for the registry. Every setting and flag is
explained in the script itself; the reasons are in
[design_notes.md, section 9](design_notes.md#9-hosting). On Windows, run it in
Git Bash; it calls `gcloud.cmd` itself.

Read the service's logs with `gcloud logging read jsonPayload.message=<event>`
(for example `api_started` or `rate_limited`). `gcloud run services logs read`
shows only plain-text lines, so it prints the app's JSON lines without their
content.

## Tests

```bash
pytest -q
```

442 tests, all without network, database or a real API key: LLM calls are
stubbed. GitHub Actions runs them, and ruff, on every push and pull request.
The settings class still requires `OPENAI_API_KEY` and `EDGAR_IDENTITY`
to be present, so set them (any value works) in `.env` or the environment.
`tests/test_prompt_versions.py` checks that the prompt versions in the code are
the ones the committed evaluation results were measured on.

Coverage of `src/`, printed by CI: 72% of lines. By area: the API 99.5%, the
query path and RAG 91%, utilities 98%, storage 57%, evaluation code 68%. The
offline acquisition and extraction code is at 10.5%: it calls SEC and OpenAI,
and is checked instead by the evaluation runs against the XBRL answer key.

## Reproduce the evaluation

These call OpenAI and need the built corpus. The committed results in
`data/eval/` are what the numbers above come from.

```bash
python -m scripts.run_eval                   # extraction: five strategies, ten filings
python -m scripts.run_heldout_eval           # extraction: the ten held-out filings
python -m scripts.run_routing_eval           # planner accuracy, 3 runs
python -m scripts.run_execution_eval         # oracle and end-to-end execution, 3 runs
python -m scripts.run_retrieval_eval --routing data/eval/routing_results_0dacd92a.json
python -m scripts.run_attribution_eval --execution data/eval/execution_results_0dacd92a_79177d36.json
```

`0dacd92a` and `79177d36` are the first eight characters of the planner and
chain versions, shown by `GET /health` and written into every result file name;
a prompt or model change gives new ones. Before
any paid call, the routing, execution and attribution scripts make one tiny
call and refuse to run unless OpenAI serves the configured model under exactly
that name (a dated snapshot, not an alias); the served name is recorded in each
result file. `python -m scripts.diagnostic.probe_served_model` shows the same
check for every configured chat model.

Extraction responses are cached by a hash of strategy, model, prompt and input
text (`data/eval/raw_outputs/`), so re-scoring the strategies costs nothing.
