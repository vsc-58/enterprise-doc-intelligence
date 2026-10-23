# Design Notes

Why the system is built the way it is. Each note states a decision, the reason
for it, and what it costs. Where a decision was tested, the evidence is given;
the full measurements are in `docs/prompt_strategy_results.md` (extraction) and
`docs/query_eval_results.md` (query path), and the API behaviour is specified in
`docs/api_contract.md`.

Contents:
1. [Acquisition and ground truth](#1-acquisition-and-ground-truth)
2. [Extraction](#2-extraction)
3. [Retrieval corpus](#3-retrieval-corpus)
4. [Query planning](#4-query-planning)
5. [Answering: figures, citations and refusals](#5-answering-figures-citations-and-refusals)
6. [Evaluation method](#6-evaluation-method)
7. [API](#7-api)

---

## 1. Acquisition and ground truth

### XBRL is the answer key, never the output

edgartools is used for two things only: acquiring a filing and its text, and
reading the filing's XBRL financials as ground truth for evaluation. XBRL values
never reach `extracted_records`, query answers or `/export`. The structured
figures the system serves come from the LLM extraction pipeline, because that
pipeline is what is being built and measured. If XBRL fed the output, the
extraction scores would be scoring the library against itself.

### Select the 10-K by fiscal period end, not by filing year

edgartools' `year` filter means the calendar year the filing was *made*. A 10-K
is filed after its fiscal year ends, so for every December year-end company the
filter returns the previous year's report. "JPM 2023" would silently save the
FY2022 10-K, and because bank figures move little from year to year, the error
could pass unnoticed. Filings are selected by `period_of_report`: `year` means
the calendar year in which the fiscal period ends. One shared selector serves
both acquisition and ground-truth capture, so the two always resolve the same
filing.

Trade-off: a retailer whose own label differs from the period-end year (Target
calls its year ending January 2023 "fiscal 2022") is stored under the period-end
year. Consistency matters more than each company's label, because the value is
a join key.

### Read ground truth by exact XBRL concept

The convenient accessors (`get_total_liabilities()` and similar) substitute a
resembling value when the exact concept is missing. For Amazon, which tags no
standalone total-liabilities line, the accessor returned total assets — a wrong
value that passes any null check. The answer key instead reads each field by its
exact concept, consolidated rows only (no segment or abstract rows), and stores
null when the concept is absent. An answer key must never hold a
confidently-wrong value; a null is honest and is excluded from scoring.

### Synonym lists hold true synonyms only, total line first

A single concept per field nulled valid data: 8 of 20 companies tag
`Revenues` rather than the newer contract-revenue concept, and two tag
`ProfitLoss` rather than `NetIncomeLoss`. Each field therefore has an ordered
list of verified synonyms, first match wins. The list never holds a
resembling-but-different concept (`…IncludingAssessedTax` adds sales tax, so it
is not a synonym).

The order matters. The first list put the contract-revenue concept first, and
Walmart tags both: the answer key captured net sales, a component, instead of
total revenues. The order was reversed (`Revenues`, the total line, first) and
the whole key regenerated from code — never hand-edited, because an answer key
adjusted after the model disagreed is no longer independent. Three documents
changed, each checked against the printed income statement. The tell was that
every strategy failed the same document identically: a prompt problem varies by
prompt, an answer-key problem does not.

Limitation: the harness catches disagreement, not shared error. A key value that
is wrong in the same way the model is wrong scores as correct.

### Some fields are null on purpose

- **Bank total revenue** (JPM, GS): a bank's top line is revenue *net* of
  interest expense, a different concept from an industrial company's total
  revenue. Scoring against it would measure whether the model picked the same
  bank-specific concept, not whether it extracted correctly. (BAC tags a
  standard revenue concept, so it is populated: the rule is "don't chase the
  bank concept", not "banks must be null".)
- **Total liabilities where untagged** (Amazon, Target, Walmart, Intel): it
  could be derived as assets minus equity, but that would put the project's own
  arithmetic into the key that is meant to check the model.

### Acquisition is all-or-nothing per document

A `Document` row is written only after both the filing text and its ground truth
are saved; on a partial failure the orphaned text file is deleted. A committed
row therefore means "fully acquired", re-runs skip only completed documents, and
an interrupted batch (Ctrl-C) leaves both stores in step. A separate status
column was rejected: it needs a second write after the work, and that write can
fail.

---

## 2. Extraction

### Slice sections in Python before the model call

The prompt needs the financial statements, not the whole filing. Telling the
model to "find Item 8" cannot reduce what was already sent: Pfizer's FY2023
filing is 136,215 tokens, above gpt-4o-mini's 128,000-token window. Slicing in
Python cuts a typical document by 60–66%. Letting the model request sections
through a tool was also rejected for a second reason: each strategy would then
see different input, and the strategy comparison would measure prompt plus tool
use instead of the prompt.

### Trust edgartools' Item 8 slice only when it is plausible

edgartools' item lookup silently collapses Item 8 to its heading on 4 of 20
filings (JPM, NFLX, QCOM, XOM — 54 to 153 tokens). Item 8 is accepted only if it
is at least edgartools' own minimum size (26,136 characters) and Item 8 plus the
cover page fits the model input budget; otherwise the full filing text is used.
Every input carries a provenance tag (`edgartools_item8`, `fallback_fulltext`,
`fallback_fulltext_oversize`) so the effect of the fallback is measurable — and
it cost the rule-based strategies nothing (5/5 on the fallback document).

A regex parser was rejected: tested against the anchor positions, it lands on
the wrong region for JPM and Intel and returns plausible wrong text. An earlier
60,000-token upper limit was removed: it fired three times with zero true
positives, and its original evidence compared token counts from two different
edgartools text renderings that differ by about 58% on identical content. The
real constraint is fitting the model window, which the budget check tests
directly.

### Only unscored metadata comes from edgartools

`cik` and `filing_date` are not reliably in Item 8, and a required `cik` field
would force the model to invent one — corrupting the join key used for scoring.
They come from the acquisition record. `company_name` and `fiscal_year_end` are
known to edgartools too, but they are scored, so they stay LLM-extracted:
injecting them would make two of seven fields trivially correct. The rule:
edgartools may supply a field only if that field is not scored.

### Five strategies, including a genuinely naive one

The planned strategy set was replaced. "Schema-guided" and "section-aware" turned
out to be constants (structured output makes every arm schema-guided; the slicer
makes every arm section-targeted), and a variable applied to every arm is not a
variable. The first run then showed the rule-based arms tied at 1.00, so a naive
arm (S0, task only) was added below them. Without it, the experiment could only
compare scaffolding on top of an already-solved problem; with it, the headline
result appeared: the three rules take numeric recall from 0.50 to 1.00.

S0 is not a pure control: the schema's field descriptions reach the model
regardless. It isolates the prompt rules, not instruction in general.

### S3 chosen over tied arms for verifiable citations

S1, S3 and S4 tied on accuracy. On a tie the deciding criterion has to be
something other than the score. S3 returns a verbatim source line for every
figure, which allows a check at write time, makes human review a one-line read,
and carries cell-level lineage into the structured store. It costs 1.3% more.
Ten filings cannot show that S1 will keep scoring 1.00; S3 makes each individual
answer checkable regardless.

### Two independent citation checks per figure

- **Evidence present**: the quoted line occurs in the source text (catches
  fabrication). Calibrated to ignore whitespace and `$`, because the text
  extractor runs labels into values (`total assets$85,501`) and no faithful quote
  could reproduce that. Before calibration it read 0.96, flagging two correct,
  faithfully-quoted Visa figures.
- **Evidence consistent**: the stored value occurs in the quoted line (catches a
  real line attached to the wrong value).

Across the 20 filings the checks found eight lineage defects in 100 fields, on
documents where every value was correct: four derived total-liabilities values
cited to the "liabilities and equity" line, two values cited to a neighbouring
concept's line, and two quotes completed from other lines (JPM's three-year row
ends with FY2021 total assets). A failed check flags the figure, never the
record.

Each check has a blind spot: evidence present misses a wrong value in a real
line; evidence consistent allows for the printed scale, so it cannot see a
1,000x unit error; ground truth covers five fields only where XBRL tags exist.
The blind spots do not overlap.

### Business description and risk factors are not stored

Both fields are in the extraction schema, but the model sees cover + Item 8, not
Item 1 or 1A. Inspection showed it filling them from Item 8 notes —
`primary_risk_factors` came back as note headings ("Uncertain tax positions").
Real text, wrong field, and neither field was scored or checked. They are dropped
at the write by an allowlist. They stay in the schema because the response cache
key does not hash the schema: removing them would change the model's output while
cached responses kept being served.

### The unused half of the corpus is held out

The ten filings not used to select S3 were scored once, report-only, with the
no-revision rule committed before the numbers were seen. Numeric F1 fell from
1.00 to 0.90; three of the four wrong values were the unit-scale trap on bank
headers the selection set never contained. The prompt was not revised: a fix
would need a new held-out set to be measured. Details in
`docs/prompt_strategy_results.md` section 7.

---

## 3. Retrieval corpus

### A separate slicer for narrative sections

Item 1, 1A, 7 and 7A are sliced by their own module rather than by extending the
extraction slicer. The extraction cache key hashes the input text, so any change
to what the extraction slicer returns would re-bill the strategy runs and
invalidate the measured held-out result, which describes a cover + Item 8 input.

### Item 8 is not in the vector store

The vector store holds narrative sections only; financial figures come from the
structured store or not at all. A financial question wrongly routed to retrieval
would otherwise find a balance sheet and let the model state a figure outside
every guarantee the SQL path gives. Item 8's boundaries are also the least
reliable in the corpus. Exclusion narrows the hole but does not close it: MD&A
summary tables carry the same figures, which is why the figure guard exists
(section 5).

### Section guards set from measurement, with no fallback

A slice is rejected if it is under 200 tokens or overlaps another item. The floor
sits in a measured gap: collapsed slices are 45–157 tokens, the smallest real
section is 596. Overlap is checked both ways because over-capture runs both ways
(META's Item 1 sits inside its Item 1A; QCOM's Item 1 contains its Item 1A).

A full-text fallback was planned and dropped once measured: Item 1A was present
on 20 of 20 filings, so no document could vanish, and a fallback would only
duplicate text. A failed item is simply absent, and the system says so. Intel,
JPM and Exxon have no captured Item 7; seven filings have no Item 7A, six of
them because they fold market risk into Item 7 by cross-reference.

### Mislabelled sections are fixed by data, not code

QCOM's and META's business descriptions are stored under the Item 1A label. A
small alias table maps "Item 1" to the label that holds it for those two
tickers; the retriever applies it generically and contains no company-specific
code. Citations show the stored label, the honest location of the text.

### Chunking is local and repeatable

Section slices are saved to disk once, so re-chunking costs only the embeddings
and needs no SEC access. Chunk ids hash the document, section, position and
text. Re-chunking deletes a document's chunks by metadata filter before
inserting, so a run that produces fewer chunks leaves no orphans. A hash of the
chunking and embedding parameters is stored with the collection and compared on
every open, so a store built with different settings is reported, not silently
used.

### Cosine distance kept; a distance threshold is not a refusal guard

Cosine was chosen partly to allow an out-of-scope threshold. Measured on the
built store, out-of-scope questions do not separate: a figure question that
belongs to SQL scored closer (0.3446) than the in-scope headline question
(0.4046). Any threshold that rejects it also rejects the demo. The benefit was
withdrawn; cosine stays because all three metrics rank OpenAI's unit-length
vectors identically and a bounded score is easier to read. Refusal comes from the
planner's plan types and the chain's typed outcomes instead.

---

## 4. Query planning

### A planner instead of a binary router

The original design sent each question to one of two paths. "What does Amazon do
and what was its net income?" has no single correct destination, so a router
must silently drop one half — and the half it answers looks correct. A planner
splits a question into 1–3 typed tasks and the application maps each task's
intent to a tool. Single-task plans are the router's behaviour as a special
case. Cost: one planner call (about 1–2 s) before anything runs.

### No dependency graph, calculator or execution modes

The corpus holds one fiscal year per company, so no plan can have a dependency
between tasks. An execution engine whose only reachable path is the empty one is
speculative. Questions that would need one are refused with a typed reason
(`cross_year`, `ratio_or_derived`); that list documents what adding a dependency
graph would unlock once the corpus holds several years.

### The planner is corpus-blind

The planner prompt has no company list, year list or embedding state; it copies
company mentions as written. Corpus membership is decided in one place, the
resolver, against the database. A list in the prompt would be a second
definition of the corpus that drifts when the corpus grows. For evaluation, a
question about a company not held is a correct plan followed by a correct
resolver refusal, not a planner error.

### Intents exist only where execution differs

Each narrative intent searches a set of sections. Intents with identical section
sets were merged (12 intents became 9: 3 SQL, 6 narrative), and a test forbids
two narrative intents from sharing a set. If two labels execute identically, the
routing metric would score a distinction the system cannot act on. Sections are
a union searched in one query and ranked by distance, with a fixed top 5.

### Schema docstrings are prompt surface

`with_structured_output` sends the Pydantic schema, docstrings included, to the
model as the function definition: when first measured, about 1,500 of the
planner's 2,452 input tokens. So `planner_version` hashes the system prompt, the model name and the
schema, and the same applies to the chain and the evaluation judge. A
"documentation-only" edit to a bound schema would otherwise change behaviour
while two results claimed the same version. A test compares the code's version
hashes with those recorded in the result files.

### Prompt revisions follow dev misses only, with a stopping rule

Every prompt change is tied to a named miss on the development questions, and
each revision is checked against held-out questions for shared phrases. Lessons:

- **Prohibitions generalise.** "Never financial_ranking" taught the planner to
  avoid rankings altogether, and "not a clarification" turned "its strategy"
  into a 20-company survey. The fixes were stated as positive discriminators (a
  filter needs a threshold; a question about companies in general differs from
  one about a specific, unnamed company).
- **Enforce in code what the plan's shape reveals.** A ranking that names one
  company would return the corpus-wide winner; the executor turns it into that
  company's lookup. "Net sales" planned as net income is a well-formed plan, so
  only the prompt and the eval can catch it.
- **Stop.** The third version was declared final before it ran, whatever it
  scored, because each further round with held-out misses visible erodes the
  held-out claim. v1's held-out score (82.5%) is the clean out-of-sample number;
  later versions are reported beside it.

### Narrative answers may quote figures other than the five stored ones

The planner and the chain once disagreed about what "figure" meant. The rule now
is the same in both: the five stored financial figures always route to SQL and
are masked from the narrative model; any other number the filing states (R&D
growth, segment revenue, store counts) may be quoted with a citation. The cost:
such a number carries its citation as its only provenance, with no scale check.
This rule is still fragile — R&D and dividend questions are sometimes refused.

---

## 5. Answering: figures, citations and refusals

### No LLM ever receives a stored figure

A prompt rule is a request; a structure is a guarantee. The worst output this
system can produce is a plausible wrong figure under a correct-looking citation.
So the SQL path renders its answer sentence in Python, the narrative chain sees
only retrieved text with stored figures masked, and the final answer is
assembled in Python. There is no synthesizer LLM. Cost: answers read as
assembled parts rather than flowing prose.

### The figure guard: mask first, redact as a backstop

Retrieved text still contains stored figures (Apple's MD&A prints total net
sales). Before the model sees the text, every stored value is masked, with scale
variants (x1,000, /1,000) for figures whose scale is in doubt. Any stored value
that still appears in an answer claim is redacted, not dropped, so the
explanation around it survives. Across nine evaluation runs, masking caught
every stored figure and the redaction backstop never fired. Known gap: a figure
the model computes or rounds by more than 0.5% would escape.

### Typed outcomes, Python-written refusals, citations checked in code

The chain returns a typed outcome (answered, not addressed, figure requested)
and a list of claims, each citing block numbers. Python checks every cited
number against the retrieved set and drops claims left without a valid citation;
sources list cited blocks only. Every refusal sentence is written in Python, so
refusals are measurable without string matching. Inline references the model
adds anyway ("(1)") are stripped in code: changing the prompt only changed their
form.

### Every served figure carries a data-quality flag

Three held-out figures were right in digits and wrong by 1,000x. Each figure is
now flagged verified, unverified or disputed, computed from stored evidence
alone: a figure is disputed when its cited line does not contain it, or when its
implied scale (stored value ÷ the number printed in its cited line) disagrees
with the rest of its document. BAC's four checkable fields split 2–2 between two
scales and are all flagged; GS's operating cash flow is flagged too. The flag
appears in answers, in the API response and in `/export`. The answer key is
never read at query time: if the check ever needed it, it would be an evaluation
metric, not a feature. Limitation: a document scaled uniformly wrong passes.

### Rankings and filters disclose what the flags might change

- If a shown row is flagged, the answer says the **ordering** may be wrong, not
  only that row: a mis-scaled figure is often why the row sorts where it does.
- A company left out of a ranking or filter is named only when its flag is a
  scale flag and a 1,000x correction would put it in the list. Naming every
  flagged company (an earlier rule) listed companies that could not belong, and
  teaches readers to skip the line that matters.
- Companies left out because a figure was not extracted are named, with what it
  means ("may be lower than any figure shown"). BAC's operating cash flow is null
  because the true value is negative, so it is probably the real answer to
  "lowest operating cash flow".
- A missing figure is blamed on the extraction, never on the filing. A lookup
  on a null figure says it "was not extracted from the filing"; a filing with
  no successful extraction at all is refused for figure questions with its own
  reason (`company_not_extracted`) and is named in rankings and filters like a
  null. An earlier version answered "the filing does not report it" — a false
  statement about the document.
- The system never re-sorts on a suspected value: that would present a computed
  value as extracted.

### Mixed fiscal years are stated

The corpus holds one year per company across FY2021–2023, so a cross-company
list can mix years. Every multi-row answer prints each row's year, and adds one
sentence when the years differ, saying the figures are not like-for-like.
Refusing all cross-company comparison would remove the most useful SQL
questions.

### Corpus-wide questions are answered per company

"Which companies flagged interest rate risk?" cannot be answered by a plain top
5: it caps the companies named, and large filers crowd out the rest (top 50
reached only 16 companies). The question is embedded once, searched per company
(2 chunks each), and judged by one verdict call per company, at most five at a
time. Companies holding none of the needed sections are listed as not
searchable, never dropped. A "not found" is worded as "no supporting passage
found in the retrieved text", never "did not disclose": two chunks are not the
filing. Cost: about 24k input tokens (about $0.004) per survey.

### One section fallback, and a test for admitting more

JPM and Exxon have neither Item 7 nor 7A captured, so market-risk questions
about them refused although their risk factors discuss rate and commodity
exposure. Market-risk questions now fall back to Item 1A when the intent's own
sections are absent, and the answer says so. The trigger is decided from the
store before any LLM call, never from the model failing to answer: retrying until
something answers biases the system toward answering. An entry is admitted only
if the fallback answers the same kind of question. MD&A → risk factors fails
that test ("why did margins fall" is not "what might go wrong"), and a test pins
it.

### Names: stored, matched and displayed are three forms

The stored company name is EDGAR's registrant name ("BANK OF AMERICA CORP
/DE/"): it is the acquisition record and is never changed. Filtering uses the
ticker, because no model writes "/DE/" and a failed match would look like a
corpus gap. Matching uses a separate normalised form, and display names are made
at the boundary with a small per-ticker override table. One form per purpose,
each with one owner.

### One definition of "latest extraction"

Extraction keeps every attempt. `latest_successful_records()` is built on a
composable query (latest successful attempt per document), and the SQL path adds
its filters, ordering and limit to the same query, so it is a real parameterised
`SELECT`, and `/export` and the SQL path cannot disagree about which record is
current.

---

## 6. Evaluation method

### Temperature 0 is not determinism

Two identical planner calls returned different plans for the same question.
Every LLM measurement therefore runs three times and reports the mean with a
per-question disagreement count. Intervals use n = questions: repeated runs
measure the model's variability, not a larger sample.

### Labels are frozen; scores are reported raw and adjudicated

The query eval set and the retrieval gold are frozen by SHA-256, and the
harnesses refuse to score an edited file; the judge's reference labels and pass
criteria were written before the judge ran. When an audit finds a label wrong,
it is not edited: the raw score stays primary and an adjudicated score sits
beside it, with each adjudication listed. Editing labels after seeing results
would make every later number unfalsifiable. Expected outcomes are derived from
the section inventory, not intuition: an intuited label measures the labeller.

### Measure each stage separately

Routing is scored on the planner alone. Execution is run twice: an *oracle* mode
that executes the gold plan (downstream quality with routing assumed perfect) and
an end-to-end mode (what a user receives). The gap is the cost of routing
errors. A harness artifact was found this way: the oracle first gave each
narrative task the whole hybrid question, and the chain correctly refused the
figure clause; each task now gets its own clause from a side file, and a
missing clause raises.

### Retrieval is scored at chunk level

Filtering on company and section makes "right company, right section" true by
construction, so it measures the filter. What can fail is which chunks of the
right section reach the five the chain reads. The gold set labels chunk
relevance with a rule fixed before labelling, over a pool of the retriever's top
10 plus keyword hits (standard pooling, so recall is an upper bound). Recall is
capped at min(k, relevant): a question with 17 relevant chunks cannot fit more
than 5 in a top 5.

### Diagnose with a controlled test before fixing

Walmart's liquidity questions had low recall. A test changed one side at a time:
prefixing each chunk with its subsection heading lifted recall@5 from 20% to 80%
(and 40% to 60%); rewording the question in the filing's terms helped less. The
cause is chunks losing their subsection heading. The fix (contextual chunk
headers) needs a full re-embed and is logged, not built.

### Validate a judge before trusting its numbers

Attribution — does each answer sentence's cited text support it? — is scored by
an LLM judge (gpt-6-luna, a different model from the system's gpt-4o-mini, to
reduce self-preference). The judge sees the claim, its company and the cited
text, not the question, so it rules on support, not relevance. Its score is used
only because it passed criteria committed before any judge output existed:

- 30 of 30 seeded errors caught (invented detail, wrong block, wrong company);
- Cohen's kappa 0.97 against reference labels;
- 3.3% label flips across runs (bar: 5%).

Kappa, not raw agreement: 39 of 40 real claims are supported, so a judge that
always said "supported" would score 97.5% agreement while catching nothing. At
that prevalence kappa is unstable too (one disagreement gives 0.66), so kappa is
computed over the test claims plus the seeded errors. That amendment was made
before the first judge run and is recorded. Reference labels were a second
model's (Claude), not a human's, with every disagreement adjudicated by the
developer; the seeds, wrong by construction, are the check that depends on
neither model.

---

## 7. API

### Read-only, one worker, one input field

The API has no `/ingest` or `/extract`: those steps call SEC and paid OpenAI,
write to a single-process Chroma store that `/query` reads, and run for minutes.
They stay as scripts. `/query` takes only `question`; extra fields are a 422,
because a company filter would conflict with the resolver's ownership of
membership and could silently drop half of a two-company question. One worker,
because the Chroma store is single-process and the usage counters are in memory.

### Refusals are answers

A refusal (year not held, section absent, figure requested) means the system
worked and the corpus cannot answer, so it is a 200 with a structured reason.
Only failures produce 5xx: 503 when every task failed or OpenAI is unavailable,
504 on timeout. OpenAI down is 503 rather than 502 because the client's action
is "retry later". The response carries plan type, per-task status, figures with
their quality flags, sources and both version hashes, instead of a single
"route taken" string, because a plan can hold up to three tasks on different
tools.

### Timeouts that nest

Each OpenAI call has 10 s and one retry (20 s worst case) inside a 30 s request
timeout. The planner call runs in a thread that cannot be cancelled, so it must
finish before the request gives up. Equal timeouts were rejected: the per-call
limit could then only fire after the request had already timed out.

### Start anyway; say what is broken

If the database is empty or the vector store cannot open, the server starts,
`/query` returns 503 `not_ready`, and `/health` returns 503 naming the failed
part. On a hosted Space this shows the cause in a browser instead of a crash
loop. `/health` has exactly one check per readiness part, enforced by a test,
after a gap where a missing figure guard produced a 503 whose every check read
"ok".

### Usage limits count requests

The public demo spends the project's OpenAI credit; the risk is the credit
running out the day a reviewer opens it. A per-client rate limit and a global
daily budget (both off by default) count `/query` requests just before the
planner runs, so invalid or not-ready requests cost nothing. Requests rather
than tokens, because a request's cost has a ceiling (at most 3 tasks and 500
characters, about $0.004 for the costliest kind), so N requests is a dollar cap.
Client identity is the address the server sees; `X-Forwarded-For` is never
parsed in code, since any client can send it.

### A demo page at `/`

The first thing a visitor to the Space sees should not be `{"detail": "Not
Found"}`. `/` serves one static page that calls the public endpoints like any
client — no second path into the pipeline — and inserts server text as text,
never as HTML. Swagger stays at `/docs`.
