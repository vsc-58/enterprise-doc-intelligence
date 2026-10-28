# Prompt-Strategy Evaluation Results

**Question:** which prompting approach extracts financial fields from 10-K text
most accurately, at acceptable cost?

**Method:** five strategies, each run over the same ten filings, scored per field
against XBRL ground truth (`data/eval/ground_truth.json`) with a 1e-5 relative
tolerance on numerics. The input text is held identical across all arms — the
prompt is the only variable.

**Winner: S3 (rules + evidence).** Tied on accuracy with S1 and S4, fully
grounded citations, at 1.3% higher cost.

---

## 1. The Strategies

| ID | Name | What varies |
|---|---|---|
| S0 | Naive | Task only. No unit, column, or null-handling rules in the prompt. |
| S1 | Rule-based | Adds three explicit rules (below). |
| S2 | Rules + few-shot | S1 plus two worked examples of reading a flattened table. |
| S3 | Rules + evidence | S1 plus quote-the-source-line-then-normalise. |
| S4 | Rules + orientation | S1 plus structural guidance on where figures sit. |

**The three rules (S1–S4):**
1. **Actual dollar units** — report figures in full units, not the scaled value
   printed under an "in millions" header.
2. **Current year, first data column** — statements print several years side by
   side; take the most recent.
3. **Null, never guess** — a field absent from the text returns null.

Rule 1 exists because the XBRL ground truth is in raw dollars (`383285000000`)
while the filing text prints `$394,328` under a scale header that may sit many
lines away once the table is flattened to text. Without it, a model that reads
the document perfectly scores as wrong on every financial field.

**On what S0 actually controls for:** S0 is a naive *prompt*, not a naive call.
`with_structured_output` sends the Pydantic field descriptions to the model as
the tool schema, and those descriptions already say "in ACTUAL DOLLARS" and
"most recent fiscal year." So S0 inherits some instruction; it isolates the
effect of the *prompt* rules, not of instruction in general.

---

## 2. Evaluation Set

Ten filings, selected for sector spread and for specific known hazards:

| Ticker | Year | Why included |
|---|---|---|
| AAPL | 2021 | Non-December fiscal year end (September) |
| MSFT | 2022 | Non-December fiscal year end (June) |
| GOOGL | 2021 | Tags `us-gaap:Revenues` rather than the contract-revenue concept |
| AMZN | 2023 | Retail; no standalone total-liabilities tag |
| WMT | 2023 | Retail; January fiscal year end |
| JNJ | 2021 | Pharma; 52/53-week year ending early January |
| NFLX | 2023 | Tags in actual dollars, not millions — the unit trap in reverse |
| V | 2022 | Tags `us-gaap:ProfitLoss` rather than `NetIncomeLoss` |
| ADBE | 2022 | Irregular fiscal year end (2022-12-02) |
| INTC | 2023 | Dense filing; no standalone total-liabilities tag |

Fields with null ground truth are scored `n_a` and excluded from the
denominator, so `total_liabilities` runs at n=7 (AMZN, WMT, INTC have no
standalone tag). The value is deliberately not derived as assets minus equity:
that would put the project's own arithmetic into the answer key that is meant
to check the model.

---

## 3. Results

Precision / recall, with the scored denominator. Precision equals recall on
every cell here, so F1 equals both — the model never asserted a wrong value
where it could instead have returned null.

| Field | S0 naive | S1 rules | S2 few-shot | S3 evidence | S4 orientation |
|---|---|---|---|---|---|
| total_revenue | 0.50 (n=10) | **1.00** | **1.00** | **1.00** | **1.00** |
| net_income | 0.50 (n=10) | **1.00** | 0.90 | **1.00** | **1.00** |
| total_assets | 0.50 (n=10) | **1.00** | **1.00** | **1.00** | **1.00** |
| total_liabilities | 0.43 (n=7) | **1.00** | **1.00** | **1.00** | **1.00** |
| operating_cash_flow | 0.50 (n=10) | **1.00** | **1.00** | **1.00** | **1.00** |
| company_name | 0.80 (n=10) | 0.80 | 0.80 | 0.80 | 0.80 |
| fiscal_year_end | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| | | | | | |
| Mean relative error | 5.14e-01 | 3.33e-08 | 2.00e-02 | 3.33e-08 | 3.33e-08 |
| Grounding rate | – | – | – | **1.00** | – |
| Input tokens | 358,030 | 360,410 | 362,310 | 360,880 | 361,937 |
| Output tokens | 1,529 | 1,555 | 1,486 | 2,541 | 1,436 |
| Cost (USD) | $0.0546 | $0.0550 | $0.0552 | **$0.0557** | $0.0552 |

**Provenance split** — numeric fields correct, by input quality. Nine documents
had a clean Item 8 slice; one (NFLX) fell back to the full filing text because
edgartools truncated its Item 8 to a 268-character heading.

| Input | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| Clean Item 8 | 23/42 | 42/42 | 41/42 | 42/42 | 42/42 |
| Fallback full text | **0/5** | 5/5 | 5/5 | 5/5 | 5/5 |

---

## 4. What the Numbers Say

### 4.1 The rules are the entire intervention

Naive recall on the five numeric fields is ~0.50 with a **51% mean relative
error** — not near-misses, but the unit trap firing on roughly half the fields:
values returned as `394328` where the truth is `394328000000`. Three sentences
of explicit instruction take that to 1.00, and the mean relative error falls
seven orders of magnitude to 3.33e-08 (float representation, not misreading).

The marginal cost of the rules is **$0.0004** across ten documents.

### 4.2 Scaffolding above the rules adds nothing

S1, S3 and S4 are identical on every accuracy metric, to three significant
figures. Structural orientation (S4) told the model nothing it had not already
inferred from the section it was given. This is a real result, not a null one:
it says the intervention that mattered was stating the unit and column
conventions, and that further prompt engineering on top of them had no headroom
to work in.

### 4.3 Few-shot examples actively degraded extraction

S2 is the only arm that got a value wrong, and its mean relative error is
2.00e-02 — roughly 600,000× S1's. On this evidence, worked examples made the
model worse.

The failure is Walmart FY2023 `net_income` (clean Item 8 input):

| Extracted | Ground truth | Relative error |
|---|---|---|
| 11,680 | 11,680,000,000 | 0.999999 |

The digits are right and the unit is wrong: S2 returned the printed value in
millions, the exact error the unit rule exists to prevent. Every other S2 field
on the same document was in actual dollars. Source:
`data/eval/results.json` (`S2` → `per_document` → `0000104169_2023`).

A plausible mechanism is that the examples anchored the model on their pattern
rather than the actual table — but this is a hypothesis from a single failure,
not a demonstrated cause, and is stated as such.

### 4.4 Missing instruction and degraded input compound

The sharpest number in the table is S0's **0/5 on the fallback document**.
Given the full 52,744-token filing instead of a clean Item 8, the naive prompt
got nothing right, while every ruled arm got 5/5. Noisy input is precisely where
explicit rules earn their cost. (One document, so this is an observation, not a
measured effect.)

Equally: the ruled arms show **no fallback penalty at all**. The full-text
fallback was built as a safety net against edgartools' silent Item 8
truncation, and on this document it cost nothing.

### 4.5 `company_name` measures normalisation, not extraction

0.80 on every arm, including the naive one. The two failures:

| Extracted | Ground truth |
|---|---|
| MICROSOFT CORPORATION | MICROSOFT CORP |
| INTEL CORPORATION | INTEL CORP |

The model read the filing correctly; XBRL's registered entity name is
abbreviated. **Invariance across arms is the diagnostic** — a prompt problem
varies by prompt.

A Corp/Corporation synonym table would push this to 1.00. It was not added:
fuzzy matching is exactly what corrupted an early version of the ground truth
(`get_total_liabilities()` returning total assets for Amazon), and inflating a
metric by loosening the comparison is not a result. Reported at 0.80 with the
cause stated.

---

## 5. Two Bugs the Harness Caught in Itself

### 5.1 The ground truth was wrong, and the failure pattern proved it

The first complete run showed `total_revenue` at 0.90 — **identically across
every strategy**, failing on the same document. Uniformity across arms was the
tell: a prompt problem varies by prompt, a ground-truth problem does not.

Walmart FY2023, extracted `611,289,000,000`, ground truth `605,881,000,000`.
The filing's income statement prints:

```
Net sales                        605,881
Membership and other income        5,408
Total revenues                   611,289
```

605,881 + 5,408 = 611,289. The answer key had captured **net sales** — a
component — not total revenue.

**Root cause:** the original revenue concept list was ordered
`[RevenueFromContractWithCustomerExcludingAssessedTax, Revenues]`, first match
wins. Walmart tags both; the first is the contract-revenue component, the second
is the total line. The ordering had never been tested against a filer that tags
both, because no earlier document did.

The original rule was "true synonyms only — same economic line." That held for
filers tagging one concept or the other. It silently failed for filers tagging
both, where the two are not synonyms but a subset and its superset.

**Fix:** the candidate order was reversed in `src/eval/ground_truth.py` and the
answer key regenerated from code — never hand-edited, because an answer key
adjusted to match the model is no longer independent. The diff corrected three
documents, each verified against the printed income-statement top line:

| Document | Old | New |
|---|---|---|
| Walmart FY2023 | 605,881,000,000 | 611,289,000,000 |
| Pfizer FY2023 | 50,914,000,000 | 58,496,000,000 |
| Tesla FY2022 | 90,929,000,000 | 81,462,000,000 |

Tesla's value moved down, which first looked like a lost component; the printed
statement confirms 81,462 as the filing's own total. "Take the largest matching
concept" was considered and rejected for that reason: the printed statement is
the arbiter, not a heuristic. Pfizer and Tesla are not in the ten-filing
evaluation set, so the fix did not change the scores in section 3 beyond
Walmart. The previous answer key is kept as `data/eval/ground_truth_PREV.json`.

**Limitation this exposes:** a ground-truth value that is wrong *and* that the
model also gets wrong in the same way would score `correct` and stay invisible.
The harness catches disagreement, not shared error.

### 5.2 The grounding check was miscalibrated

S3's grounding rate first read **0.96** — two failures, both on Visa FY2022
(`total_assets`, `total_liabilities`), both on values that scored **correct**.

The check tested whether the model's quoted `source_line` appears as a substring
of the input. Comparing model quote to source text:

| | Model quote | Source text |
|---|---|---|
| total_assets | `Total assets $85,501 $82,896` | `total assets$85,501 $82,896` |
| total_liabilities | `Total liabilities $49,920 $45,307` | `total liabilities49,920 45,307` |

The text extractor runs labels into their values (`intangible assets, net25,065`
appears the same way), so a *faithful* quote could never contain a space there.
The model had normalised the mangled spacing back to readable form and added a
`$` to match the neighbouring convention. The values, labels and ordering all
matched exactly.

The check was flagging cosmetic normalisation as fabrication. It was loosened to
ignore whitespace and currency symbols on both sides, which still catches an
invented label, digits absent from the source, or a quote lifted from a
different statement — the failures it exists to detect. Grounding then read
**1.00**.

**Stated honestly:** removing all whitespace slightly increases the chance of an
accidental match across a boundary the original did not have. At the length of a
label-plus-digits string this is negligible, but the check is a fabrication
detector, not an exact-provenance proof.

---

## 6. Why S3 Wins

S1, S3 and S4 are indistinguishable on accuracy. The tie is broken on
**operational value, not on the score**:

- S3 returns a verbatim source line for every extracted figure, and 100% of
  those lines were verified present in the source document.
- That enables a grounding check at write time: a citation not found in the
  source means the value was fabricated, and the record can be flagged rather
  than trusted — automatically, with no human involved.
- It is what makes human review economical. A reviewer reads one quoted line
  instead of opening a 100-page filing.
- It is cell-level lineage for an unstructured source: the provenance of each
  figure survives into the structured store.

The cost is **$0.0007 per ten documents** — 1.3% — almost entirely output
tokens (2,541 vs 1,555).

A strategy that shows its work is the safer choice on documents outside this
corpus, where accuracy is unmeasured. Ten filings cannot establish that S1 will
keep scoring 1.00 on the next hundred; S3 makes each individual answer
checkable regardless.

**Consequence for the pipeline:** `ExtractedRecord` stores each figure's quoted
source line and two independent per-field checks, run at write time:
`<field>_evidence_present` (the quoted line occurs in the source text — catches
fabrication) and `<field>_evidence_consistent` (the stored value occurs in the
quoted line — catches a real line attached to the wrong value). A failed check
flags that figure, not the record: discarding four verified figures over one bad
citation loses more than it protects.

---

## 7. Out-of-Sample Check

S3 was selected on these ten filings, so its 1.00 is in-sample. The other ten
filings in the corpus also have XBRL ground truth and were never used for
selection. They were scored once with S3, and it was decided in writing, before
the numbers were seen, that no prompt would be revised in response: revising
would turn them into a second development set and leave no out-of-sample claim.
Results are in `data/eval/held_out_results.json`, kept separate from
`results.json` so the selection evidence stays exactly as it was.

**Numeric mean F1: 1.00 in-sample, 0.90 held-out** — four wrong values and one
missing across 46 scored numeric fields.

| Document | Field | Extracted | Ground truth | Cause |
|---|---|---|---|---|
| BAC FY2022 | total_assets | 3,051,375,000 | 3,051,375,000,000 | Unit scale |
| BAC FY2022 | total_liabilities | 2,778,178,000 | 2,778,178,000,000 | Unit scale |
| GS FY2021 | operating_cash_flow | 921,000 | 921,000,000 | Unit scale |
| XOM FY2022 | total_revenue | 398,675,000,000 | 413,680,000,000 | Concept choice |
| BAC FY2022 | operating_cash_flow | null | −6,327,000,000 | Missing |

- **Three of the four wrong values are the unit-scale trap**, both on banks.
  BAC's balance-sheet header reads "Dollars in millions, except per share
  information; shares in thousands" — two scales in one header — and the model
  applied the thousands. GS's header is the terse "$ in millions". Every header
  in the ten selection filings was a conventional single-scale form, so the
  unit rule had only been tested on the easy phrasings. That is the case for a
  held-out set in one example.
- **XOM** returned "Sales and other operating revenue"; the key holds "Total
  revenues and other income", which the printed statement confirms. The model
  took the line that most literally says revenue.
- **BAC's operating cash flow** is negative in truth. The model returned null
  rather than a wrong figure, which the scoring treats as the safer error.
- `company_name` reads 0.50 held-out for the same reason as in section 4.5:
  EDGAR registrant names ("BANK OF AMERICA CORP /DE/") against the names printed
  on the filing.

No prompt was changed in response. The query path instead flags these figures:
a scale-coherence check compares each figure's implied scale with the rest of
its document, using stored evidence only (never the answer key), and BAC's four
checkable fields and GS's operating cash flow are served as disputed. A prompt
revision for non-standard scale headers would be a new experiment, needing a
new held-out set to be measured honestly.

---

## 8. Limitations

Stated because a result whose boundaries are not stated is not a result.

- **n=10.** Enough to separate 0.50 from 1.00; not enough to separate 1.00 from
  0.98. S1, S3 and S4's tie is a tie at this sample size, nothing stronger.
- **Sector coverage is constrained by the answer key.** Pure-play banks were
  excluded because their top line (`RevenuesNetOfInterestExpense`) has no
  comparable "total revenue" concept; scoring extraction against it would
  measure concept-matching, not extraction.
- **One fallback document.** The 0/5 versus 5/5 contrast is a single data point.
- **`company_name` at 0.80 measures string normalisation**, not extraction.
- **Caching freezes one sample per document.** At temperature 0 the model is not
  bit-deterministic; a 2-point gap between arms would sit inside the noise
  floor. The gaps reported here (0.50 vs 1.00) do not, but S2's single failure
  is one draw, not a distribution.
- **A shared model/ground-truth error is invisible** (see 5.1).
- **The grounding check is a fabrication detector**, not proof that the cited
  line supports the value it is attached to. The pipeline's second check,
  evidence consistency (section 6), covers that gap, and is itself blind to
  unit scale because it allows for the printed scale when matching.

---

## 9. Artifacts

| Path | Contents |
|---|---|
| `data/eval/ground_truth.json` | XBRL answer key, per document |
| `data/eval/results.json` | Per-field, per-strategy precision/recall/F1, denominators, provenance split, per-document scores |
| `data/eval/held_out_results.json` | S3 on the ten filings not used for selection (section 7) |
| `data/eval/ground_truth_PREV.json` | The answer key before the revenue-concept fix (section 5.1) |
| `data/eval/raw_outputs/{strategy}/` | Every model response, cached by content hash — the evidence behind the table |
| `docs/design_notes.md` | Why the extraction pipeline and its evaluation are built this way |

Raw outputs are keyed by a hash of strategy, model, prompt template and input
text, so a changed prompt or re-sliced input invalidates the cache
automatically. Scoring re-runs read from disk at zero API cost — which is how
the ground-truth fix and the grounding recalibration were re-scored without
paying for a single additional call.

**Total cost of the five-arm bake-off: approximately $0.28.**