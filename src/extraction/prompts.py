"""
src/extraction/prompts.py
Prompt templates for the four extraction strategies (the experiment's only
independent variable).

Every template carries three INVARIANT rules — held constant across all arms so
the arms differ only in their intended intervention:
  1. actual-dollar units   (the project's top silent failure)
  2. current year / first data column
  3. null, never guess

Arms:
  S1 zero_shot     : baseline — rules + task, nothing else
  S2 few_shot      : S1 + two worked examples of reading a flattened table
  S3 evidence      : quote-the-source-line-then-normalise (binds the evidence schema)
  S4 context_aware : S1 + structural orientation (what the section is, where figures sit)

Dependencies: langchain-core.
"""

from langchain_core.prompts import ChatPromptTemplate

# --- Invariant rules: identical in every arm ------------------------------

_COMMON_RULES = """\
Rules that apply to every field:

1. UNITS — report every monetary figure in ACTUAL DOLLARS (full units), not the
   scaled figure printed in the table. Financial statements print a scale header
   such as "in millions" or "in thousands" above the numbers. If the statements
   are in millions, multiply by 1,000,000 (394,328 in millions -> 394328000000).
   If in thousands, multiply by 1,000. Return a plain number: no commas, no
   currency symbols, no scale words.

2. CURRENT YEAR / FIRST COLUMN — statements show several fiscal years side by
   side. Extract the value for the MOST RECENT fiscal year, which is the FIRST
   data column after the line label. Do not take a prior-year column.

3. NULL, NEVER GUESS — if a field is not present in the text you are given,
   return null. Do not infer, estimate, average, or fabricate. A null is a
   correct answer when the value is absent; a guessed number is not."""

_TASK = """\
Extract the target fields from the 10-K text below. Return only the structured
result.

TEXT:
{section_text}"""

# --- S0: naive baseline ----------------------------------------------------
# Deliberately carries NONE of the invariant rules. This is the control that
# makes S1's rules measurable: without it, every arm shares the intervention
# and the experiment can only compare scaffolding on top of an already-solved
# problem.

NAIVE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured financial data from SEC 10-K filings.",
        ),
        (
            "human",
            "Extract the target fields from the 10-K text below.\n\n"
            "TEXT:\n{section_text}",
        ),
    ]
)

# --- S1: zero-shot baseline -----------------------------------------------

ZERO_SHOT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured financial data from SEC 10-K filings "
            "accurately and conservatively.\n\n" + _COMMON_RULES,
        ),
        ("human", _TASK),
    ]
)


# --- S2: few-shot ----------------------------------------------------------
# Worked examples use a small INVENTED table (not a real filing) to demonstrate
# the unit multiply, first-column selection, and comma/symbol stripping.

_FEW_SHOT_EXAMPLES = """\
Two worked examples of reading a flattened statement correctly.

Example A — statements are "in millions", two years shown:
    Net revenues              394,328     365,817
    Net income             $   99,803   $  94,680
    Total assets              352,755     351,002
Correct extraction:
    total_revenue = 394328000000   (first column, x1,000,000)
    net_income    = 99803000000    ($ and comma stripped, x1,000,000)
    total_assets  = 352755000000

Example B — statements are "in thousands", operating cash flow is negative:
    Net cash provided by (used in) operating activities   (6,327)   4,512
Correct extraction:
    operating_cash_flow = -6327000   (first column, negative, x1,000)"""

FEW_SHOT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured financial data from SEC 10-K filings "
            "accurately and conservatively.\n\n"
            + _COMMON_RULES
            + "\n\n"
            + _FEW_SHOT_EXAMPLES,
        ),
        ("human", _TASK),
    ]
)


# --- S3: evidence-first ----------------------------------------------------
# Bind FilingExtractionWithEvidence to this one (not FilingExtraction).

_EVIDENCE_INSTRUCTION = """\
For every monetary field, GROUND your answer:
  - Find the exact line in the text that contains the figure.
  - Copy that line VERBATIM into the field's `source_line`.
  - Then apply the unit and first-column rules to produce `value`.
If a field is absent, set both `value` and `source_line` to null. Do not write a
`source_line` that is not present word-for-word in the text above."""

EVIDENCE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured financial data from SEC 10-K filings "
            "accurately and conservatively.\n\n"
            + _COMMON_RULES
            + "\n\n"
            + _EVIDENCE_INSTRUCTION,
        ),
        ("human", _TASK),
    ]
)


# --- S4: context-aware -----------------------------------------------------
# `{section_label}` is filled by the extractor from provenance: "Item 8
# (Financial Statements)" for a clean slice, or "the full 10-K filing" for a
# fallback document — so the orientation is always truthful.

_ORIENTATION = """\
You are given {section_label} from a 10-K. Orientation for where the figures sit:
  - The income statement (often "Consolidated Statements of Operations") shows
    total revenue as the top line and net income near the bottom.
  - The balance sheet ("Consolidated Balance Sheets") shows total assets and
    total liabilities.
  - The cash flow statement shows "Net cash provided by operating activities".
  - The scale header ("in millions"/"in thousands") usually appears ABOVE a
    statement, possibly many lines from the figure — apply it regardless of
    distance.
  - The auditor's report ("Report of Independent Registered Public Accounting
    Firm") names the audit firm."""

CONTEXT_AWARE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured financial data from SEC 10-K filings "
            "accurately and conservatively.\n\n"
            + _COMMON_RULES
            + "\n\n"
            + _ORIENTATION,
        ),
        ("human", _TASK),
    ]
)


# --- Registry: extractor selects by strategy id ---------------------------

STRATEGIES: dict[str, ChatPromptTemplate] = {
    "S0": NAIVE,
    "S1": ZERO_SHOT,
    "S2": FEW_SHOT,
    "S3": EVIDENCE,
    "S4": CONTEXT_AWARE,
}

# Which strategies bind the evidence schema vs the plain schema. The extractor
# reads this to choose FilingExtractionWithEvidence for S3 and FilingExtraction
# for the rest.
EVIDENCE_STRATEGIES: frozenset[str] = frozenset({"S3"})