# src/query/intents.py
# Module: Intent policy table (tool selection + section policy)
# Purpose: The application-controlled mapping from intent to tool and to the set
#          of corpus sections a narrative intent searches. Data, not logic.
# Depends on: pydantic v2, src.query.schemas

from pydantic import BaseModel, ConfigDict

from src.query.schemas import Intent, Tool

ITEM_1 = "Item 1"
ITEM_1A = "Item 1A"
ITEM_7 = "Item 7"
ITEM_7A = "Item 7A"

CORPUS_SECTIONS: frozenset[str] = frozenset({ITEM_1, ITEM_1A, ITEM_7, ITEM_7A})


class IntentPolicy(BaseModel):
    """
    How one intent executes.

    The sections are a UNION, searched in one filtered query and ranked by
    distance alone, so the section a chunk came from carries no weight (D38).
    A frozenset rather than a tuple makes that structural: there is no order to
    misread as priority. A miss — the company holds none of the listed sections —
    is always a refusal, which is why there is no `required` flag: under a union
    it would be true for every intent and so carry no information. That is what
    makes an MD&A question about INTC, JPM or XOM refuse rather than answer from
    risk factors.

    No retrieval hint. Hints were the stated reason intents sharing a section
    could still differ (D28); they were unvalidated, the planner's rephrased
    question already carries the same vocabulary, and a generic hint dilutes a
    specific question. Query expansion is the lever to revisit if the 5C
    retrieval hit-rate is poor.

    Attributes:
        tool: Which capability serves it. Read by the executor; the planner never
            sets this, so a plan cannot route itself (D25).
        sections: Corpus sections a narrative intent may draw from. Empty for SQL
            intents, which never touch the vector store.
    """

    model_config = ConfigDict(frozen=True)

    tool: Tool
    sections: frozenset[str] = frozenset()


INTENT_POLICY: dict[Intent, IntentPolicy] = {
    Intent.FINANCIAL_METRIC: IntentPolicy(tool=Tool.SQL),
    Intent.FINANCIAL_RANKING: IntentPolicy(tool=Tool.SQL),
    Intent.FINANCIAL_FILTER: IntentPolicy(tool=Tool.SQL),
    Intent.COMPANY_OVERVIEW: IntentPolicy(
        tool=Tool.RAG, sections=frozenset({ITEM_1}),
    ),
    Intent.STRATEGY: IntentPolicy(
        tool=Tool.RAG, sections=frozenset({ITEM_1, ITEM_7}),
    ),
    Intent.COMPETITION_AND_REGULATION: IntentPolicy(
        tool=Tool.RAG, sections=frozenset({ITEM_1, ITEM_1A}),
    ),
    Intent.RISK_FACTORS: IntentPolicy(
        tool=Tool.RAG, sections=frozenset({ITEM_1A}),
    ),
    Intent.MANAGEMENT_COMMENTARY: IntentPolicy(
        tool=Tool.RAG, sections=frozenset({ITEM_7}),
    ),
    Intent.MARKET_RISK: IntentPolicy(
        tool=Tool.RAG, sections=frozenset({ITEM_7A, ITEM_7}),
    ),
}

# Where a section's text is actually stored, for documents whose labels are
# known to be wrong. Keyed by ticker, then by the canonical section an intent
# asks for; the value REPLACES that section in the filter (include the section
# itself in the value to search both). Data, not code: the retriever applies
# this generically, so a newly found mislabelling is one entry here.
#
# QCOM and META: the Item 1 slice was rejected by the overlap check because
# their business description sits inside the Item 1A slice (D20). An Item 1
# question must therefore search Item 1A for them — and only Item 1A, not the
# whole document, so the intent still controls which text is eligible (D40).
#
# Candidate for automation: the Phase 4 overlap check detects exactly this
# condition at ingestion and could emit these entries (stretch goal).
SECTION_ALIASES: dict[str, dict[str, frozenset[str]]] = {
    "QCOM": {ITEM_1: frozenset({ITEM_1A})},
    "META": {ITEM_1: frozenset({ITEM_1A})},
}


def sections_for(intent: Intent, ticker: str) -> frozenset[str]:
    """
    The sections one company is searched under for one intent.

    Applies SECTION_ALIASES to the intent's canonical sections. Companies with
    no entry are searched under the canonical sections unchanged.

    Args:
        intent: A narrative intent.
        ticker: The company being searched.

    Returns:
        Stored section labels to filter on. Empty for SQL intents.
    """
    aliases = SECTION_ALIASES.get(ticker, {})
    stored: set[str] = set()
    for section in policy_for(intent).sections:
        stored |= aliases.get(section, frozenset({section}))
    return frozenset(stored)

def policy_for(intent: Intent) -> IntentPolicy:
    """
    Look up how an intent executes.

    Args:
        intent: The intent to resolve.

    Returns:
        Its policy.

    Raises:
        KeyError: if an intent has no policy, which means the enum and this table
                  have drifted apart — a programming error, not a runtime one.
    """
    return INTENT_POLICY[intent]