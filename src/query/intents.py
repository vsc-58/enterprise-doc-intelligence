# src/query/intents.py
# Module: Intent policy table (tool selection + retrieval policy)
# Purpose: The application-controlled mapping from intent to tool, corpus
#          section, and retrieval phrasing. Data, not logic.
# Depends on: src.query.schemas

from pydantic import BaseModel, ConfigDict

from src.query.schemas import Intent, Tool

ITEM_1 = "Item 1"
ITEM_1A = "Item 1A"
ITEM_7 = "Item 7"
ITEM_7A = "Item 7A"


class IntentPolicy(BaseModel):
    """
    How one intent executes.

    Attributes:
        tool: Which capability serves it. Read by the executor; the planner never
            sets this, so a plan cannot route itself (D25).
        sections: Ordered section filters to try. Empty means no section filter.
        required: When True, the first section is the only acceptable source and
            a miss is a refusal. This is what makes MD&A questions about the three
            filings with no Item 7 refuse deterministically rather than answering
            from risk factors.
        retrieval_hint: Vocabulary appended to the task's question before
            embedding. The reason nine intents over four sections are not four
            intents wearing different labels: intents sharing a section still
            retrieve differently (D28). Effectiveness is measured in Phase 5C.
    """

    model_config = ConfigDict(frozen=True)

    tool: Tool
    sections: tuple[str, ...] = ()
    required: bool = False
    retrieval_hint: str = ""


INTENT_POLICY: dict[Intent, IntentPolicy] = {
    Intent.FINANCIAL_METRIC: IntentPolicy(tool=Tool.SQL),
    Intent.FINANCIAL_RANKING: IntentPolicy(tool=Tool.SQL),
    Intent.FINANCIAL_FILTER: IntentPolicy(tool=Tool.SQL),
    Intent.COMPANY_DESCRIPTION: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_1,),
        retrieval_hint="principal business activities, operations, what the company does",
    ),
    Intent.SEGMENTS_AND_PRODUCTS: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_1,),
        retrieval_hint="reportable segments, product lines, services offered, markets served",
    ),
    Intent.STRATEGY: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_1, ITEM_7),
        retrieval_hint="strategy, growth priorities, investment focus, long-term objectives",
    ),
    Intent.COMPETITION: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_1, ITEM_1A),
        retrieval_hint="competition, competitors, competitive landscape, market share pressure",
    ),
    Intent.RISK_FACTORS: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_1A,),
        retrieval_hint="risk factors, adverse effects, uncertainties the business faces",
    ),
    Intent.REGULATORY_AND_LEGAL: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_1A, ITEM_1),
        retrieval_hint="regulation, compliance, legal proceedings, government investigations",
    ),
    Intent.MANAGEMENT_COMMENTARY: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_7,),
        required=True,
        retrieval_hint="management discussion, drivers of change, results of operations",
    ),
    Intent.LIQUIDITY_AND_CAPITAL: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_7,),
        required=True,
        retrieval_hint="liquidity, capital resources, cash requirements, capital expenditure",
    ),
    Intent.MARKET_RISK: IntentPolicy(
        tool=Tool.RAG,
        sections=(ITEM_7A, ITEM_7),
        retrieval_hint="interest rate risk, foreign currency exposure, commodity price risk",
    ),
}

# Companies whose Item 1 content sits inside their Item 1A slice (D20). Section
# filtering would exclude the very chunks that answer an Item 1 question, so they
# are filtered by company only, with the reason stated rather than discovered.
SECTION_FILTER_EXEMPT_TICKERS: frozenset[str] = frozenset({"QCOM", "META"})


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