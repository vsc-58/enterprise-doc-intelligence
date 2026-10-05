# src/utils/text.py
# Module: Display formatting for names
# Purpose: Turn EDGAR registrant names into something readable in an answer.
# Depends on: standard library only

import re

_STATE_SUFFIX = re.compile(r"/[A-Z]{2}/?\s*$")
_SMALL_WORDS = frozenset({"of", "and", "the", "for", "in", "on", "at", "to"})

# Names that title-casing mangles. Keyed by ticker because the registrant string
# is what is being corrected, so keying on it would be circular.
DISPLAY_OVERRIDES: dict[str, str] = {
    "JPM": "JPMorgan Chase & Co.",
    "CRM": "Salesforce, Inc.",
    "QCOM": "Qualcomm Inc.",
    "BAC": "Bank of America Corp.",
    "TGT": "Target Corp.",
    "GS": "The Goldman Sachs Group, Inc.",
    "XOM": "Exxon Mobil Corp.",
    "MSFT": "Microsoft Corp.",
}


def display_name(registrant: str, ticker: str | None = None) -> str:
    """
    Render a company name for display.

    EDGAR stores registrant names as filed: shouted ("MICROSOFT CORP"), with an
    incorporation-state suffix ("BANK OF AMERICA CORP /DE/"), or already mixed
    case ("Apple Inc."). Mixed-case names are left alone; all-caps names are
    title-cased with small words lowered; a few that title-casing mangles are
    overridden by ticker.

    Args:
        registrant: The name as stored on Document.company_name.
        ticker: The company's ticker, used to look up an override.

    Returns:
        A readable name.
    """
    if ticker and ticker in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[ticker]

    cleaned = _STATE_SUFFIX.sub("", registrant).strip().rstrip(",")
    if any(char.islower() for char in cleaned):
        return cleaned

    words = cleaned.split()
    titled = [
        word.lower() if index and word.lower() in _SMALL_WORDS else word.title()
        for index, word in enumerate(words)
    ]
    return " ".join(titled)