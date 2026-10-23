# src/api/routes/documents.py
# Module: GET /documents
# Purpose: List the filings the corpus holds (docs/api_contract.md §4).
# Depends on: fastapi, src.api.*, src.query.resolve, src.utils.text
#
# Read from the corpus index built at startup: no database call per request.
# The list answers the question every refusal raises — "which companies and
# years DO you hold?" — so a demo viewer can ask answerable questions.

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import documents_index
from src.api.models import DocumentOut, ErrorBody
from src.query.resolve import CorpusIndex
from src.utils.text import display_name

router = APIRouter(tags=["corpus"])


@router.get(
    "/documents",
    response_model=list[DocumentOut],
    responses={503: {"model": ErrorBody, "description": "The corpus index is not loaded."}},
    summary="List the filings in the corpus",
    description=(
        "One 10-K per company. `fiscal_year` is the only year the system can "
        "answer for that company; a question about another year is refused."
    ),
)
async def list_documents(
    index: Annotated[CorpusIndex, Depends(documents_index)],
) -> list[DocumentOut]:
    """
    List every filing in the corpus index, ordered by ticker.

    Args:
        index: The corpus index; provided only when it is loaded.

    Returns:
        One entry per filing, with the display name used in answers.
    """
    return [
        DocumentOut(
            ticker=entry.ticker,
            company_name=display_name(entry.company_name, entry.ticker),
            cik=entry.cik,
            fiscal_year=entry.filing_year,
            fiscal_year_end=entry.fiscal_year_end,
            is_extracted=entry.is_extracted,
            is_embedded=entry.is_embedded,
        )
        for entry in sorted(index.entries, key=lambda e: e.ticker)
    ]
