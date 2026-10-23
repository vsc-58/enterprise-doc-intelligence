# src/api/routes/export.py
# Module: GET /export
# Purpose: The extracted figures as a CSV a reviewer can open in a spreadsheet,
#          with every figure's D29 quality flag beside it (docs/api_contract.md §5).
# Depends on: fastapi, pandas, src.api.*, src.query.schemas, src.storage.*,
#             src.utils.text
#
# THE EDGARTOOLS BOUNDARY RULE: this reads extracted_records only, through
# latest_successful_records() (D31). It never reads ground_truth.json or XBRL:
# those are the evaluation answer key, and the values here are the system's own
# output — the thing being evaluated.
#
# Encoding is UTF-8 WITH a byte-order mark ("utf-8-sig"). Source lines are
# quoted filing text and contain characters like ’ and —; Excel on Windows reads
# a UTF-8 file without the mark as the local code page and garbles them.
#
# Not streamed: 20 rows. A plain Response is simpler and lets the whole file be
# built (and fail) before any byte is sent.
#
# Known limit, not handled: a cell beginning with = + - or @ can be read as a
# formula by spreadsheet programs ("CSV injection"). Every value here comes from
# the filings and our own pipeline, not from users, so the risk is low; a
# public upload feature would change that.

import asyncio
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, Response

from src.api.dependencies import database_ready
from src.api.models import ErrorBody
from src.query.schemas import MetricField
from src.storage.metadata_store import Document, ExtractedRecord, latest_successful_records
from src.storage.quality import assess_record
from src.utils.logger import get_logger
from src.utils.text import display_name

logger = get_logger(__name__)

router = APIRouter(tags=["corpus"])

EXPORT_FILENAME = "extracted_records.csv"

IDENTITY_COLUMNS = ["ticker", "company_name", "cik", "fiscal_year", "fiscal_year_end", "auditor_name"]
METRIC_SUFFIXES = ["", "_quality", "_quality_reason", "_quality_note", "_source_line"]
PROVENANCE_COLUMNS = ["extraction_strategy_used", "prompt_version", "model_name", "extracted_at"]
EXPORT_COLUMNS = (
    IDENTITY_COLUMNS
    + [f"{metric.value}{suffix}" for metric in MetricField for suffix in METRIC_SUFFIXES]
    + PROVENANCE_COLUMNS
)


def export_row(doc: Document, record: ExtractedRecord) -> dict[str, Any]:
    """
    One CSV row: identity, then per metric its value, flag, flag note and the
    line the model cited, then provenance.

    The flag comes from assess_record(), the same function the query path
    uses, and the note from QualityFlag.note(), so the CSV, the API and the
    answer text say the same thing about a figure.

    Args:
        doc: The filing.
        record: Its latest successful extraction.

    Returns:
        Column name -> value, with exactly EXPORT_COLUMNS as keys.
    """
    flags = assess_record(record)
    row: dict[str, Any] = {
        "ticker": doc.ticker,
        "company_name": display_name(doc.company_name, doc.ticker),
        "cik": doc.cik,
        "fiscal_year": doc.filing_year,
        "fiscal_year_end": record.fiscal_year_end,
        "auditor_name": record.auditor_name,
    }
    for metric in MetricField:
        name = metric.value
        flag = flags[name]
        row[name] = getattr(record, name)
        row[f"{name}_quality"] = flag.verdict.value
        row[f"{name}_quality_reason"] = flag.reason.value
        row[f"{name}_quality_note"] = flag.note()
        row[f"{name}_source_line"] = getattr(record, f"{name}_source_line")
    row.update(
        extraction_strategy_used=record.extraction_strategy_used,
        prompt_version=record.prompt_version,
        model_name=record.model_name,
        extracted_at=record.extracted_at.isoformat() if record.extracted_at else None,
    )
    return row


def build_csv(pairs: list[tuple[Document, ExtractedRecord]]) -> bytes:
    """
    Render the export file.

    Args:
        pairs: (Document, latest successful ExtractedRecord), ordered by ticker.

    Returns:
        The CSV as UTF-8 bytes with a byte-order mark, header row included.
    """
    frame = pd.DataFrame([export_row(doc, record) for doc, record in pairs], columns=EXPORT_COLUMNS)
    return frame.to_csv(index=False).encode("utf-8-sig")


@router.get(
    "/export",
    response_class=Response,
    responses={
        200: {"content": {"text/csv": {}}, "description": "The CSV file."},
        503: {"model": ErrorBody, "description": "The database is not available."},
    },
    dependencies=[Depends(database_ready)],
    summary="Download the extracted figures as CSV",
    description=(
        "One row per filing: the five figures in actual dollars, each with its "
        "quality flag (verified / unverified / disputed), the reason, the "
        "explanatory note, and the line the extraction cited. Never contains "
        "XBRL ground truth."
    ),
)
async def export_csv() -> Response:
    """
    Build and return the CSV.

    Returns:
        text/csv as an attachment named extracted_records.csv.
    """
    pairs = await asyncio.to_thread(latest_successful_records)
    content = build_csv(pairs)
    logger.info("export_built", rows=len(pairs), bytes=len(content))
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{EXPORT_FILENAME}"'},
    )
