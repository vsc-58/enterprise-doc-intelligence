# src/storage/metadata_store.py
# Module: Metadata store (SQLite via SQLAlchemy 2.x)
# Purpose: Defines the Document tracking table (Phase 1), the ExtractedRecord
#          extraction-output table (Phase 3), and the engine/session machinery
#          used across the project.
# Depends on: sqlalchemy>=2.0, src.utils.config, src.utils.logger
#
# BOUNDARY RULE: no XBRL financial value ever reaches extracted_records. Every
# numeric in that table is produced by the LLM extraction pipeline. XBRL lives
# only in data/eval/ground_truth.json, on the evaluation side, where it scores
# this table rather than populating it.

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The five numeric fields that carry per-field evidence and grounding verdicts.
# Single source of truth for the pipeline, the export layer, and any code that
# iterates the evidence columns — so adding a sixth field is one edit, not five.
EVIDENCE_FIELDS: tuple[str, ...] = (
    "total_revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "operating_cash_flow",
)


class ExtractionStatus(str, Enum):
    """
    Outcome of one extraction attempt.

    SUCCESS: the model returned a schema-valid extraction. The record holds its
        fields; per-field grounding flags qualify individual figures. A record is
        SUCCESS even when one field is ungrounded — discarding four verified
        figures over one bad citation loses more than it protects.
    EXTRACTION_FAILED: the call completed and was billed, but produced no usable
        extraction (schema validation failed, or the output was unusable). The
        row is written anyway, because a paid response must be persisted before
        anything else happens to it (Phase 2 bug 3).
    TECHNICAL_FAILED: the call never completed — transport error, timeout, rate
        limit, context-length rejection. Not a statement about the model.

    Only SUCCESS sets Document.is_extracted, so both failure kinds are picked up
    by a re-run. The two failure kinds are kept apart for the same reason the
    Phase 2 cache kept them apart: a timeout is not a result, a validation
    failure is.

    str-valued so the enum member persists as readable text in SQLite and is
    directly comparable in a raw SQL query from DB Browser.
    """

    SUCCESS = "success"
    EXTRACTION_FAILED = "extraction_failed"
    TECHNICAL_FAILED = "technical_failed"


class Base(DeclarativeBase):
    """Declarative base for all ORM models in the metadata store."""

    pass


class Document(Base):
    """
    Tracking record for one acquired 10-K filing.

    One row per (company, fiscal year). The (cik, filing_year) pair is the
    domain key — a company files exactly one 10-K per fiscal year — and is
    enforced as a unique constraint so duplicate acquisition is rejected at
    the database level regardless of which code path writes.

    The two boolean flags (is_extracted, is_embedded) track two independent
    downstream processes — extraction (Phase 3) and embedding (Phase 4) —
    that both read the acquired text. They are deliberately separate flags
    rather than a single linear status, because a document can be embedded
    but not yet extracted, or vice versa.

    Attributes:
        id: Surrogate primary key.
        company_name: Human-readable company name from the filing.
        ticker: Stock ticker used to request the filing.
        cik: EDGAR Central Index Key, stored as a string to preserve the
             canonical 10-digit zero-padded form (int would drop leading zeros).
        filing_year: Fiscal year the 10-K covers.
        filing_date: EDGAR filing date. Joined into extraction output rather
             than extracted by the LLM (Phase 2 D9).
        accession_number: EDGAR's unique submission ID. Stored for reference
             only — not uniquely constrained, since the access pattern is by
             (cik, filing_year).
        local_path: Path to the saved raw filing text on disk.
        is_extracted: Set True by the Phase 3 extraction pipeline, on SUCCESS only.
        is_embedded: Set True by the Phase 4 embedding pipeline.
        created_at: UTC timestamp when the row was created.
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("cik", "filing_year", name="uq_document_cik_year"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_name: Mapped[str] = mapped_column(String, nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    cik: Mapped[str] = mapped_column(String, nullable=False)
    filing_year: Mapped[int] = mapped_column(Integer, nullable=False)
    filing_date: Mapped[str | None] = mapped_column(String, nullable=True)
    accession_number: Mapped[str | None] = mapped_column(String, nullable=True)
    local_path: Mapped[str | None] = mapped_column(String, nullable=True)
    is_extracted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    is_embedded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return (
            f"Document(id={self.id}, ticker={self.ticker!r}, "
            f"cik={self.cik!r}, filing_year={self.filing_year})"
        )


class ExtractedRecord(Base):
    """
    One extraction attempt against one document.

    ONE ROW PER ATTEMPT, not per document. A re-run after a prompt change appends
    a row rather than overwriting, so a failure and its eventual fix both survive
    for diagnosis. The cost is that every consumer wanting "the current answer"
    must select the latest SUCCESS per document — use latest_successful_records()
    rather than reading the table directly.

    Every value here is produced by the extraction pipeline. XBRL never populates
    this table; it scores it.

    EVIDENCE IS STORED TWICE, deliberately. evidence_json is the verbatim payload
    the model returned — the audit record, which survives a change to the field
    set. The flattened *_source_line columns are the readable projection of the
    same data: a JSON blob is opaque in DB Browser, unsortable in a plain SELECT,
    and unreadable in the /export CSV a reviewer opens. Storage stays flexible;
    the artifact stays legible.

    TWO INDEPENDENT GROUNDING VERDICTS per numeric field, because Phase 2 D12
    established the first is not sufficient:
      - *_evidence_present:    the quoted line occurs in the source text. Catches
        fabricated citations. Read 1.00 across the eval-10.
      - *_evidence_consistent: the stored value occurs in the quoted line. Catches
        misattribution — a real line cited for a figure it does not contain. Found
        4 failures the first check passed, concentrated where the target line does
        not exist (AMZN/WMT/INTC total_liabilities, per Phase 1 D5) plus INTC
        net_income, where the model returned NetIncomeLoss while quoting the
        ProfitLoss line.

    Both flags are bool | None. None means NOT CHECKABLE — the field came back
    null, so there is no value and no quote to compare. That is different from
    False (checked, failed). Collapsing them would make the legitimately-null bank
    revenue and untagged liabilities look like grounding failures, which is the
    honest-gap-versus-wrong-value distinction Phase 1 D5 drew.

    Neither flag failing fails the record: it marks one figure unverified so
    consumers filter per field.

    Attributes:
        id: Surrogate primary key.
        document_id: FK to the parent document.
        company_name: LLM-extracted registrant name (a scored field — kept as an
            LLM target rather than joined from Document, because injecting a
            scored field would score edgartools against edgartools, Phase 2 D9).
        fiscal_year_end: LLM-extracted fiscal period end (scored field).
        total_revenue / net_income / total_assets / total_liabilities /
            operating_cash_flow: LLM-extracted figures in ACTUAL DOLLAR UNITS.
        auditor_name: LLM-extracted audit firm from the Item 8 auditor's report.
        <field>_source_line: verbatim line the model cited for that figure.
        <field>_evidence_present: whether the cited line occurs in the source.
        <field>_evidence_consistent: whether the value occurs in the cited line.
        evidence_json: the full per-field evidence payload, verbatim.
        extraction_strategy_used: strategy id (e.g. 'S3').
        prompt_version: first 12 hex chars of the prompt-template hash. Derived,
            never hand-maintained, so it cannot drift from the template it names.
        model_name: the model that produced this record.
        section_source: provenance of the input the model saw — lets a result be
            attributed to input quality vs prompt quality.
        input_tokens / output_tokens: usage for this call.
        extraction_status: one of ExtractionStatus, stored as its string value.
        failure_reason: error detail for non-SUCCESS rows.
        extracted_at: attempt timestamp; orders re-run passes.
    """

    __tablename__ = "extracted_records"
    __table_args__ = (
        # Supports the latest-attempt-per-document lookup that every consumer
        # needs under the one-row-per-attempt model.
        Index(
            "ix_extracted_document_status_time",
            "document_id",
            "extraction_status",
            "extracted_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"), nullable=False, index=True
    )

    # --- LLM-extracted fields --------------------------------------------
    company_name: Mapped[str | None] = mapped_column(String, nullable=True)
    fiscal_year_end: Mapped[str | None] = mapped_column(String, nullable=True)
    total_revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_liabilities: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_cash_flow: Mapped[float | None] = mapped_column(Float, nullable=True)
    auditor_name: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- Per-field cited source lines (flattened from evidence_json) ------
    total_revenue_source_line: Mapped[str | None] = mapped_column(Text, nullable=True)
    net_income_source_line: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_assets_source_line: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_liabilities_source_line: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    operating_cash_flow_source_line: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # --- Grounding check 1: is the cited line in the source text? ---------
    total_revenue_evidence_present: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    net_income_evidence_present: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    total_assets_evidence_present: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    total_liabilities_evidence_present: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    operating_cash_flow_evidence_present: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )

    # --- Grounding check 2: is the value in the cited line? ---------------
    total_revenue_evidence_consistent: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    net_income_evidence_consistent: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    total_assets_evidence_consistent: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    total_liabilities_evidence_consistent: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    operating_cash_flow_evidence_consistent: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )

    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Provenance and run metadata --------------------------------------
    extraction_strategy_used: Mapped[str] = mapped_column(String, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    section_source: Mapped[str | None] = mapped_column(String, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    extraction_status: Mapped[str] = mapped_column(
        String, nullable=False, index=True
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return (
            f"ExtractedRecord(id={self.id}, document_id={self.document_id}, "
            f"status={self.extraction_status!r}, "
            f"strategy={self.extraction_strategy_used!r})"
        )


# Engine: one per process, pointed at the configured SQLite file.
# echo=False — SQL logging is handled by structlog at the call sites, not by
# SQLAlchemy's own logger.
_engine = create_engine(
    f"sqlite:///{settings.SQLITE_DB_PATH}",
    echo=False,
)

# Session factory. expire_on_commit=False keeps attributes accessible on
# returned objects after the session commits and closes, so callers can read
# e.g. doc.id without triggering a lazy reload against a closed session.
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    """
    Create the metadata database and all tables if they do not exist.

    Idempotent: SQLAlchemy issues CREATE TABLE IF NOT EXISTS, so repeated
    calls never drop or modify existing data. Also ensures the parent
    directory of the SQLite file exists, so a fresh clone (where db/ is
    gitignored) works without manual setup.

    IMPORTANT LIMIT: create_all creates missing TABLES; it does not ALTER
    existing ones. Adding extracted_records to a database that already holds
    documents is therefore safe, but a future column added to documents will
    NOT appear in an existing file — that needs an explicit migration or a
    regenerated database.

    Must be called explicitly once at startup by the caller that owns
    process lifecycle — the acquisition script in Phase 1, the extraction
    script in Phase 3, the FastAPI lifespan manager in Phase 6. It is
    deliberately NOT called at import time: importing this module must have
    no side effects (no file writes).

    Returns:
        None

    Raises:
        OSError: If the database directory cannot be created.
    """
    db_path = Path(settings.SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(_engine)
    logger.info("metadata_db_initialised", path=settings.SQLITE_DB_PATH)


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Provide a transactional database session as a context manager.

    Scope is one unit of work — one document, in both the acquisition and the
    extraction pipelines. On clean exit the transaction is committed; on any
    exception it is rolled back and the exception re-raised for the caller to
    handle. The session is always closed. This per-unit transaction boundary is
    what lets a batch pipeline isolate failures: one document's failed write
    rolls back only that document, leaving already-committed rows untouched.

    Yields:
        Session: an open SQLAlchemy 2.x session.

    Raises:
        Exception: re-raises whatever was raised inside the with-block,
                   after rolling back the transaction.
    """
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def document_exists(ticker: str, year: int) -> bool:
    """
    Return True if a Document row already exists for this (ticker, year).

    The acquisition pipeline's pre-flight skip check: a local SQLite read run
    BEFORE any network fetch, so an already-acquired document costs a query, not
    a filing download. Keyed on (ticker, filing_year) — the key the pipeline
    holds at the top of its loop. The hard uniqueness guarantee remains the
    (cik, filing_year) table constraint underneath.

    Args:
        ticker: Stock ticker symbol.
        year: Filing year (period-end convention).

    Returns:
        True if a matching row exists, else False.
    """
    with get_session() as session:
        stmt = select(Document.id).where(
            Document.ticker == ticker,
            Document.filing_year == year,
        )
        return session.execute(stmt).first() is not None


def create_document(
    *,
    company_name: str,
    ticker: str,
    cik: str,
    filing_year: int,
    accession_number: str | None,
    local_path: str,
    filing_date: str | None = None,
) -> int:
    """
    Persist one Document tracking row and return its primary key.

    Called only after BOTH text acquisition and ground-truth capture succeed —
    the committed row is the success signal under the implicit-completeness
    model (no acquisition-status column; the row's existence means the
    document is fully acquired). is_extracted and is_embedded stay at their
    defaults (False); Phases 3 and 4 set them. Keyword-only arguments prevent
    positional mistakes among the several string fields.

    Args:
        company_name: Company name from the filing.
        ticker: Requested ticker.
        cik: Zero-padded 10-digit CIK.
        filing_year: Period-end year.
        accession_number: EDGAR accession, or None.
        local_path: Path to the saved raw-text file.
        filing_date: EDGAR filing date, or None.

    Returns:
        The new row's integer id.

    Raises:
        IntegrityError: If a row for (cik, filing_year) already exists — the
                        unique constraint surfacing a skip-check miss or a
                        concurrent insert, rather than silently duplicating.
    """
    with get_session() as session:
        doc = Document(
            company_name=company_name,
            ticker=ticker,
            cik=cik,
            filing_year=filing_year,
            accession_number=accession_number,
            local_path=local_path,
            filing_date=filing_date,
        )
        session.add(doc)
        session.flush()  # assigns the autoincrement id within the transaction
        return doc.id


def get_pending_extraction_ids() -> list[int]:
    """
    Return the ids of documents that have not yet been successfully extracted.

    Selection is on Document.is_extracted rather than on the presence of an
    extracted_records row, because under one-row-per-attempt a document can have
    several rows and still need extracting — is_extracted is set only by a
    SUCCESS write, so it is the single unambiguous signal.

    Returns:
        Document ids ordered by id, for a stable, resumable run order.
    """
    with get_session() as session:
        stmt = (
            select(Document.id)
            .where(Document.is_extracted.is_(False))
            .order_by(Document.id)
        )
        return list(session.execute(stmt).scalars().all())


def create_extracted_record(
    *,
    document_id: int,
    extraction_status: ExtractionStatus,
    extraction_strategy_used: str,
    prompt_version: str,
    model_name: str,
    fields: dict[str, object] | None = None,
    source_lines: dict[str, str | None] | None = None,
    evidence_present: dict[str, bool | None] | None = None,
    evidence_consistent: dict[str, bool | None] | None = None,
    evidence_json: str | None = None,
    section_source: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    failure_reason: str | None = None,
) -> int:
    """
    Persist one extraction attempt, flipping Document.is_extracted on SUCCESS.

    The record write and the flag update share ONE transaction, so the two can
    never disagree: a crash between them is impossible, and a rolled-back record
    leaves the document pending rather than marked done with nothing stored.
    This is the extraction-side equivalent of the Phase 1 D6 write ordering.

    Failure rows are persisted, not skipped. An EXTRACTION_FAILED response was
    billed; discarding it repeats the Phase 2 bug where ten paid responses were
    lost to an uncaught post-call exception. TECHNICAL_FAILED rows cost nothing
    but record that the attempt happened and why it did not complete.

    Args:
        document_id: FK to the parent document.
        extraction_status: SUCCESS, EXTRACTION_FAILED, or TECHNICAL_FAILED.
        extraction_strategy_used: strategy id (e.g. 'S3').
        prompt_version: first 12 hex chars of the prompt-template hash.
        model_name: the model that produced the response.
        fields: extracted field values keyed by column name (company_name,
            fiscal_year_end, the five numerics, auditor_name). Unknown keys are
            ignored so a schema change upstream cannot silently write nothing;
            omitted keys stay null.
        source_lines: cited line per numeric field, keyed by field name.
        evidence_present: per-field verdict for grounding check 1; None means
            not checkable (the field itself was null).
        evidence_consistent: per-field verdict for grounding check 2; same None
            semantics.
        evidence_json: verbatim per-field evidence payload as a JSON string.
        section_source: provenance of the input the model saw.
        input_tokens: prompt tokens billed.
        output_tokens: completion tokens billed.
        failure_reason: error detail; required in practice for non-SUCCESS rows.

    Returns:
        The new row's integer id.

    Raises:
        ValueError: If document_id does not match an existing document — a
            dangling extraction row would be undiagnosable, so it is rejected
            rather than written.
    """
    allowed_fields = {
        "company_name",
        "fiscal_year_end",
        "auditor_name",
        *EVIDENCE_FIELDS,
    }

    with get_session() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            raise ValueError(f"no document with id {document_id}")

        record = ExtractedRecord(
            document_id=document_id,
            extraction_status=extraction_status.value,
            extraction_strategy_used=extraction_strategy_used,
            prompt_version=prompt_version,
            model_name=model_name,
            evidence_json=evidence_json,
            section_source=section_source,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            failure_reason=failure_reason,
        )

        for name, value in (fields or {}).items():
            if name in allowed_fields:
                setattr(record, name, value)
            else:
                logger.warning(
                    "extracted_field_ignored",
                    document_id=document_id,
                    field=name,
                )

        for field in EVIDENCE_FIELDS:
            setattr(
                record,
                f"{field}_source_line",
                (source_lines or {}).get(field),
            )
            setattr(
                record,
                f"{field}_evidence_present",
                (evidence_present or {}).get(field),
            )
            setattr(
                record,
                f"{field}_evidence_consistent",
                (evidence_consistent or {}).get(field),
            )

        session.add(record)

        if extraction_status is ExtractionStatus.SUCCESS:
            doc.is_extracted = True

        session.flush()
        logger.info(
            "extracted_record_written",
            document_id=document_id,
            record_id=record.id,
            status=extraction_status.value,
            ticker=doc.ticker,
            year=doc.filing_year,
            is_extracted=doc.is_extracted,
        )
        return record.id


def latest_successful_records() -> list[tuple[Document, ExtractedRecord]]:
    """
    Return the most recent SUCCESS attempt for every extracted document.

    THE canonical read for /export, /query, and any spot-check. Under
    one-row-per-attempt, selecting from extracted_records directly would return
    superseded attempts and failure rows alongside current answers — this is the
    function that resolves that, so consumers never re-implement the rule
    differently.

    "Most recent" is by extracted_at, falling back to id for rows written inside
    the same clock tick.

    Returns:
        (Document, ExtractedRecord) pairs, one per document, ordered by ticker.
        Documents with no successful attempt are omitted.
    """
    with get_session() as session:
        stmt = (
            select(Document, ExtractedRecord)
            .join(ExtractedRecord, ExtractedRecord.document_id == Document.id)
            .where(ExtractedRecord.extraction_status == ExtractionStatus.SUCCESS.value)
            .order_by(
                Document.ticker,
                ExtractedRecord.extracted_at.desc(),
                ExtractedRecord.id.desc(),
            )
        )
        latest: dict[int, tuple[Document, ExtractedRecord]] = {}
        for doc, record in session.execute(stmt).all():
            latest.setdefault(doc.id, (doc, record))
        return list(latest.values())