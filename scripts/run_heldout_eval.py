"""
scripts/run_heldout_eval.py
Score the winning strategy on the held-out half of the corpus.

The eval-10 is the set S3 was SELECTED on, so its 1.00 is in-sample. The other
ten documents have XBRL ground truth and were never used for selection, so
scoring them is a free out-of-sample measurement — the advantage of an automatic
answer key over hand labelling.

REPORT-ONLY BY PRE-COMMITMENT. Whatever this returns, the prompt is not revised
in response to it. A prompt changed to fix a held-out number turns the held-out
set into a second dev set and destroys the generalisation claim; recovering one
would need freshly acquired documents. If a revision is warranted, it is a new
experiment against a new held-out set, and this result stands as the record of
the unrevised strategy.

Results are written to a SEPARATE file from the bake-off. data/eval/results.json
is the evidence for the S3 selection decision and is left exactly as it was when
that decision was made.

Reads cached responses only — no API cost.

Run: python -m scripts.run_heldout_eval [--strategy S3]

Dependencies: src.eval.evaluate, src.eval.eval_set, src.eval.grounding,
src.extraction.extractor, src.storage.metadata_store, src.utils.*
"""

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from src.eval.eval_set import is_eval_document
from src.eval.evaluate import evaluate_strategy
from src.eval.grounding import check_evidence_consistency
from src.extraction.extractor import ExtractionResult, extract_with_strategy
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _held_out_documents() -> list[tuple[str, int, str]]:
    """
    Return (cik, filing_year, ticker) for every document outside the eval set.

    The corpus comes from SQLite and the split comes from src.eval.eval_set, so
    "held out" means precisely "acquired but not used to select the strategy" —
    a document added to the corpus later is held out automatically.

    Returns:
        Held-out documents ordered by ticker.
    """
    with get_session() as session:
        rows = session.execute(
            select(Document.cik, Document.filing_year, Document.ticker).order_by(
                Document.ticker
            )
        ).all()
    return [
        (cik, year, ticker)
        for cik, year, ticker in rows
        if not is_eval_document(ticker, year)
    ]


def _consistency_rate(results: list[ExtractionResult]) -> tuple[float | None, dict]:
    """
    Compute the value-in-cited-line rate that evaluate_strategy does not cover.

    evaluate_strategy is the Phase 2 artifact and runs check_grounding only, so
    it reports fabricated quotes but not miscitation. Rather than edit it — which
    would make the bake-off and this run incomparable — the second verdict is
    computed here and reported alongside.

    Args:
        results: the scored ExtractionResults.

    Returns:
        (rate over checkable fields, per-document field verdicts). Rate is None
        when nothing was checkable.
    """
    per_document: dict[str, dict[str, bool | None]] = {}
    checks: list[bool] = []
    for result in results:
        verdicts = check_evidence_consistency(result.evidence)
        if not verdicts:
            continue
        per_document[f"{result.cik}_{result.year}"] = verdicts
        checks.extend(v for v in verdicts.values() if v is not None)
    rate = sum(checks) / len(checks) if checks else None
    return rate, per_document


def main() -> None:
    """Score the held-out documents from cache and write the results file."""
    parser = argparse.ArgumentParser(
        description="Score the winning strategy out of sample."
    )
    parser.add_argument("--strategy", type=str, default="S3")
    args = parser.parse_args()

    held_out = _held_out_documents()
    logger.info(
        "heldout_eval_start",
        documents=len(held_out),
        strategy=args.strategy,
        tickers=[t for _, _, t in held_out],
    )

    results: list[ExtractionResult] = []
    missing: list[str] = []
    for cik, year, ticker in held_out:
        try:
            result = extract_with_strategy(args.strategy, cik, year)
            if not result.from_cache:
                # A miss means the prompt, model or sliced input changed since
                # extraction. Scoring a fresh call against records written from a
                # different one would compare two different things.
                logger.warning(
                    "heldout_cache_miss",
                    ticker=ticker,
                    year=year,
                    note="response differs from the one written to extracted_records",
                )
            results.append(result)
        except Exception as exc:  # per-item: one failure never stops the batch
            missing.append(f"{ticker}_{year}")
            logger.error(
                "heldout_result_unavailable",
                ticker=ticker,
                year=year,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    if not results:
        logger.error("heldout_eval_no_results")
        return

    scored = evaluate_strategy(args.strategy, results)
    consistency_rate, consistency_detail = _consistency_rate(results)

    payload = {
        "split": "held_out",
        "note": (
            "Documents never used to select the strategy. Report-only: the "
            "prompt is not revised in response to these numbers."
        ),
        "documents": [f"{t}_{y}" for _, y, t in held_out],
        "unavailable": missing or None,
        "evidence_consistency_rate": consistency_rate,
        "evidence_consistency_per_document": consistency_detail,
        **scored,
    }

    out_path = Path(settings.EVAL_DATA_PATH) / "held_out_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(
        "heldout_eval_complete",
        documents=len(results),
        grounding_rate=scored["grounding_rate"],
        evidence_consistency_rate=consistency_rate,
        validation_failures=scored["validation_failures"],
        path=str(out_path),
    )
    for field, metrics in scored["fields"].items():
        logger.info("heldout_field", field=field, **metrics)


if __name__ == "__main__":
    main()