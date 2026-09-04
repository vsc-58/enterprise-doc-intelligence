"""
scripts/run_eval.py
Runs the prompt-strategy bake-off and writes the results artifact.

For each strategy over each evaluation document: extract (reading the on-disk
cache where the content hash matches, calling the API otherwise), score against
XBRL ground truth, aggregate, and write data/eval/results.json. Prints a
per-strategy summary table plus the provenance split.

Per-document errors are caught and logged: one failed document never stops the
run, and because transport failures are not cached, a re-run retries only what
failed.

Run: python -m scripts.run_eval                 (all strategies)
     python -m scripts.run_eval S1 S3           (a subset)
     python -m scripts.run_eval --no-cache S1   (force fresh API calls)
"""

import json
import sys
from pathlib import Path

from sqlalchemy import select

from src.eval.evaluate import NUMERIC_FIELDS, SCORED_FIELDS, evaluate_strategy
from src.extraction.extractor import ExtractionResult, extract_with_strategy
from src.extraction.prompts import STRATEGIES
from src.storage.metadata_store import Document, get_session
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The evaluation set, mirroring scripts/build_sections.py. Kept as (ticker,
# year); CIK is resolved from the Document row so the key is never re-derived.
# EVAL_SET: list[tuple[str, int]] = [
#     ("AAPL", 2021),
#     ("MSFT", 2022),
#     ("GOOGL", 2021),
#     ("AMZN", 2023),
#     ("WMT", 2023),
#     ("JNJ", 2021),
#     ("NFLX", 2023),
#     ("V", 2022),
#     ("ADBE", 2022),
#     ("INTC", 2023),
# ]

# GPT-4o-mini pricing, USD per 1M tokens. Used for the cost axis only.
_INPUT_COST_PER_M = 0.15
_OUTPUT_COST_PER_M = 0.60


def _resolve_ciks(eval_set: list[tuple[str, int]]) -> list[tuple[str, str, int]]:
    """
    Resolve each (ticker, year) to its canonical CIK from the Document rows.

    Args:
        eval_set: the (ticker, year) pairs to evaluate.

    Returns:
        List of (ticker, cik, year); pairs with no Document row are omitted
        and logged.
    """
    resolved: list[tuple[str, str, int]] = []
    with get_session() as session:
        for ticker, year in eval_set:
            doc = session.execute(
                select(Document).where(
                    Document.ticker == ticker, Document.filing_year == year
                )
            ).scalar_one_or_none()
            if doc is None:
                logger.error("eval_doc_no_row", ticker=ticker, year=year)
                continue
            resolved.append((ticker, doc.cik, year))
    return resolved


def run_strategy(
    strategy_id: str,
    documents: list[tuple[str, str, int]],
    use_cache: bool,
) -> list[ExtractionResult]:
    """
    Run one strategy over every evaluation document.

    Args:
        strategy_id: the strategy to run.
        documents: (ticker, cik, year) triples.
        use_cache: read/write the on-disk extraction cache.

    Returns:
        The successful ExtractionResults. Documents whose call raised are
        logged and omitted (not cached, so a re-run retries them).
    """
    results: list[ExtractionResult] = []
    for index, (ticker, cik, year) in enumerate(documents, start=1):
        try:
            result = extract_with_strategy(
                strategy_id, cik, year, use_cache=use_cache
            )
            results.append(result)
        except Exception as exc:  # per-item: one failure never stops the batch
            logger.error(
                "extraction_failed",
                strategy=strategy_id,
                ticker=ticker,
                cik=cik,
                year=year,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            continue
        if index % 5 == 0:
            logger.info(
                "progress", strategy=strategy_id, done=index, total=len(documents)
            )
    return results


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a token spend at the configured model's rates."""
    return (
        input_tokens / 1_000_000 * _INPUT_COST_PER_M
        + output_tokens / 1_000_000 * _OUTPUT_COST_PER_M
    )


def _fmt(value: float | None, places: int = 3) -> str:
    """Format an optional metric for the table; '-' when undefined."""
    return "-" if value is None else f"{value:.{places}f}"


def print_summary(evaluations: dict[str, dict]) -> None:
    """
    Print the per-strategy results table.

    Every cell carries its denominator, because a precision over four documents
    is not the same evidence as one over ten.

    Args:
        evaluations: strategy id -> the evaluate_strategy output.
    """
    print("\n" + "=" * 96)
    print("PROMPT-STRATEGY BAKE-OFF — precision / recall (n scored)")
    print("=" * 96)

    header = f"{'field':<22}" + "".join(f"{sid:>18}" for sid in evaluations)
    print(header)
    print("-" * 96)

    for field in SCORED_FIELDS:
        row = f"{field:<22}"
        for evaluation in evaluations.values():
            metrics = evaluation["fields"][field]
            cell = (
                f"{_fmt(metrics['precision'], 2)}/"
                f"{_fmt(metrics['recall'], 2)} "
                f"(n={metrics['n_scored']})"
            )
            row += f"{cell:>18}"
        print(row)

    print("-" * 96)
    for label, key in (("mean rel error", "mean_rel_error"),):
        row = f"{label:<22}"
        for evaluation in evaluations.values():
            errors = [
                evaluation["fields"][f][key]
                for f in NUMERIC_FIELDS
                if evaluation["fields"][f][key] is not None
            ]
            mean = sum(errors) / len(errors) if errors else None
            row += f"{('-' if mean is None else f'{mean:.2e}'):>18}"
        print(row)

    for label, formatter in (
        ("validation failures", lambda e: str(e["validation_failures"])),
        ("input tokens", lambda e: f"{e['input_tokens']:,}"),
        ("output tokens", lambda e: f"{e['output_tokens']:,}"),
        ("cost (USD)", lambda e: f"${_cost_usd(e['input_tokens'], e['output_tokens']):.4f}"),
        ("grounding rate", lambda e: _fmt(e["grounding_rate"], 2)),
    ):
        row = f"{label:<22}" + "".join(
            f"{formatter(e):>18}" for e in evaluations.values()
        )
        print(row)

    print("=" * 96)

    print("\nPROVENANCE SPLIT — recall on numeric fields by input quality")
    print("-" * 96)
    sources = sorted({s for e in evaluations.values() for s in e["by_provenance"]})
    for source in sources:
        row = f"{source:<22}"
        for evaluation in evaluations.values():
            buckets = evaluation["by_provenance"].get(source)
            if buckets is None:
                row += f"{'-':>18}"
                continue
            correct = sum(buckets[f]["correct"] for f in NUMERIC_FIELDS)
            scored = sum(buckets[f]["n_scored"] for f in NUMERIC_FIELDS)
            cell = f"{correct}/{scored}" if scored else "-"
            row += f"{cell:>18}"
        print(row)
    print("-" * 96 + "\n")


def main() -> None:
    """Run the bake-off over the requested strategies and write results.json."""
    args = [a for a in sys.argv[1:] if a != "--no-cache"]
    use_cache = "--no-cache" not in sys.argv
    strategy_ids = args or list(STRATEGIES)

    unknown = [s for s in strategy_ids if s not in STRATEGIES]
    if unknown:
        logger.error("unknown_strategies", unknown=unknown, known=list(STRATEGIES))
        sys.exit(1)

    documents = _resolve_ciks(EVAL_SET)
    if not documents:
        logger.error("no_eval_documents_resolved")
        sys.exit(1)

    logger.info(
        "bakeoff_start",
        strategies=strategy_ids,
        documents=len(documents),
        use_cache=use_cache,
    )

    evaluations: dict[str, dict] = {}
    for strategy_id in strategy_ids:
        results = run_strategy(strategy_id, documents, use_cache)
        if not results:
            logger.error("strategy_produced_no_results", strategy=strategy_id)
            continue
        evaluations[strategy_id] = evaluate_strategy(strategy_id, results)

    if not evaluations:
        logger.error("bakeoff_produced_no_evaluations")
        sys.exit(1)

    out_path = Path(settings.EVAL_DATA_PATH) / "results.json"
    out_path.write_text(
        json.dumps(evaluations, indent=2, default=str), encoding="utf-8"
    )
    logger.info("results_written", path=str(out_path), strategies=list(evaluations))

    print_summary(evaluations)


if __name__ == "__main__":
    main()