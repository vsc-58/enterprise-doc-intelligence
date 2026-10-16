# src/eval/retrieval_eval.py
# Module: Retrieval evaluation — gold labels and ranking metrics
# Purpose: Load the chunk-level retrieval gold, score one ranking against it
#          (hit, recall and precision at k, first relevant rank, unjudged
#          chunks), and pull the planner's rephrased task questions out of a
#          routing results file so retrieval can be scored on what production
#          actually embeds.
# Depends on: pydantic v2, src.eval.query_eval, src.utils.config, src.utils.logger
#
# Pure scoring: no embedding call and no store access here. The runner supplies
# rankings, which is why every rule below is unit-testable.
#
# Three label states, not two. A pooled chunk is JUDGED: labelled rel, labelled
# border, or (unlabelled) not relevant. A chunk outside the pool is UNJUDGED —
# nobody read it. Scoring counts an unjudged chunk as not relevant, which can
# only understate precision, and reports how many there were so the number is
# never read as final while any remain.
#
# Recall at k is CAPPED: relevant found in the top k divided by min(k, R). A
# question with 17 relevant chunks cannot have more than 5 of them in a top 5,
# and plain recall would call a perfect top 5 "29%". The cap turns the metric
# into "how much of what fits in k is relevant to what was asked for", which is
# what the chain reading k chunks experiences. R is reported beside it.

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.eval.query_eval import canonical_sha256
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GoldError(RuntimeError):
    """Raised when the retrieval gold file is missing or malformed."""


class Relevance(str, Enum):
    """
    A pooled chunk's label.

    REL: a reader could write at least one sentence answering the question from
        this chunk alone, for this company.
    BORDER: on topic but generic, a summary-list fragment, hypothetical, or only
        partly about the question. Counted only in lenient scores.
    """

    REL = "rel"
    BORDER = "border"


class GoldLabel(BaseModel):
    """One labelled chunk with the phrase that justifies the label."""

    model_config = ConfigDict(frozen=True)

    key: str
    relevance: Relevance
    evidence: str


class GoldQuestion(BaseModel):
    """
    Retrieval gold for one question.

    Attributes:
        id: Eval-set question id.
        split: held or dev, copied from the eval set.
        question: The question as asked.
        ticker: The labelled company.
        intent: The labelled intent; fixes the sections searched.
        broad: Most of the section answers it ("what are the main risks"):
            recall is meaningless, so only precision is reported.
        audit: Included to check a refusal (N20), not to measure ranking.
        pool: Every judged chunk key (TICKER|section|index).
        labels: rel and border chunks; pooled chunks without a label are not
            relevant.
        notes: Caveats that change how the numbers read (keyword caps,
            truncated reading).
        recheck: Pooled chunks judged on partial text; re-read in the next
            labelling round.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    split: str
    question: str
    ticker: str
    intent: str
    broad: bool = False
    audit: bool = False
    pool: list[str] = Field(default_factory=list)
    labels: list[GoldLabel] = Field(default_factory=list)
    notes: str | None = None
    recheck: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _labels_are_pooled_and_unique(self) -> "GoldQuestion":
        """Every label names a pooled chunk, once."""
        keys = [label.key for label in self.labels]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.id}: a chunk is labelled twice")
        outside = set(keys) - set(self.pool)
        if outside:
            raise ValueError(f"{self.id}: labelled chunks missing from the pool: {sorted(outside)}")
        return self

    def relevant(self, lenient: bool) -> frozenset[str]:
        """
        The chunks counted as relevant.

        Args:
            lenient: Count border chunks too.

        Returns:
            Chunk keys.
        """
        accepted = {Relevance.REL, Relevance.BORDER} if lenient else {Relevance.REL}
        return frozenset(label.key for label in self.labels if label.relevance in accepted)

    def judged(self) -> frozenset[str]:
        """
        The chunks someone read.

        Returns:
            Pooled chunk keys.
        """
        return frozenset(self.pool)


class RetrievalGold(BaseModel):
    """The whole gold file."""

    model_config = ConfigDict(frozen=True)

    status: str
    labelled_by: str
    relevance_rule: str
    pool_method: str
    questions: list[GoldQuestion]

    @model_validator(mode="after")
    def _unique_ids(self) -> "RetrievalGold":
        """One entry per question."""
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question ids in retrieval gold")
        return self


def load_gold(path: Path | None = None, expected_sha256: str | None = None) -> RetrievalGold:
    """
    Load the retrieval gold, check it is the frozen version, and validate it.

    Args:
        path: Gold file; settings.EVAL_DATA_PATH / settings.RETRIEVAL_GOLD_FILE
            when omitted.
        expected_sha256: Required hash; settings.RETRIEVAL_GOLD_SHA256 when
            omitted. An empty string skips the check (tests, relabelling work).

    Returns:
        The validated gold.

    Raises:
        GoldError: if the file is missing, differs from the frozen hash, is not
            JSON, or fails validation.
    """
    path = path or Path(settings.EVAL_DATA_PATH) / settings.RETRIEVAL_GOLD_FILE
    expected = settings.RETRIEVAL_GOLD_SHA256 if expected_sha256 is None else expected_sha256
    try:
        actual = canonical_sha256(path)
        if expected and actual != expected:
            raise GoldError(
                f"retrieval gold {path} has hash {actual[:12]}, frozen hash is {expected[:12]}: "
                "the labels changed after verification"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        gold = RetrievalGold.model_validate(raw)
    except FileNotFoundError as exc:
        raise GoldError(f"retrieval gold not found: {path}") from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GoldError(f"retrieval gold is malformed: {exc}") from exc
    logger.info("retrieval_gold_loaded", path=str(path), questions=len(gold.questions),
                status=gold.status, sha256=actual[:12], hash_checked=bool(expected))
    return gold


# --- Scoring -----------------------------------------------------------------


class CutoffScore(BaseModel):
    """
    Metrics at one cut-off k.

    Attributes:
        k: The cut-off.
        hit: Any relevant chunk in the top k. None when the question has no
            relevant chunk, so it cannot be hit.
        recall: Relevant in the top k / min(k, R). None when R is zero.
        precision: Relevant in the top k / chunks in the top k (fewer than k
            when the section holds fewer). Unjudged chunks count as not
            relevant, so this is a lower bound while any remain.
        unjudged: Unjudged chunks in the top k.
    """

    model_config = ConfigDict(frozen=True)

    k: int
    hit: bool | None
    recall: float | None
    precision: float
    unjudged: int


class RankingScore(BaseModel):
    """
    One ranking scored against one relevance set.

    Attributes:
        relevant_total: R, the number of relevant chunks in the gold.
        first_relevant_rank: 1-based rank of the nearest relevant chunk; None
            if none appears in the ranking.
        cutoffs: One CutoffScore per k, ascending.
    """

    model_config = ConfigDict(frozen=True)

    relevant_total: int
    first_relevant_rank: int | None
    cutoffs: list[CutoffScore]

    def at(self, k: int) -> CutoffScore:
        """
        The score at cut-off k.

        Args:
            k: A cut-off that was scored.

        Returns:
            Its CutoffScore.

        Raises:
            KeyError: if k was not scored.
        """
        for cutoff in self.cutoffs:
            if cutoff.k == k:
                return cutoff
        raise KeyError(f"cut-off {k} was not scored")


def score_ranking(
    ranking: list[str],
    relevant: frozenset[str],
    judged: frozenset[str],
    ks: list[int],
) -> RankingScore:
    """
    Score one ranking.

    Args:
        ranking: Chunk keys, nearest first.
        relevant: Keys counted as relevant.
        judged: Keys someone read (relevant ones included).
        ks: Cut-offs to score.

    Returns:
        The RankingScore.

    Raises:
        ValueError: if a cut-off is below 1 — a programming error.
    """
    if any(k < 1 for k in ks):
        raise ValueError(f"cut-offs must be at least 1, got {ks}")
    total = len(relevant)
    first = next((i for i, key in enumerate(ranking, start=1) if key in relevant), None)
    cutoffs: list[CutoffScore] = []
    for k in sorted(set(ks)):
        top = ranking[:k]
        found = sum(key in relevant for key in top)
        cutoffs.append(CutoffScore(
            k=k,
            hit=(found > 0) if total else None,
            recall=found / min(k, total) if total else None,
            precision=found / len(top) if top else 0.0,
            unjudged=sum(key not in judged for key in top),
        ))
    return RankingScore(relevant_total=total, first_relevant_rank=first, cutoffs=cutoffs)


def unjudged_in_top(ranking: list[str], judged: frozenset[str], k: int) -> list[str]:
    """
    Chunks in the top k that nobody has read.

    Args:
        ranking: Chunk keys, nearest first.
        judged: Keys someone read.
        k: Cut-off.

    Returns:
        Unjudged keys, in rank order.
    """
    return [key for key in ranking[:k] if key not in judged]


# --- Planner rephrasings -------------------------------------------------------


class PlannerQuery(BaseModel):
    """What production searched with in one run: the planner's intent and question."""

    model_config = ConfigDict(frozen=True)

    intent: str
    question: str


def planner_queries(
    record: dict[str, Any],
    accepted_intents: list[str],
    ticker: str,
) -> list[PlannerQuery | None]:
    """
    The planner's narrative task for this company, per run.

    Production embeds the planner's self-contained task question, not the user's
    words, and filters on the planner's intent, so both are what retrieval must
    be scored with. Any intent the labels accept for this company counts (R24 is
    labelled competition_and_regulation but accepts risk_factors, which searches
    different sections); the first accepted intent in label order wins when a plan
    has several. A run whose plan has no such task (a routing miss, or a failed
    call) yields None and is counted, not dropped, so routing misses stay visible
    instead of silently shrinking n.

    Args:
        record: One question's entry from a routing results file
            ({"runs": [{"observed": ObservedPlan dump or None}, ...]}).
        accepted_intents: Intents the labels accept for this company, strict first.
        ticker: The gold company.

    Returns:
        One entry per run: the planner's query, or None.
    """
    queries: list[PlannerQuery | None] = []
    for run in record.get("runs", []):
        observed = run.get("observed") or {}
        candidates = [
            task for task in observed.get("tasks", [])
            if task.get("company") == ticker and task.get("question")
            and task.get("intent") in accepted_intents
        ]
        candidates.sort(key=lambda task: accepted_intents.index(task["intent"]))
        queries.append(
            PlannerQuery(intent=candidates[0]["intent"], question=candidates[0]["question"])
            if candidates else None
        )
    return queries


def mean_or_none(values: list[float | None]) -> float | None:
    """
    Mean of the non-None values.

    Args:
        values: Numbers, some possibly None (not measurable).

    Returns:
        The mean, or None if nothing was measurable.
    """
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
