"""Usage limits for /query: a per-client rate limit and a daily budget.

Stop one client, or everyone together, from spending the OpenAI credit that
keeps the public demo alive. Two limits, both OFF unless set:

  RATE_LIMIT_PER_MINUTE  per client IP, sliding 60-second window
  DAILY_QUERY_BUDGET     all clients together, per UTC day

A blocked request gets 429 with Retry-After: rate_limited (wait a moment) or
daily_budget_exhausted (resets 00:00 UTC).

WHY COUNT REQUESTS, NOT TOKENS OR DOLLARS. A request's cost has a ceiling:
the planner allows at most three tasks, questions are capped at
QUERY_MAX_CHARS, and the most expensive kind (a corpus-wide survey, ~20 LLM
calls) costs about $0.004. So a budget of N requests is also a dollar cap of
about N x $0.004. If that ceiling ever goes — no task cap, long documents as
input — count tokens instead, as OpenAI's own limits do (requests AND tokens).

ONLY REQUESTS THAT WILL SPEND MONEY ARE COUNTED. The route calls check() after
body validation and after the readiness check, immediately before the planner:
a 422 or a not-ready 503 uses up nobody's allowance. A request refused by one
limit is not counted against the other.

IN-MEMORY, ONE WORKER. Counters live in this process; the API runs one worker
because Chroma's local store is single-process (docs/api_contract.md §1). They
reset on restart; for a demo that is acceptable and stated. Several workers
would each keep their own counts and multiply the limits.

CLIENT IDENTITY is request.client.host. Behind a proxy (such as a Hugging
Face Space) that is the PROXY's address unless uvicorn runs with
--proxy-headers and --forwarded-allow-ips, which make it the real client's.
X-Forwarded-For is deliberately NOT parsed here: any client can send that
header, so trusting it in code would let one client pose as many. The daily
budget does not depend on client identity and works either way.

Thread-safety: check() never awaits, so on one event loop it runs to the end
without interleaving; no lock is needed.
"""

import math
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

from src.api.errors import DailyBudgetError, RateLimitedError
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

WINDOW_SECONDS = 60.0     # "per minute": the window the setting's name promises


def _utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


class UsageLimiter:
    """Per-client and daily limits on /query, both optional.

    Attributes:
        per_minute: Requests allowed per client per 60 s, or None for no limit.
        daily_budget: Requests allowed per UTC day across all clients, or None.
    """

    def __init__(
        self,
        per_minute: int | None = None,
        daily_budget: int | None = None,
        monotonic: Callable[[], float] | None = None,
        utc_now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Create a limiter with empty counters.

        Args:
            per_minute: RATE_LIMIT_PER_MINUTE.
            daily_budget: DAILY_QUERY_BUDGET.
            monotonic: Clock for the sliding window; replaceable in tests.
            utc_now: Clock for the UTC day; replaceable in tests.
        """
        self.per_minute = per_minute
        self.daily_budget = daily_budget
        self._monotonic = monotonic or time.monotonic
        self._utc_now = utc_now
        self._recent: dict[str, deque[float]] = {}
        self._last_sweep = self._monotonic()
        self._day: date = self._utc_now().date()
        self._used_today = 0

    @classmethod
    def from_settings(cls) -> "UsageLimiter":
        """Build the limiter from the current settings.

        Returns:
            A limiter; with both settings None it never blocks.
        """
        return cls(per_minute=settings.RATE_LIMIT_PER_MINUTE,
                   daily_budget=settings.DAILY_QUERY_BUDGET)

    @property
    def enabled(self) -> bool:
        """Whether any limit is set."""
        return self.per_minute is not None or self.daily_budget is not None

    @property
    def used_today(self) -> int:
        """Requests counted against today's budget so far (UTC day)."""
        self._roll_day()
        return self._used_today

    def check(self, client: str) -> None:
        """Count one request from a client, or refuse it.

        Per-client limit first, then the daily budget. A refused request is
        counted against neither.

        Args:
            client: The client's identity (its IP address).

        Raises:
            RateLimitedError: per-client limit; DailyBudgetError: daily budget.
                Both carry retry_after_s, the seconds until the
                request would be accepted.
        """
        if not self.enabled:
            return
        now = self._monotonic()
        self._roll_day()
        self._sweep(now)

        recent = self._recent.setdefault(client, deque())
        if self.per_minute is not None:
            while recent and now - recent[0] >= WINDOW_SECONDS:
                recent.popleft()
            if len(recent) >= self.per_minute:
                wait = WINDOW_SECONDS - (now - recent[0])
                logger.warning("rate_limited", limit="per_client", client=client,
                               per_minute=self.per_minute)
                raise RateLimitedError(
                    f"client {client} over {self.per_minute}/min",
                    retry_after_s=max(1, math.ceil(wait)),
                )

        if self.daily_budget is not None and self._used_today >= self.daily_budget:
            logger.warning("rate_limited", limit="daily_budget", client=client,
                           daily_budget=self.daily_budget)
            raise DailyBudgetError(
                f"daily budget of {self.daily_budget} used",
                retry_after_s=self._seconds_to_utc_midnight(),
            )

        recent.append(now)
        self._used_today += 1

    def _roll_day(self) -> None:
        """Reset the daily count when the UTC date has changed."""
        today = self._utc_now().date()
        if today != self._day:
            logger.info("daily_budget_reset", previous_day=str(self._day),
                        used=self._used_today, budget=self.daily_budget)
            self._day, self._used_today = today, 0

    def _seconds_to_utc_midnight(self) -> int:
        """Whole seconds until the budget resets, at least 1."""
        now = self._utc_now()
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC)
        return max(1, math.ceil((midnight - now).total_seconds()))

    def _sweep(self, now: float) -> None:
        """Forget clients with no request inside the window, at most once a window.

        Without this, every address that ever called would stay in memory
        forever; a scan of many addresses could grow it without bound.

        Args:
            now: The current monotonic time.
        """
        if now - self._last_sweep < WINDOW_SECONDS:
            return
        stale = [c for c, times in self._recent.items()
                 if not times or now - times[-1] >= WINDOW_SECONDS]
        for client in stale:
            del self._recent[client]
        self._last_sweep = now
