"""Which model OpenAI actually serves behind a configured name, and a guard on it.

A model name such as "gpt-4o-mini" can be an alias that OpenAI moves to a
newer snapshot; every response reports the snapshot that answered. The prompt
version hashes cover the configured name only, so they cannot see such a move.
The evaluation scripts therefore make one tiny call before they start, record
the served name next to the configured one, and refuse to run when the two
differ: a result file then always names the snapshot that produced it.

Used by the query-path evaluation scripts and by
scripts/diagnostic/probe_served_model.py. Not used by the API: the served
query path makes no extra call per request.

Cost: one call with a one-word reply per check (a fraction of a cent).
"""

from collections.abc import Callable

from langchain_openai import ChatOpenAI
from openai import OpenAIError

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROBE_PROMPT = "Reply with the single word OK."


class ServedModelError(RuntimeError):
    """The served model could not be confirmed to equal the configured one."""


class ServedModelMismatch(ServedModelError):
    """The response reports a different model from the configured name.

    Attributes:
        configured: The model name that was called.
        served: The model name the response reported.
    """

    def __init__(self, configured: str, served: str) -> None:
        """Build the error with a message that says what to change.

        Args:
            configured: The model name that was called.
            served: The model name the response reported.
        """
        super().__init__(
            f"configured model {configured!r} is served as {served!r}: the configured "
            "name is an alias. Set it to the dated snapshot to measure a fixed model."
        )
        self.configured = configured
        self.served = served


def served_model(configured: str) -> str | None:
    """Make one tiny call and return the model name the response reports.

    Args:
        configured: The model name to call.

    Returns:
        The served model name from the response metadata, or None if absent.

    Raises:
        OpenAIError: If the call fails (bad key, unknown model, network).
    """
    llm = ChatOpenAI(
        model=configured,
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    )
    response = llm.invoke(PROBE_PROMPT)
    return response.response_metadata.get("model_name")


def require_served(
    configured: str, probe: Callable[[str], str | None] = served_model
) -> str:
    """Confirm the configured model is served under exactly that name.

    Args:
        configured: The model name from settings.
        probe: Function that calls the model and returns the served name.
            Tests pass a stub; the default makes a real call.

    Returns:
        The served model name, equal to configured.

    Raises:
        ServedModelMismatch: If the response names a different model.
        ServedModelError: If the response names no model, or the call failed.
    """
    try:
        served = probe(configured)
    except OpenAIError as exc:
        raise ServedModelError(
            f"probe call to {configured!r} failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not served:
        raise ServedModelError(f"the response from {configured!r} reported no model name")
    if served != configured:
        raise ServedModelMismatch(configured, served)
    return served


def check_served_or_log(
    role: str, configured: str, probe: Callable[[str], str | None] = served_model
) -> str | None:
    """Run require_served for an evaluation script; log the outcome either way.

    The scripts turn None into exit code 1 before any paid work starts.

    Args:
        role: What the model does in this run ("query", "judge"); for the log.
        configured: The model name from settings.
        probe: Function that calls the model and returns the served name.

    Returns:
        The served model name, or None when the check failed (already logged).
    """
    try:
        served = require_served(configured, probe)
    except ServedModelError as exc:
        logger.error("served_model_check_failed", role=role, configured=configured,
                     error_type=type(exc).__name__, error=str(exc))
        return None
    logger.info("served_model_confirmed", role=role, configured=configured, served=served)
    return served
