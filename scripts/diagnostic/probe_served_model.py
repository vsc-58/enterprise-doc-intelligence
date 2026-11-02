"""Show which dated model OpenAI actually serves behind each configured model name.

Diagnostic (print() allowed). A model name such as "gpt-4o-mini" is an alias
that OpenAI can move to a newer snapshot; the response reports the snapshot
that answered. This prints, for the query model, the extraction model and the
evaluation judge, the configured name and the served name.

Expected today: query and judge are fixed snapshots (served == configured);
extraction is the alias on purpose (src/utils/config.py says why).

Three one-word calls (a fraction of a cent). No store access.

Usage:
    python -m scripts.diagnostic.probe_served_model
"""

from __future__ import annotations

import sys

from openai import OpenAIError

from src.eval.served_model import served_model
from src.utils.config import settings


def main() -> int:
    """Probe each configured chat model; print configured vs served.

    Returns:
        Process exit code: 0 if every call succeeded, 1 otherwise.
    """
    ok = True
    roles = (
        ("query", settings.QUERY_MODEL),
        ("extraction", settings.EXTRACTION_MODEL),
        ("judge", settings.JUDGE_MODEL),
    )
    for role, configured in roles:
        try:
            served = served_model(configured)
        except OpenAIError as exc:
            print(f"{role:10s} configured={configured!r}  FAILED: {type(exc).__name__}: {exc}")
            ok = False
            continue
        pinned = "fixed snapshot" if served == configured else "alias"
        print(f"{role:10s} configured={configured!r}  served={served!r}  ({pinned})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
