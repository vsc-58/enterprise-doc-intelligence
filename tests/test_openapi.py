"""What Swagger shows at /docs: descriptions written for API readers.

The API reuses two planner enums whose docstrings are planner prompt text,
frozen at the measured prompt version. The OpenAPI schema replaces their
descriptions; these tests check the replacement reaches /openapi.json without
touching the classes the planner model sees.
"""

import json
import re

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.models import OPENAPI_SCHEMA_DESCRIPTIONS
from src.query.schemas import Intent, UnsupportedReason

PRIVATE_REFERENCE = re.compile(r"\bD\d{1,2}\b")


def _schema() -> dict:
    """Fetch /openapi.json from a fresh app, as a browser would."""
    return TestClient(create_app()).get("/openapi.json").json()


def test_openapi_shows_no_private_decision_references() -> None:
    assert PRIVATE_REFERENCE.findall(json.dumps(_schema())) == []


def test_planner_enums_get_the_api_descriptions() -> None:
    components = _schema()["components"]["schemas"]
    for name, text in OPENAPI_SCHEMA_DESCRIPTIONS.items():
        assert components[name]["description"] == text


def test_planner_enum_docstrings_are_untouched() -> None:
    """The replacement is in the schema only: the planner still sees its own text."""
    assert "D25" in (Intent.__doc__ or "")
    assert "D32" in (UnsupportedReason.__doc__ or "")


def test_health_summary_names_every_check() -> None:
    summary = _schema()["paths"]["/health"]["get"]["summary"]
    for check in ("database", "vector store", "corpus index", "figure guard"):
        assert check in summary
