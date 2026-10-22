"""GET / serves the demo page, safely, and it is not part of the API schema."""

import re

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes.demo import DEMO_PAGE_PATH
from src.utils.config import settings


def _page() -> str:
    return TestClient(create_app()).get("/").text


def test_root_serves_the_demo_page() -> None:
    """Step 10: "/" returned {"detail": "Not Found"} — a visitor's first view of the Space."""
    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Ask the 10-K filings</title>" in response.text


def test_page_is_not_in_the_api_schema() -> None:
    spec = TestClient(create_app()).get("/openapi.json").json()
    assert "/" not in spec["paths"]


def test_input_limit_matches_the_api_limit() -> None:
    page = _page()
    assert f'maxlength="{settings.QUERY_MAX_CHARS}"' in page
    assert "__QUERY_MAX_CHARS__" not in page


def test_page_never_parses_server_text_as_html() -> None:
    """Answers contain model-written text: inserting it as HTML would let it run as code."""
    source = DEMO_PAGE_PATH.read_text(encoding="utf-8")
    for api in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert api not in source, api


def test_page_uses_only_the_public_endpoints() -> None:
    """No second way into the pipeline: the page is just another API client."""
    source = DEMO_PAGE_PATH.read_text(encoding="utf-8")
    called = set(re.findall(r'fetch\("([^"]+)"', source))
    assert called == {"/query", "/openapi.json", "/documents"}


def test_developer_links_sit_in_the_footer_not_the_header() -> None:
    """The top of the page serves the question box; API and source links are for engineers."""
    page = _page()
    header = page[page.index("<header>"):page.index("</header>")]
    footer = page[page.index("<footer>"):page.index("</footer>")]
    assert "/docs" not in header and "/export" not in header
    for link in ('href="/docs"', 'href="/export"', 'href="https://github.com/'):
        assert link in footer


def test_corpus_list_sits_next_to_the_question_box() -> None:
    """One place to see which companies and years exist, before the answer area."""
    page = _page()
    assert 'href="#corpus"' not in page
    assert page.index('id="corpus"') < page.index('id="result"')
