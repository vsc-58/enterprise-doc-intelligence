# src/api/routes/demo.py
# Module: GET / — the demo page
# Purpose: Serve one static HTML page where a non-developer can ask a question
#          and read the answer, with figures, quality marks and sources laid
#          out (Step 10 finding: raw JSON in Swagger reads as unfinished, and
#          "/" returned {"detail": "Not Found"} — the first thing a visitor of
#          the Space would see). Swagger stays at /docs for developers.
# Depends on: fastapi, src.utils.config
#
# The page is plain HTML + JavaScript in src/api/static/demo.html, calling the
# public endpoints (/query, /documents, /openapi.json) like any other client:
# no second code path into the query pipeline, no build step, no framework.
# It is read once at import; QUERY_MAX_CHARS is filled in so the input box and
# the API enforce the same limit.

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.utils.config import settings

router = APIRouter()

DEMO_PAGE_PATH = Path(__file__).resolve().parent.parent / "static" / "demo.html"


def load_demo_page() -> str:
    """
    Read the page and fill in the settings it shows.

    Returns:
        The HTML, with __QUERY_MAX_CHARS__ replaced.

    Raises:
        FileNotFoundError: if the page is missing from the package — raised at
            import, so a broken build fails at startup rather than on first visit.
    """
    html = DEMO_PAGE_PATH.read_text(encoding="utf-8")
    return html.replace("__QUERY_MAX_CHARS__", str(settings.QUERY_MAX_CHARS))


_PAGE = load_demo_page()


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def demo_page() -> HTMLResponse:
    """
    The demo page.

    Returns:
        The HTML page. Not listed in the OpenAPI schema: it is not part of the API.
    """
    return HTMLResponse(_PAGE)
