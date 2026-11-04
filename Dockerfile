# Dockerfile — the API image: FastAPI app served by uvicorn on port 8000.
#
# The corpus (db/) is NOT in this image: docker-compose.yml mounts it (read-write,
# because Chroma writes to its own files when it opens a store). Without it the
# API still starts and /health answers 503 naming what is missing.
# The image holds only what the API imports (src/) and the pinned requirements.
# Dockerfile.cloudrun builds the hosted image from this one, adding the corpus.

FROM python:3.11-slim-bookworm

# Logs go straight to `docker logs`; no .pyc files written into the image.
# TIKTOKEN_CACHE_DIR: where the token encoding baked in below is read from.
# ANONYMIZED_TELEMETRY: Chroma's product telemetry is on by default and would
# call a third-party host; the API should only ever call OpenAI.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes,
# not on every code edit.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Every narrative question is embedded, and the embedding client counts tokens
# with tiktoken's cl100k_base first. tiktoken downloads that file on first use;
# fetching it here means no request ever waits on (or fails at) that download.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Run as a non-root user (UID 1000), here and in the hosted image.
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app src/ src/
USER app

EXPOSE 8000

# /health is 200 only when every data part is ready, and 503 without a corpus,
# which is correctly "unhealthy". urlopen raises on any 4xx/5xx, so the exit
# code is non-zero exactly when /health is not 200. Python, because the slim
# image has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"]

# One worker: Chroma's local store is single-process, and the usage counters live
# in memory (more workers would each keep their own). Exec form, so no shell
# script: uvicorn is PID 1 and receives the stop signal directly.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
