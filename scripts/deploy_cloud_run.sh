#!/usr/bin/env bash
# scripts/deploy_cloud_run.sh — build, check, push and deploy the API to Google Cloud Run.
#
# Usage, from the repo root, in Git Bash (Windows) or any bash:
#   bash scripts/deploy_cloud_run.sh build            build both images; prints the tag
#   bash scripts/deploy_cloud_run.sh check  <tag>     prove what the hosted image holds
#   bash scripts/deploy_cloud_run.sh push   <tag>     push it to Artifact Registry
#   bash scripts/deploy_cloud_run.sh deploy <tag>     deploy that image to Cloud Run
#   bash scripts/deploy_cloud_run.sh verify <tag>     compare the live /health with the image
#   bash scripts/deploy_cloud_run.sh all              all five, in order, one new tag
#
# A script and not CD (automatic deploy on merge): the corpus in db/ is
# deliberately not in git, so a CI runner has no corpus to build the image
# from, and deploying from GitHub would need keyless login to Google Cloud.
# The script keeps every flag in one reviewed file, so a deploy is repeatable.
#
# Prerequisites (one-time setup): gcloud signed in, the project and region
# set, the Artifact Registry repository, the runtime service account, the two
# secrets, and `gcloud auth configure-docker <region>-docker.pkg.dev`.
# Docker running, db/ present, `docker compose` stopped.
#
# No secret value appears here: the key and the EDGAR identity are read by
# Cloud Run from Secret Manager when an instance starts.

set -euo pipefail

# Git Bash rewrites arguments that start with "/" into Windows paths, which
# would turn the probe path /health into C:/Program Files/Git/health.
export MSYS_NO_PATHCONV=1

# --- Settings ------------------------------------------------------------------
# Change these here, not on the command line, so every deploy uses the same values.
PROJECT="edi-demo-vsc58"
REGION="asia-south1"
REPOSITORY="edi"                     # Artifact Registry repository (Docker format)
IMAGE_NAME="api"
SERVICE="edi-api"                    # Cloud Run service name; part of the public URL
RUNTIME_SA="edi-run@${PROJECT}.iam.gserviceaccount.com"
LOCAL_IMAGE="enterprise-doc-intelligence:local"

# Secret versions are pinned: a revision always runs with the same key, and a
# key rotation is a new secret version plus a deploy, never a silent change.
OPENAI_KEY_VERSION="3"
EDGAR_IDENTITY_VERSION="2"

# Usage limits, on only here (local runs leave them off). Per client: a person
# asking questions never reaches 10 a minute. Per day, all clients together:
# 50 x $0.004 (the costliest kind of question) = $0.20 a day, so even a
# worst-case abuser needs about a week to use a $1.50 monthly OpenAI limit.
RATE_LIMIT_PER_MINUTE="10"
DAILY_QUERY_BUDGET="50"

IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${IMAGE_NAME}"

# On Windows, gcloud must be called as gcloud.cmd from bash (the bash launcher
# hands Windows Python a Linux-style path). Aliases do not work in scripts.
if command -v gcloud.cmd >/dev/null 2>&1; then GCLOUD="gcloud.cmd"; else GCLOUD="gcloud"; fi

die() { echo "ERROR: $*" >&2; exit 1; }
need_tag() { [[ -n "${1:-}" ]] || die "give the image tag printed by the build step"; }

# --- Steps ---------------------------------------------------------------------

build() {
  [[ -f db/metadata.db && -d db/chroma ]] || die "db/metadata.db and db/chroma/ must exist"
  # Chroma's local store is single-process; a running local container may be
  # writing to db/ while it is copied into the image.
  if [[ -n "$(docker ps -q --filter "ancestor=${LOCAL_IMAGE}")" ]]; then
    die "a local container is running: run 'docker compose down' first"
  fi
  local tag
  tag="$(git rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M)"
  # linux/amd64: the only platform Cloud Run runs, whatever machine builds.
  docker build --platform linux/amd64 -t "${LOCAL_IMAGE}" .
  docker build --platform linux/amd64 -f Dockerfile.cloudrun -t "${IMAGE_REPO}:${tag}" .
  echo "Built ${IMAGE_REPO}:${tag}"
  echo "TAG=${tag}"
}

check() {
  need_tag "${1:-}"
  local image="${IMAGE_REPO}:$1"
  # Runs Python inside the image, as the app user, with no network and no
  # secrets: lists /app, then fails on anything that must not be there.
  docker run --rm --network none --entrypoint python "${image}" -c '
import os, pathlib, sys
app = pathlib.Path("/app")
print("user id:", os.getuid())
for p in sorted(app.iterdir()):
    print(" ", p.name + ("/" if p.is_dir() else ""))
problems = [n for n in (".env", "data", "tests", "scripts", "docs", ".git") if (app / n).exists()]
db = app / "db"
if not (db / "metadata.db").is_file() or not (db / "chroma").is_dir():
    problems.append("db/metadata.db or db/chroma missing")
wrong_owner = [str(p) for p in db.rglob("*") if p.stat().st_uid != 1000]
if wrong_owner:
    problems.append(f"{len(wrong_owner)} corpus files not owned by uid 1000")
if os.getuid() != 1000:
    problems.append("not running as uid 1000")
size = sum(p.stat().st_size for p in db.rglob("*") if p.is_file())
print(f"corpus: {size / 1e6:.1f} MB in /app/db")
print("FORBIDDEN/WRONG:", problems or "none")
sys.exit(1 if problems else 0)
'
  # The build commands themselves must not carry a key (they are stored in
  # the image history, readable by anyone who can pull the image).
  if docker history --no-trunc --format '{{.CreatedBy}}' "${image}" | grep -q "sk-"; then
    die "the image history contains 'sk-'"
  fi
  echo "Image history: no key-like text."
}

push() {
  need_tag "${1:-}"
  docker push "${IMAGE_REPO}:$1"
}

deploy() {
  need_tag "${1:-}"
  # Every flag, and why:
  #   --service-account        the runtime identity: no project roles, may read
  #                            only the two secrets. Without this flag Cloud Run
  #                            would use the default compute account.
  #   --set-secrets            env var = secret:version, read at instance start.
  #   --set-env-vars           usage limits and JSON logs. QUERY_MODEL is NOT set:
  #                            the pinned default in src/utils/config.py is the
  #                            model every reported number was measured on.
  #   --port=8000              where uvicorn listens (Dockerfile.cloudrun).
  #   --memory=1Gi --cpu=1     about 200 MB idle process + a copy of the Chroma
  #                            folder (Chroma rewrites it on open, and the
  #                            container filesystem is in memory) + headroom.
  #   --cpu-boost              extra CPU while the instance starts: shorter cold start.
  #   --cpu-throttling         request-based billing: CPU only while serving
  #                            (and starting); nothing billed while idle.
  #   --execution-environment=gen2   full Linux compatibility for Chroma's
  #                            native code and SQLite file locking.
  #   --concurrency=10         at most 10 requests at once on the one instance.
  #   --timeout=120            longer than the app's own 30 s query limit plus a
  #                            cold start, so the app's JSON 504 answers first.
  #   --min-instances=0        scale to zero when idle: $0, and a cold start
  #                            for the first visitor after a quiet spell.
  #   --max-instances=1        one instance ever: Chroma's store is
  #                            single-process, the usage counters are in
  #                            memory, and it caps the compute bill.
  #   --startup-probe          HTTP GET /health every 5 s, up to 24 times (2 min).
  #                            /health is 200 only when the corpus loaded, so an
  #                            image with a broken corpus never takes traffic:
  #                            the deploy fails and the old revision keeps serving.
  #   --deploy-health-check    start one instance and wait for its probe before
  #                            moving traffic. On by default, but set
  #                            explicitly: a one-off update made with
  #                            --no-deploy-health-check stays on the service.
  #   --allow-unauthenticated  public: anyone with the URL can use the demo.
  local secrets env_vars probe
  secrets="OPENAI_API_KEY=OPENAI_API_KEY:${OPENAI_KEY_VERSION}"
  secrets+=",EDGAR_IDENTITY=EDGAR_IDENTITY:${EDGAR_IDENTITY_VERSION}"
  env_vars="RATE_LIMIT_PER_MINUTE=${RATE_LIMIT_PER_MINUTE}"
  env_vars+=",DAILY_QUERY_BUDGET=${DAILY_QUERY_BUDGET},LOG_FORMAT=json"
  probe="httpGet.path=/health,httpGet.port=8000"
  probe+=",initialDelaySeconds=0,periodSeconds=5,timeoutSeconds=3,failureThreshold=24"
  "${GCLOUD}" run deploy "${SERVICE}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --image="${IMAGE_REPO}:$1" \
    --service-account="${RUNTIME_SA}" \
    --set-secrets="${secrets}" \
    --set-env-vars="${env_vars}" \
    --port=8000 \
    --memory=1Gi \
    --cpu=1 \
    --cpu-boost \
    --cpu-throttling \
    --execution-environment=gen2 \
    --concurrency=10 \
    --timeout=120 \
    --min-instances=0 \
    --max-instances=1 \
    --startup-probe="${probe}" \
    --deploy-health-check \
    --allow-unauthenticated
}

verify() {
  need_tag "${1:-}"
  local url expected live
  url="$("${GCLOUD}" run services describe "${SERVICE}" --project="${PROJECT}" \
    --region="${REGION}" --format='value(status.url)' | tr -d '\r')"
  echo "URL: ${url}"
  # The versions the image computes with its default settings. If the live
  # service reports different ones, a setting (such as QUERY_MODEL) was
  # overridden in production and the reported numbers would not apply.
  local versions_code='
from src.query.planner import planner_version
from src.rag.chain import chain_version
print(planner_version(), chain_version())'
  expected="$(docker run --rm --network none -e OPENAI_API_KEY=unused -e EDGAR_IDENTITY=unused \
    --entrypoint python "${IMAGE_REPO}:$1" -c "${versions_code}" \
    2>/dev/null | tail -n 1 | tr -d '\r')"
  live="$(curl -sS "${url}/health")"
  echo "Live /health: ${live}"
  python - "${expected}" "${live}" <<'EOF'
import json, sys
expected_planner, expected_chain = sys.argv[1].split()
health = json.loads(sys.argv[2])
checks = {
    "status ok": health["status"] == "ok",
    "planner version": health["planner_version"] == expected_planner,
    "chain version": health["chain_version"] == expected_chain,
}
for name, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
print(f"  documents={health['documents']}  chunks={health['chunks']}")
sys.exit(0 if all(checks.values()) else 1)
EOF
}

# --- Main ----------------------------------------------------------------------

step="${1:-}"
case "${step}" in
  build)  build ;;
  check)  check "${2:-}" ;;
  push)   push "${2:-}" ;;
  deploy) deploy "${2:-}" ;;
  verify) verify "${2:-}" ;;
  all)
    tag="$(build | tee /dev/stderr | sed -n 's/^TAG=//p')"
    check "${tag}"
    push "${tag}"
    deploy "${tag}"
    verify "${tag}"
    ;;
  *) die "usage: bash scripts/deploy_cloud_run.sh build|check|push|deploy|verify <tag> | all" ;;
esac
