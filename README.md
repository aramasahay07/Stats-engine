# AI Data Lab (Stats & Transforms)

DuckDB-backed statistical and transform engine that stores datasets in Supabase storage/registry and exposes both legacy session endpoints and unified `/datasets/*` routes. All computations run against Parquet data tied to a canonical `dataset_id` while maintaining backward-compatible session aliases.

## Features
- ~140 statistical analyses (descriptives, t-tests, ANOVA, regression) executed in DuckDB/NumPy.
- ~60 data transforms plus table operations (group/pivot/merge/filter) via `transform_service`.
- Dataset upload/profiling pipeline that persists raw + parquet files to Supabase and registers metadata.
- Knowledge base and agent routes remain available; legacy session routes bridge to dataset-backed parquet.
- Health checks and smoke-test script for quick verification.

## Requirements
- Python 3.10+
- Supabase project with:
  - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_SERVICE_KEY`), and `SUPABASE_ANON_KEY` (for auth-enabled flows)
  - Storage bucket (default: `datasets`), table for dataset registry (default: `datasets`)
- Optional: `AUTH_DISABLED=true` to bypass auth during local development.

See `DEPLOY.md` for the full Railway variable list and Supabase table/bucket expectations.

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Required environment
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="..."
export SUPABASE_ANON_KEY="..."           # optional if AUTH_DISABLED=true
export SUPABASE_STORAGE_BUCKET="datasets" # optional override
export DATASETS_TABLE="datasets"          # optional override

uvicorn main:app --host 0.0.0.0 --port 8000
```

### Health check
```
curl http://localhost:8000/health
```

### Automated smoke test
With the API running locally (and env vars exported), execute:
```bash
bash scripts/smoke_test.sh --base-url http://localhost:8000 --email user@example.com --password ****
```
Set `AUTH_DISABLED=true` to skip token acquisition. See `TESTING.md` for endpoint-specific curl examples.

## Railway deployment
- Provision a Python service and supply the env vars listed in `DEPLOY.md` (including Supabase keys and `PORT`).
- The provided `render.yaml`/`Dockerfile` bind to `$PORT` for Railway compatibility.
- Use `scripts/smoke_test.sh --base-url https://<railway-host>` after deploy to validate endpoints.

## Repository layout
- `main.py` – FastAPI app with legacy session routes, transforms, and agents bridged to dataset-backed parquet.
- `app/routers` – DuckDB dataset/query/stats routers.
- `app/services` – Dataset creation/persistence, storage/registry clients, DuckDB manager, and session bridge.
- `scripts/smoke_test.sh` – End-to-end endpoint exercise script.
- `DEPLOY.md`, `TESTING.md`, `ARCHITECTURE.md` – Deployment, testing, and flow documentation.

