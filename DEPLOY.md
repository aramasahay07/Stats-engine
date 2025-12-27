# Deployment Guide (Railway + Supabase)

## Required environment variables
Set these variables in Railway (or your runtime):

- `SUPABASE_URL` – base URL of your Supabase project (e.g., `https://xxxx.supabase.co`).
- `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_SERVICE_KEY`) – service role key used for datasets registry + storage access.
- `SUPABASE_ANON_KEY` – preferred for read-only KB endpoints; falls back to service role key if omitted.
- `SUPABASE_STORAGE_BUCKET` – storage bucket name for datasets (default: `datasets`).
- `DATASETS_TABLE` – Postgres table name for dataset registry metadata (default: `datasets`).
- `SESSION_DATASET_LINKS_TABLE` – Postgres table for legacy session↔dataset persistence (default: `session_dataset_links`).
- `PORT` – port provided by Railway (the app binds to `$PORT`).

Optional (reported by `/health` as degraded when absent):

- `DATABASE_URL` – direct Postgres connection string (used for `SELECT 1` connectivity checks).

## Supabase setup
- **Storage**: bucket named `datasets` (or the value of `SUPABASE_STORAGE_BUCKET`). Objects are written under `{user_id}/datasets/{dataset_id}/raw/{filename}` and `{user_id}/datasets/{dataset_id}/parquet/data.parquet`.
- **Dataset registry table** (default `datasets`):
  - `dataset_id` (uuid/text, primary key)
  - `user_id` (text)
  - `project_id` (text, nullable)
  - `file_name` (text)
  - `raw_file_ref` (text)
  - `parquet_ref` (text)
  - `n_rows` (integer)
  - `n_cols` (integer)
  - `schema_json` (jsonb)
  - `profile_json` (jsonb)
  - `created_at` (timestamp, default now())
- **Session mapping table** (default `session_dataset_links`):
  - `session_id` (text, primary key)
  - `dataset_id` (uuid/text)
  - `user_id` (text)
  - `project_id` (text, nullable)
  - `metadata` (jsonb, optional)
  - `created_at` (timestamptz default now())
  - `updated_at` (timestamptz default now(); trigger updates on modification)
- **Knowledge base tables** (read endpoints expect these): `stat_topics`, `stat_concepts`, `stat_concept_aliases`, `stat_concept_links`, `stat_formulas`, `stat_examples`, `stat_prerequisites`, `stat_resources`.
- **RLS expectations**: KB endpoints prefer `SUPABASE_ANON_KEY` for read-only SELECT. Dataset registry/storage use the service role key (no RLS assumed).

## Build & run (Railway Docker)
1. Railway auto-builds using the provided `Dockerfile`.
2. Start command: `uvicorn main:app --host 0.0.0.0 --port ${PORT}` (handled by `CMD`).
3. Health check: `GET /health` reports liveness, presence of required environment variables (names only, no values), Supabase DB connectivity (via `DATABASE_URL`), storage bucket reachability, and knowledge base reachability. Missing pieces return `status="degraded"` with an `errors` list while keeping HTTP 200.

## Local development
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Export the environment variables above, then run `uvicorn main:app --reload`.
