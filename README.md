# Stats Engine v2 (Clean Rewrite)

This backend is a **dataset-first** FastAPI service that uses **DuckDB + Parquet** as the single source of computation.
It is designed to support **1M+ rows / ~1GB** datasets, replayable **PowerQuery-like pipelines**, **multi-user auth via Supabase JWT**, and **real-time job progress via SSE**.

## What this replaces
- v1 session-based endpoints and in-memory dataframe storage are intentionally not present.

## One-truth model
- `dataset_id` identifies the dataset
- `pipeline_hash` identifies an immutable transform pipeline state
- `(dataset_id, pipeline_hash)` identifies the computed "view" over the dataset

## Data persistence
- Raw file: Supabase Storage bucket `datasets`
- Parquet artifact: Supabase Storage (`datasets/{user_id}/{dataset_id}/data.parquet`) or optional disk cache
- Durable cache: Supabase Postgres tables (`datasets`, `analysis_results`, `datalab_sessions`, plus `pipelines`, `jobs`, `artifacts`)
- DuckDB: one DB file **per user** stored on persistent disk (Render) at `${DATA_DIR}/duckdb/{user_id}.duckdb`

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Environment variables
See `.env.example`.

## Deploy (Render)
- Add a persistent disk mounted at `/data`.
- Set `DATA_DIR=/data`.

