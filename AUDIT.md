# Repository Audit (current state)

## Overview
The codebase serves FastAPI endpoints for statistical analysis, transforms, and knowledge-base access. Two parallel data paths exist: legacy session endpoints compute directly on pandas/NumPy in `main.py`, while newer dataset endpoints depend on DuckDB + Parquet via services under `app/services`. Supabase is required for both dataset metadata/Parquet storage and knowledge-base reads.

## Entry point and routing
- **`main.py`**: Declares the FastAPI app, CORS, and includes the knowledge-base router and agent router. Defines `/health` twice (preflight check at ~L90 and legacy liveness at ~L1075) plus `/healthz` and root metadata. Legacy session endpoints (upload/analysis/query/transforms/etc.) remain here and operate on pandas DataFrames kept in Parquet via the session→dataset bridge. New dataset/DuckDB routers are conditionally included based on import success (~L39-L53). Startup logging/preflight is implemented via `require_env_vars` and `build_health_status` but duplicate health definitions remain.
- **Routers under `app/routers/`**: Dataset-facing endpoints for creation (`datasets.py`), queries (`query.py`), and stats (`stats.py`) all run through DuckDB and the shared Parquet loader. Each expects `user_id` as a query parameter and returns 422/404 on missing datasets.
- **Knowledge base router `knowledge/routers/kb.py`**: Mounted at `/kb`, provides topic/concept listing, concept search, and enrichment using Supabase REST accessors.
- **Agent router `ai_agent/router.py`**: Mounted at `/agents`, relies on legacy session DataFrames via `_get_df` from `main.py`.

## Data layers and parallel systems
- **DuckDB dataset path**: `app/services/dataset_creator.py` builds Parquet files, uploads raw/parquet to Supabase Storage, profiles via `DuckDBManager`, and registers rows in Supabase Postgres through `DatasetRegistry`. Dataset queries/stats read Parquet with `ensure_parquet_local` and execute DuckDB SQL in `DuckDBManager`.
- **Legacy session path**: Session IDs map to datasets via the in-memory `SessionDatasetBridge` (`app/services/session_dataset_bridge.py`). Session endpoints in `main.py` load Parquet with `_get_df` (pandas), perform analyses using pandas/NumPy/Scipy/statsmodels/sklearn, and persist updates back to Parquet via `persist_dataframe`. This path bypasses DuckDB for most computations, creating a duplicate compute system.
- **Transform engine**: Legacy transform endpoints in `main.py` depend on optional `transformers` package and `transform_service.py`; they operate on pandas DataFrames.
- **Knowledge base**: Uses Supabase REST endpoints via `knowledge/client.py` and `knowledge/queries.py`, requiring `stat_*` tables; no DuckDB involvement.

## Supabase dependencies
- **Environment variables**: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_SERVICE_KEY`) required by registry/storage clients; `SUPABASE_STORAGE_BUCKET` defaults to `datasets`; `DATASETS_TABLE` defaults to `datasets`; `SUPABASE_ANON_KEY` recommended for KB reads.
- **Postgres tables (expected columns)**:
  - `datasets` (default): `dataset_id`, `user_id`, `project_id`, `file_name`, `raw_file_ref`, `parquet_ref`, `n_rows`, `n_cols`, `schema_json`, `profile_json` (writes happen in `DatasetRegistry`).
  - Knowledge base tables accessed via REST: `stat_topics`, `stat_concepts`, `stat_concept_aliases`, `stat_concept_links`, `stat_formulas`, `stat_examples`, `stat_prerequisites`, `stat_resources` with columns implied by REST selects in `knowledge/queries.py`.
- **Storage buckets**: `datasets` bucket (default) with paths `{user_id}/datasets/{dataset_id}/raw/...` and `.../parquet/data.parquet` (`SupabaseStorageClient`).
- **Auth assumptions**: No runtime auth enforcement in FastAPI; relies on caller providing `user_id`. Supabase keys are used server-side (service role) for registry/storage/KB; optional `SUPABASE_ANON_KEY` for read-only KB access if RLS allows.

## Execution engines
- **DuckDB**: Used for dataset profiling, querying, and stats in `/datasets/*` endpoints (`DuckDBManager`).
- **Pandas/NumPy/Scipy/Statsmodels/SKLearn**: Used heavily in legacy session endpoints for analysis, regression, QC charts, transforms, etc. This is a parallel compute path not routed through DuckDB/Parquet SQL.

## Endpoint surface (high level)
See `ENDPOINTS.md` for a complete list. Highlights:
- Health: `/health` (preflight + legacy), `/healthz`, `/`.
- Dataset path: `/datasets` (POST create), `/datasets/{dataset_id}` (GET), `/datasets/{dataset_id}/query`, `/datasets/{dataset_id}/stats`.
- Legacy session path: `/upload`, `/rehydrate/dataset/{dataset_id}`, `/analysis/{session_id}`, `/advanced-analysis/{session_id}`, `/query/{session_id}`, `/transform/*`, `/session/*` table ops, control chart/capability/regression endpoints.
- Knowledge base: `/kb/topics`, `/kb/concepts/{slug}`, `/kb/search`, `/kb/enrich`.
- Agents: `/agents/explore|patterns|causality|unified`.

## Reliability risks (current)
- Duplicate `/health` definitions in `main.py` create ambiguity and could mask preflight failures.
- In-memory `SessionDatasetBridge` is not persisted; session mappings are lost on restart, breaking legacy flows after deployment restarts.
- Session endpoints rely on pandas/NumPy computations with no guards for large datasets, risking memory pressure vs DuckDB-backed paths.
- Supabase dependencies are mandatory but not lazily handled; missing env vars throw runtime errors during import (dataset registry/storage) even when endpoints unused.
- Knowledge base and dataset Supabase calls assume tables/buckets exist; errors are returned at runtime without centralized retry/backoff.
- Transform/agent features rely on optional packages (`transformers`, `transform_service`, ML libs) and may raise ImportErrors if missing; health endpoints still report “ok”.
- Dockerfile copies repo root but may miss generated cache data; service binds to `0.0.0.0:$PORT` but relies on external Supabase connectivity.
- No auth enforcement; relying on `user_id` query param allows cross-user access if a dataset_id is guessed.
- Smoke test script assumes running backend and Supabase keys; lacks mocking/fallbacks.

## Capability preservation check
- Statistical analyses (~140 concepts) remain in `knowledge` seed/REST access and legacy stats functions in `main.py` plus DuckDB stats router.
- Transforms (~60) exposed via `transformers.registry` and transform endpoints; dependent on optional package installation.
- Knowledge base endpoints intact via `/kb` router.
- Agent endpoints present and wired to session DataFrames through `_get_df`.
