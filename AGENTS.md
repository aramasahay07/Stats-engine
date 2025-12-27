# Project: Stats Engine Stabilization (v1 → unified)

## End goal
A stable, clean backend statistical engine that:
- has no parallel systems (no session_id vs dataset_id split)
- keeps all existing capabilities (stats concepts, transforms, KB, agents if present)
- has all endpoints working without breaks
- is easy to understand and maintain
- is deployable to Railway using Supabase (Auth, Storage, Postgres)

## Non-negotiables
- Do NOT remove or reduce existing statistical capabilities (~140 concepts).
- Do NOT remove or reduce existing transforms (~60).
- Preserve knowledge base functionality and tables.
- Keep API endpoints working; if an endpoint must change, provide backward-compatible behavior or a clear migration.
- One source of truth: dataset_id is canonical.
- DuckDB is the single compute engine for stats & queries (no parallel pandas-only compute paths).

## Architecture rules
- session_id must be removed OR mapped internally to dataset_id (temporary bridge allowed).
- All computations must run against dataset-backed data (Parquet/DuckDB).
- Keep database/storage access centralized (no scattered ad-hoc Supabase calls).
- No new required environment variables unless documented in README/DEPLOY.md.

## Deliverables
- Clean refactor with minimal diff where possible.
- A DEPLOY.md that lists required env vars + Supabase tables/buckets.
- A TESTING.md with exact curl commands to validate every endpoint.

## Safety
- Never commit secrets (.env, keys).
- Add robust startup checks with clear error messages for missing env vars/tables.
