# Refactor Plan (dataset_id + DuckDB single-path)

## Principles
- Preserve all statistical concepts, transforms, KB, and agent behaviors.
- Maintain backward compatibility for legacy session endpoints by translating to dataset_id until deprecation is communicated.
- Centralize Supabase access and DuckDB compute; remove pandas-only execution where feasible.

## Phase A: Stabilize and observability
1) **Harden startup**: Consolidate `/health` into a single preflight-aware endpoint; keep `/healthz` as alias. Fail fast when `SUPABASE_*` vars or required tables/bucket are missing; add clear logging.
2) **Persist session↔dataset mapping**: Replace in-memory `SessionDatasetBridge` with Supabase table (e.g., `session_dataset_links`) or persisted cache to survive restarts. Add migration note.
3) **Schema validation**: Add checks that `datasets` table has required columns and storage bucket exists; surface actionable errors.
4) **Telemetry**: Add structured logs for dataset creation/query/stats invocations (dataset_id, user_id, duration, path) to trace failures.

## Phase B: Remove parallel compute paths
1) **Route legacy endpoints through DuckDB**: Refactor `_get_session`/`_set_session` to read/write via DuckDB views instead of pandas when feasible; keep response shapes stable. Migrate `/query/{session_id}` to call `/datasets/{dataset_id}/query` internally.
2) **Unify stats implementations**: Reimplement legacy stats endpoints to call DuckDB-based functions in `app/routers/stats.py` while maintaining legacy payload compatibility. Add shims for any missing analyses.
3) **Transform engine alignment**: Rework transform endpoints to operate via DuckDB or convert results back to Parquet consistently; deprecate any pandas-only code paths after bridging.
4) **Cache hygiene**: Standardize cache directories and cleanup; ensure Parquet existence is validated before compute.

## Phase C: Cleanup, tests, and documentation
1) **Dead code removal**: Delete unused session-only helpers after DuckDB migration; reorganize modules for clarity without breaking imports (use re-export shims if needed).
2) **Test suite**: Expand smoke test to cover both dataset and legacy compatibility flows post-unification; add unit tests for Supabase registry/storage clients (mocked) and DuckDB manager SQL generation.
3) **Docs and diagrams**: Update README/ARCHITECTURE to describe the single-path flow, Supabase assets, and auth expectations; provide migration guide for session_id callers.
4) **Safety nets**: Add payload size guards and memory limits, document resource requirements, and ensure Docker image contains only needed artifacts.

## Risks and mitigations
- **Supabase availability**: Add retries/backoff and graceful degradation for KB endpoints; cache concept metadata when possible.
- **Large datasets**: Enforce file size/row limits before ingestion; stream uploads to disk to avoid memory spikes.
- **Optional dependencies**: Gate transform/agent features with clear error responses and health reporting when packages are missing.
