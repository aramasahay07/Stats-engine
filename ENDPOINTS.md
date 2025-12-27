# Endpoint Inventory

Auth model: No runtime auth middleware. Callers are expected to supply `user_id` query params (dataset endpoints) or session IDs; Supabase keys are server-side. If Supabase env vars are missing, dataset/KB endpoints raise at runtime.

## Health and meta
| Method | Path | Description | Auth |
| --- | --- | --- | --- |
| GET | `/health` | Preflight health with DB/storage checks (imported from `app.services.preflight`). Duplicate definition exists later in `main.py`. | None |
| GET | `/health` (legacy) | Legacy liveness report with feature flags in `main.py`. | None |
| GET | `/healthz` | Alias to legacy health. | None |
| GET | `/` | Root metadata + endpoint list. | None |

## Dataset-first (DuckDB) endpoints
| Method | Path | Description | Auth/Params |
| --- | --- | --- | --- |
| POST | `/datasets` | Create dataset from upload; stores raw/parquet to Supabase, registers row. Body: multipart file + `user_id` (Form) + optional `project_id`. | None enforced; requires Supabase env vars and `user_id` field |
| GET | `/datasets/{dataset_id}` | Fetch dataset metadata from Supabase registry. Requires `user_id` query param. | None enforced; Supabase registry access |
| POST | `/datasets/{dataset_id}/query` | Run DuckDB SQL or structured QuerySpec over Parquet. Requires `user_id` query param. | None enforced; Parquet must be present/accessible |
| POST | `/datasets/{dataset_id}/stats` | DuckDB-backed stats (descriptives, t-test, anova, regression). Requires `user_id` query param. | None enforced; Parquet + Supabase reachable |

## Legacy session endpoints (pandas-based)
| Method | Path | Description | Auth |
| --- | --- | --- | --- |
| POST | `/upload` | Upload CSV/Excel, profile, return `session_id`; bridges to dataset storage/registry under the hood. | None |
| POST | `/rehydrate/dataset/{dataset_id}` | Load dataset by id for UI reuse; returns new session bound to dataset Parquet. Query params: `user_id`, optional `project_id`. | None |
| GET | `/analysis/{session_id}` | Comprehensive stats analysis (correlation/tests/regression/normality). | None |
| POST | `/advanced-analysis/{session_id}` | Advanced analytics (normality, variance tests, time series, PCA, clustering). | None |
| POST | `/control-chart/{session_id}` | Control charts (X-bar/I/P charts). | None |
| POST | `/process-capability/{session_id}` | Process capability metrics. | None |
| POST | `/regression/{session_id}` | Regression with diagnostics. | None |
| POST | `/transform/{session_id}` | Apply legacy transforms via TransformService. | None; depends on `transform_service.py` |
| GET | `/transform/{session_id}/suggest` | Suggest transforms based on data types. | None |
| GET | `/sessions/{session_id}/info` | Return info/profile for session. | None |
| DELETE | `/sessions/{session_id}` | Delete session (in-memory bridge entry). | None |
| GET | `/sample/{session_id}` | Return sample rows. | None |
| GET | `/schema/{session_id}` | Return schema info. | None |
| POST | `/query/{session_id}` | Run SQL-like queries via pandas; not DuckDB. | None |
| GET | `/transforms` | List available transforms (conditional on transformers package). | None |
| GET | `/transforms/for/{column_type}` | List transforms for a column type. | None |
| POST | `/session/{session_id}/suggest/{column}` | Suggest transforms for a column. | None |
| POST | `/session/{session_id}/transform/preview` | Preview transform. | None |
| POST | `/session/{session_id}/transform/apply` | Apply transform. | None |
| POST | `/session/{session_id}/transform/batch` | Apply multiple transforms. | None |
| POST | `/session/{session_id}/group_by` | Group by operation. | None |
| POST | `/session/{session_id}/pivot` | Pivot operation. | None |
| POST | `/session/{session_id}/unpivot` | Unpivot operation. | None |
| POST | `/session/{session_id}/merge/{other_session_id}` | Merge two sessions. | None |
| POST | `/session/{session_id}/remove_duplicates` | Deduplicate rows. | None |
| POST | `/session/{session_id}/fill_missing` | Fill missing values. | None |
| POST | `/session/{session_id}/filter` | Filter rows. | None |
| GET | `/session/{session_id}/export` | Export session as CSV. | None |

## Knowledge Base endpoints
| Method | Path | Description | Auth |
| --- | --- | --- | --- |
| GET | `/kb/topics` | List KB topics from `stat_topics`. | None; uses Supabase anon/service key |
| GET | `/kb/concepts/{slug}` | Fetch concept with related children/aliases/resources. | None; Supabase REST |
| GET | `/kb/search` | Search concepts/aliases. | None |
| POST | `/kb/enrich` | Enrich stats payload. | None |

## Agent endpoints
| Method | Path | Description | Auth |
| --- | --- | --- | --- |
| POST | `/agents/explore` | Agent-driven exploratory analysis on session DataFrame. | None |
| POST | `/agents/patterns` | Pattern detection agent. | None |
| POST | `/agents/causality` | Causality agent. | None |
| POST | `/agents/unified` | Runs explore/patterns/causality sequence and returns combined findings. | None |
