# Architecture

```
[Client]
   |  (multipart upload + user_id)
   v
/datasets                /upload (compat)
   |                         |d
   |  create_dataset_from_upload
   |  - write raw to cache
   |  - build parquet
   |  - upload raw/parquet to Supabase Storage
   |  - registry row in Postgres (dataset_id canonical)
   v
[Parquet on disk + Supabase]
   |
   |  (DuckDB reads parquet for all compute)
   v
[DuckDB Manager]
   |\
   | \_ /datasets/{dataset_id}/query
   |   \_/datasets/{dataset_id}/stats
   |
   |  (legacy compatibility)
   |  session_bridge maps session_id -> dataset_id,user_id
   |  _get_session reads parquet via mapping
   |  _set_session rewrites parquet + registry via persist_dataframe
   |  session endpoints call pandas/scipy logic using dataframe loaded from parquet
   |
   +--> agents/ KB/ transforms reuse _get_session
```

Key points:
- `dataset_id` is the single source of truth. Legacy `session_id` values are thin aliases stored in `session_bridge`.
- All stats and query paths rely on DuckDB over the cached parquet file; updates rewrite the parquet and patch Supabase registry + storage.
- Supabase services remain centralized in `app/services` (storage, registry, dataset creator, session bridge) with cache paths under `./cache`.
