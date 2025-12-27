# Testing

## Phase A verification (startup + compatibility)
Prerequisites: running backend, `curl`, and `jq` in PATH.

1) Start the server (`uvicorn main:app --host 0.0.0.0 --port 8000`).
2) Check `/health`:
   ```bash
   curl -s http://localhost:8000/health | jq
   ```
   - `status: ok` means DB (via `DATABASE_URL`), storage, and KB checks are reachable and required env vars are present (see `checks.env.required`).
   - `status: degraded` means some inputs are missing/unreachable; inspect the `errors` array for actionable items (missing env vars, storage errors, missing `DATABASE_URL`, etc.).
3) Verify session↔dataset persistence survives restarts:
   - Upload a file and capture `session_id`:
     ```bash
     upload=$(curl -sf -F "file=@/path/to/data.csv" -F "user_id=tester" http://localhost:8000/upload)
     session_id=$(echo "$upload" | jq -r '.session_id')
     ```
   - Restart the service, then call a legacy endpoint with the same `session_id` (e.g., `/analysis/{session_id}`). If the Supabase `session_dataset_links` table is configured, the mapping should persist and the endpoint should return successfully without re-uploading.

## Quick smoke test (hits every endpoint once)
Prerequisites: running backend, `curl`, and `jq` in PATH.

### Local (auth disabled)
```bash
BASE_URL="http://localhost:8000" AUTH_DISABLED=true bash scripts/smoke_test.sh
```

### Railway/remote (token auth or Supabase password grant)
Use an explicit token (service role recommended for testing):
```bash
BASE_URL="https://<your-host>" \
SMOKE_USER_ID="smoke-test-user" \
AUTH_TOKEN="$SUPABASE_SERVICE_ROLE_KEY" \
bash scripts/smoke_test.sh
```

Or let the script exchange Supabase credentials for a token:
```bash
BASE_URL="https://<your-host>" \
SUPABASE_URL="https://<your-supabase-ref>.supabase.co" \
SMOKE_EMAIL="tester@example.com" SMOKE_PASSWORD="password" \
SUPABASE_ANON_KEY="<anon-or-service-key>" \
bash scripts/smoke_test.sh
```

The script uploads a fixture, waits for dataset profiling to be readable, exercises dataset + legacy session endpoints (analysis, 5+ stats concepts, transforms including suggest/preview/apply, regression, control chart, process capability, exports), KB, and agent routes. It prints PASS/FAIL and exits non-zero on failure.

## Manual cURL coverage
Export a few helpers first:

```bash
BASE_URL=${BASE_URL:-"http://localhost:8000"}
AUTH_DISABLED=${AUTH_DISABLED:-true}
AUTH_HEADER=()
if [[ "${AUTH_DISABLED,,}" != "true" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY")
fi
```

### Health
```bash
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/health"
```

### Dataset-first flow (canonical)
Upload and capture `dataset_id`:
```bash
resp=$(curl -sf "${AUTH_HEADER[@]}" -F "file=@/path/to/data.csv" -F "user_id=tester" "$BASE_URL/datasets")
dataset_id=$(echo "$resp" | jq -r '.dataset_id')
```

Profile from registry:
```bash
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/datasets/${dataset_id}?user_id=tester"
```

DuckDB query (SQL or QuerySpec):
```bash
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"sql":"SELECT * FROM dataset LIMIT 5"}' \
  "$BASE_URL/datasets/${dataset_id}/query?user_id=tester"
```

DuckDB stats (descriptives + other analyses):
```bash
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"analysis":"descriptives","params":{"columns":["col1","col2"]}}' \
  "$BASE_URL/datasets/${dataset_id}/stats?user_id=tester"
```

### Legacy session compatibility (session_id maps to dataset_id)
Create session via upload (returns `session_id` which equals the dataset_id):
```bash
resp=$(curl -sf "${AUTH_HEADER[@]}" -F "file=@/path/to/data.csv" "$BASE_URL/upload")
session_id=$(echo "$resp" | jq -r '.session_id')
```

Bridge an existing dataset into a session:
```bash
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/rehydrate/dataset/${dataset_id}?user_id=tester"
```

Core stats endpoints (all backed by the dataset parquet):
```bash
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/analysis/${session_id}"
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"analysis_type":"normality","target":"score"}' \
  "$BASE_URL/advanced-analysis/${session_id}"
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"chart_type":"i","column":"score"}' \
  "$BASE_URL/control-chart/${session_id}"
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"column":"score","usl":1,"lsl":0,"target":0.5}' \
  "$BASE_URL/process-capability/${session_id}"
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"target":"score","predictors":["age"],"include_diagnostics":false}' \
  "$BASE_URL/regression/${session_id}"
```

Session query (DuckDB-backed compatibility path):
```bash
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"filters":[],"select":["age","score"],"aggregations":{}}' \
  "$BASE_URL/query/${session_id}"
```

Transform endpoints:
```bash
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  "$BASE_URL/transform/${session_id}/suggest"

curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"column":"score","transform_type":"normalize","params":{"method":"zscore"}}' \
  "$BASE_URL/session/${session_id}/transform/preview"

curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"column":"score","transforms":[{"type":"normalize","params":{"method":"zscore"}}]}' \
  "$BASE_URL/session/${session_id}/transform/apply"
```

Data access + export:
```bash
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/sample/${session_id}?n=5"
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/schema/${session_id}"
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/sessions/${session_id}/info"
curl -sf "${AUTH_HEADER[@]}" -o /tmp/export.csv "$BASE_URL/session/${session_id}/export?format=csv"
```

### Knowledge base + agents
```bash
curl -sf "${AUTH_HEADER[@]}" "$BASE_URL/kb/topics"
curl -sf "${AUTH_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"session_id":"'"$session_id"'","focus_areas":["summary"]}' \
  "$BASE_URL/agents/explore"
```
