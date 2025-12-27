#!/usr/bin/env bash
set -u -o pipefail

BASE_URL=${BASE_URL:-"http://localhost:8000"}
AUTH_DISABLED=${AUTH_DISABLED:-false}
SMOKE_USER_ID=${SMOKE_USER_ID:-"smoke-test-user"}
SUPABASE_URL=${SUPABASE_URL:-""}
AUTH_TOKEN=${AUTH_TOKEN:-""}
SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY:-""}
SUPABASE_SERVICE_ROLE_KEY=${SUPABASE_SERVICE_ROLE_KEY:-""}
SMOKE_EMAIL=${SMOKE_EMAIL:-""}
SMOKE_PASSWORD=${SMOKE_PASSWORD:-""}

tmpdir=$(mktemp -d)
fixture="$tmpdir/smoke_fixture.csv"
failures=0

temp_cleanup() {
  rm -rf "$tmpdir"
}
trap temp_cleanup EXIT

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command '$1' not found in PATH" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd jq

fetch_auth_token() {
  if [[ "${AUTH_DISABLED,,}" == "true" ]]; then
    echo "Auth disabled; skipping token fetch"
    return
  fi

  if [[ -n "$AUTH_TOKEN" ]]; then
    echo "Using provided AUTH_TOKEN"
    return
  fi

  if [[ -n "${SUPABASE_ACCESS_TOKEN:-}" ]]; then
    AUTH_TOKEN="$SUPABASE_ACCESS_TOKEN"
    echo "Using auth token from SUPABASE_ACCESS_TOKEN"
    return
  fi

  if [[ -n "$SUPABASE_URL" && -n "$SMOKE_EMAIL" && -n "$SMOKE_PASSWORD" ]]; then
    api_key="${SUPABASE_SERVICE_ROLE_KEY:-$SUPABASE_ANON_KEY}"
    if [[ -z "$api_key" ]]; then
      echo "No SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY provided for auth token exchange" >&2
      return
    fi
    echo "Requesting token from Supabase for $SMOKE_EMAIL"
    token_resp=$(curl -sf \
      -H "apikey: $api_key" \
      -H "Content-Type: application/json" \
      -d "{\"email\":\"$SMOKE_EMAIL\",\"password\":\"$SMOKE_PASSWORD\"}" \
      "$SUPABASE_URL/auth/v1/token?grant_type=password" 2>/dev/null || true)
    if [[ -n "$token_resp" ]]; then
      AUTH_TOKEN=$(echo "$token_resp" | jq -r '.access_token // empty')
      if [[ -n "$AUTH_TOKEN" ]]; then
        echo "Obtained Supabase access token via password grant"
        return
      fi
    fi
    echo "Failed to obtain Supabase access token via password grant" >&2
  fi
}

fetch_auth_token

cat >"$fixture" <<'CSV'
name,age,score,group
Alice,30,0.50,control
Bob,40,0.70,treatment
Cara,35,0.60,control
Dan,45,0.90,treatment
Eve,38,0.80,control
CSV

AUTH_HEADERS=()
if [[ "${AUTH_DISABLED,,}" == "true" ]]; then
  echo "Auth disabled; sending requests without Authorization header"
else
  token_source=""
  token="$AUTH_TOKEN"
  if [[ -z "$token" && -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]]; then
    token="${SUPABASE_SERVICE_ROLE_KEY}"; token_source="SUPABASE_SERVICE_ROLE_KEY"
  fi
  if [[ -n "$token" ]]; then
    AUTH_HEADERS=(-H "Authorization: Bearer $token" -H "apikey: $token")
    echo "Using auth token from ${token_source:-environment}" | sed 's/\(.*\)/Auth: \1/'
  else
    echo "No auth token available; continuing without Authorization header"
  fi
fi

record_pass() { echo "PASS: $1"; }
record_fail() { echo "FAIL: $1"; failures=$((failures + 1)); }

health_check() {
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/health"); then
    record_pass "Health endpoint responded"
    echo "$resp"
  else
    record_fail "Health endpoint failed"
  fi
}

create_dataset() {
  echo "Uploading dataset fixture to /datasets"
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -F "file=@${fixture}" -F "user_id=${SMOKE_USER_ID}" "$BASE_URL/datasets"); then
    dataset_id=$(echo "$resp" | jq -r '.dataset_id // .profile.dataset_id // empty')
    if [[ -n "$dataset_id" ]]; then
      record_pass "Dataset created: $dataset_id"
    else
      record_fail "Dataset creation response missing dataset_id"
    fi
  else
    record_fail "Dataset upload failed"
  fi
}

await_dataset_ready() {
  if [[ -z "${dataset_id:-}" ]]; then
    record_fail "Dataset readiness skipped (no dataset_id)"
    return
  fi
  for i in {1..5}; do
    if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/datasets/${dataset_id}?user_id=${SMOKE_USER_ID}"); then
      has_schema=$(echo "$resp" | jq -r '(.profile.schema // .schema) | length' 2>/dev/null || echo "0")
      if [[ "$has_schema" != "0" ]]; then
        record_pass "Dataset metadata ready"
        echo "$resp"
        return
      fi
    fi
    sleep 1
  done
  record_fail "Dataset metadata not reachable after wait"
}

run_dataset_query() {
  if [[ -z "${dataset_id:-}" ]]; then
    record_fail "Dataset query skipped (no dataset_id)"
    return
  fi
  payload='{"sql":"SELECT COUNT(*) AS n FROM ds"}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$payload" "$BASE_URL/datasets/${dataset_id}/query?user_id=${SMOKE_USER_ID}"); then
    record_pass "Dataset query endpoint"
    echo "$resp"
  else
    record_fail "Dataset query endpoint"
  fi
}

run_dataset_stats() {
  if [[ -z "${dataset_id:-}" ]]; then
    record_fail "Dataset stats skipped (no dataset_id)"
    return
  fi
  declare -a analyses=(
    '{"analysis":"descriptives","params":{"columns":["age","score"]}}'
    '{"analysis":"ttest","params":{"type":"two_sample","x":"score","group":"group"}}'
    '{"analysis":"ttest","params":{"type":"one_sample","x":"score","mu":0.5}}'
    '{"analysis":"anova_oneway","params":{"y":"score","factor":"group"}}'
    '{"analysis":"regression_ols","params":{"y":"score","x":["age"]}}'
  )
  for body in "${analyses[@]}"; do
    if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
      -d "$body" "$BASE_URL/datasets/${dataset_id}/stats?user_id=${SMOKE_USER_ID}"); then
      record_pass "Dataset stats: $(echo "$body" | jq -r '.analysis')"
      echo "$resp"
    else
      record_fail "Dataset stats request failed for $(echo "$body" | jq -r '.analysis')"
    fi
  done
}

create_session() {
  echo "Uploading fixture to session-based /upload"
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -F "file=@${fixture}" "$BASE_URL/upload"); then
    session_id=$(echo "$resp" | jq -r '.session_id // empty')
    if [[ -n "$session_id" ]]; then
      record_pass "Session created: $session_id"
    else
      record_fail "Session upload missing session_id"
    fi
  else
    record_fail "Session upload failed"
  fi
}

run_legacy_analysis() {
  if [[ -z "${session_id:-}" ]]; then
    record_fail "Legacy analysis skipped (no session_id)"
    return
  fi
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/analysis/${session_id}"); then
    record_pass "Legacy analysis endpoint"
    echo "$resp" | head -c 400
  else
    record_fail "Legacy analysis endpoint"
  fi
}

run_session_query() {
  if [[ -z "${session_id:-}" ]]; then
    record_fail "Session query skipped (no session_id)"
    return
  fi
  payload='{"filters":[],"select":["age","score"],"aggregations":{}}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$payload" "$BASE_URL/query/${session_id}"); then
    record_pass "Session query endpoint"
    echo "$resp"
  else
    record_fail "Session query endpoint"
  fi
}

run_transforms() {
  if [[ -z "${session_id:-}" ]]; then
    record_fail "Transforms skipped (no session_id)"
    return
  fi
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/transform/${session_id}/suggest"); then
    record_pass "Transform suggestions"
  else
    record_fail "Transform suggestions"
  fi
  preview_body='{"column":"score","transform_type":"normalize","params":{"method":"zscore"}}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$preview_body" "$BASE_URL/session/${session_id}/transform/preview"); then
    record_pass "Transform preview"
    echo "$resp"
  else
    record_fail "Transform preview"
  fi

  apply_body='{"column":"score","transforms":[{"type":"normalize","params":{"method":"zscore"}}]}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$apply_body" "$BASE_URL/session/${session_id}/transform/apply"); then
    record_pass "Transform apply chain"
    echo "$resp"
  else
    record_fail "Transform apply chain"
  fi
}

run_extended_stats() {
  if [[ -z "${session_id:-}" ]]; then
    record_fail "Extended stats skipped (no session_id)"
    return
  fi

  adv_body='{"analysis_type":"normality","target":"score"}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$adv_body" "$BASE_URL/advanced-analysis/${session_id}"); then
    record_pass "Advanced analysis"
  else
    record_fail "Advanced analysis"
  fi

  ctrl_body='{"chart_type":"i","column":"score"}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$ctrl_body" "$BASE_URL/control-chart/${session_id}"); then
    record_pass "Control chart"
  else
    record_fail "Control chart"
  fi

  cap_body='{"column":"score","usl":1.0,"lsl":0.0,"target":0.5}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$cap_body" "$BASE_URL/process-capability/${session_id}"); then
    record_pass "Process capability"
  else
    record_fail "Process capability"
  fi

  reg_body='{"target":"score","predictors":["age"],"include_diagnostics":false}'
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$reg_body" "$BASE_URL/regression/${session_id}"); then
    record_pass "Regression analysis"
  else
    record_fail "Regression analysis"
  fi
}

run_data_views() {
  if [[ -z "${session_id:-}" ]]; then
    record_fail "Data view checks skipped (no session_id)"
    return
  fi

  if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/sample/${session_id}?n=3"); then
    record_pass "Sample rows"
  else
    record_fail "Sample rows"
  fi

  if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/schema/${session_id}"); then
    record_pass "Schema endpoint"
  else
    record_fail "Schema endpoint"
  fi

  if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/sessions/${session_id}/info"); then
    record_pass "Session info"
  else
    record_fail "Session info"
  fi

  if curl -sf "${AUTH_HEADERS[@]}" -o /dev/null "$BASE_URL/session/${session_id}/export?format=csv"; then
    record_pass "Session export"
  else
    record_fail "Session export"
  fi
}

run_kb() {
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" "$BASE_URL/kb/topics"); then
    record_pass "Knowledge base topics"
    echo "$resp"
  else
    record_fail "Knowledge base topics"
  fi
}

run_agents() {
  if [[ -z "${session_id:-}" ]]; then
    record_fail "Agent test skipped (no session_id)"
    return
  fi
  payload=$(jq -n --arg sid "$session_id" '{session_id:$sid, focus_areas:["summary"]}')
  if resp=$(curl -sf "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
    -d "$payload" "$BASE_URL/agents/explore"); then
    record_pass "Agent explore endpoint"
    echo "$resp"
  else
    record_fail "Agent explore endpoint"
  fi
}

main() {
  echo "=== Smoke test against ${BASE_URL} ==="
  health_check
  create_dataset
  await_dataset_ready
  run_dataset_query
  run_dataset_stats
  create_session
  run_legacy_analysis
  run_session_query
  run_transforms
  run_extended_stats
  run_data_views
  run_kb
  run_agents

  if ((failures > 0)); then
    echo "SMOKE TEST FAILED with ${failures} error(s)"
    exit 1
  fi
  echo "SMOKE TEST PASSED"
}

main "$@"
