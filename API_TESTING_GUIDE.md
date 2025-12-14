# API Testing Guide

Quick reference for testing your transform engine endpoints.

## 🚀 Quick Start

### 1. Start the Server
```bash
python main.py
```

Server will start at: `http://localhost:8000`  
API Docs: `http://localhost:8000/docs`

---

## 🧪 Test Sequence

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{
  "status": "ok",
  "version": "2.0.0",
  "features": ["transforms", "stats", "query", "table_ops"]
}
```

---

### Test 2: Upload Sample Data
```bash
# Create a sample CSV first
echo "date,product,amount,quantity
2024-01-15,Widget A,150.00,5
2024-01-20,Widget B,200.50,3
2024-02-10,Widget A,175.25,6
2024-02-15,Widget C,99.99,2" > sample_sales.csv

# Upload it
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_sales.csv"
```

**Save the session_id from response!**

---

### Test 3: Get Available Transforms
```bash
# All transforms
curl http://localhost:8000/transforms | jq

# For datetime columns only
curl http://localhost:8000/transforms/for/datetime | jq
```

---

### Test 4: Get Transform Suggestions
```bash
# Replace SESSION_ID with actual ID from Test 2
curl -X POST http://localhost:8000/session/SESSION_ID/suggest/date | jq
```

---

### Test 5: Preview Transform
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/transform/preview \
  -H "Content-Type: application/json" \
  -d '{
    "column": "date",
    "transform_type": "month",
    "params": {"format": "name"},
    "n_rows": 10
  }' | jq
```

---

### Test 6: Apply Transform
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/transform/apply \
  -H "Content-Type: application/json" \
  -d '{
    "column": "date",
    "transforms": [
      {
        "type": "month",
        "params": {"format": "name"}
      }
    ],
    "new_column_name": "sale_month",
    "replace_original": false
  }' | jq
```

---

### Test 7: Batch Apply Multiple Transforms
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/transform/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transforms": {
      "sale_month": {
        "column": "date",
        "transforms": [{"type": "month", "params": {"format": "name"}}]
      },
      "sale_year": {
        "column": "date",
        "transforms": [{"type": "year", "params": {}}]
      },
      "amount_log": {
        "column": "amount",
        "transforms": [{"type": "log_transform", "params": {"base": 10}}]
      }
    }
  }' | jq
```

---

### Test 8: Group By & Aggregate
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/group_by \
  -H "Content-Type: application/json" \
  -d '{
    "group_by": ["product"],
    "aggregations": {
      "total_sales": "amount:sum",
      "avg_quantity": "quantity:mean",
      "num_orders": "amount:count"
    },
    "create_new_session": true
  }' | jq
```

**This creates a NEW session with grouped data!**

---

### Test 9: Filter Data
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/filter \
  -H "Content-Type: application/json" \
  -d '{
    "filters": [
      {
        "column": "amount",
        "operator": "gt",
        "value": 100
      },
      {
        "column": "product",
        "operator": "contains",
        "value": "Widget"
      }
    ],
    "create_new_session": false
  }' | jq
```

---

### Test 10: Statistical Analysis
```bash
curl http://localhost:8000/session/SESSION_ID/analysis | jq
```

---

### Test 11: Export Data
```bash
# As CSV
curl "http://localhost:8000/session/SESSION_ID/export?format=csv" \
  -o exported_data.csv

# As JSON
curl "http://localhost:8000/session/SESSION_ID/export?format=json" \
  -o exported_data.json
```

---

## 🧩 Advanced Test Cases

### Complex Transform Chain
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/transform/apply \
  -H "Content-Type: application/json" \
  -d '{
    "column": "amount",
    "transforms": [
      {"type": "log_transform", "params": {"base": 10}},
      {"type": "z_score", "params": {}},
      {"type": "bin", "params": {
        "bins": [-3, -1, 1, 3],
        "labels": ["very_low", "low", "high", "very_high"]
      }}
    ],
    "new_column_name": "amount_category"
  }' | jq
```

### Merge Two Sessions
```bash
# First create second session with different data
curl -X POST http://localhost:8000/upload \
  -F "file=@customer_data.csv"

# Save the second session ID, then merge:
curl -X POST http://localhost:8000/session/SESSION_ID_1/merge/SESSION_ID_2 \
  -H "Content-Type: application/json" \
  -d '{
    "on": ["customer_id"],
    "how": "inner"
  }' | jq
```

### Pivot Table
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/pivot \
  -H "Content-Type: application/json" \
  -d '{
    "index": ["sale_month"],
    "columns": "product",
    "values": "amount",
    "aggfunc": "sum"
  }' | jq
```

### Advanced Query
```bash
curl -X POST http://localhost:8000/session/SESSION_ID/query \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "aggregate",
    "filters": [
      {"column": "amount", "operator": "gt", "value": 50}
    ],
    "transforms": {
      "date": {
        "type": "month",
        "params": {"format": "name"}
      }
    },
    "virtual_columns": {
      "revenue": {
        "type": "expression",
        "params": {
          "expression": "df[\"amount\"] * df[\"quantity\"]"
        }
      }
    },
    "group_by": ["date_transformed", "product"],
    "aggregations": {
      "total_revenue": "revenue:sum",
      "avg_amount": "amount:mean"
    },
    "limit": 50
  }' | jq
```

---

## 📊 Testing with Python

### Simple Test Script
```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Upload
with open("sample_sales.csv", "rb") as f:
    response = requests.post(f"{BASE_URL}/upload", files={"file": f})
    data = response.json()
    session_id = data["session_id"]
    print(f"Session ID: {session_id}")

# 2. Get suggestions
response = requests.post(f"{BASE_URL}/session/{session_id}/suggest/date")
suggestions = response.json()
print("Suggestions:", json.dumps(suggestions, indent=2))

# 3. Apply transform
transform_request = {
    "column": "date",
    "transforms": [
        {"type": "month", "params": {"format": "name"}}
    ],
    "new_column_name": "sale_month"
}
response = requests.post(
    f"{BASE_URL}/session/{session_id}/transform/apply",
    json=transform_request
)
print("Transform result:", response.json())

# 4. Group by
groupby_request = {
    "group_by": ["sale_month"],
    "aggregations": {
        "total_sales": "amount:sum"
    }
}
response = requests.post(
    f"{BASE_URL}/session/{session_id}/group_by",
    json=groupby_request
)
print("GroupBy result:", response.json())

# 5. Export
response = requests.get(f"{BASE_URL}/session/{session_id}/export?format=csv")
with open("output.csv", "wb") as f:
    f.write(response.content)
print("Exported to output.csv")
```

---

## 🎯 Common Test Scenarios

### Scenario 1: Date Analysis
```bash
# 1. Extract month
curl -X POST http://localhost:8000/session/SESSION_ID/transform/apply \
  -H "Content-Type: application/json" \
  -d '{"column": "date", "transforms": [{"type": "month"}], "new_column_name": "month"}'

# 2. Extract year
curl -X POST http://localhost:8000/session/SESSION_ID/transform/apply \
  -H "Content-Type: application/json" \
  -d '{"column": "date", "transforms": [{"type": "year"}], "new_column_name": "year"}'

# 3. Group by month and year
curl -X POST http://localhost:8000/session/SESSION_ID/group_by \
  -H "Content-Type: application/json" \
  -d '{"group_by": ["year", "month"], "aggregations": {"sales": "amount:sum"}}'
```

### Scenario 2: Data Cleaning
```bash
# 1. Remove duplicates
curl -X POST http://localhost:8000/session/SESSION_ID/remove_duplicates

# 2. Fill missing values
curl -X POST http://localhost:8000/session/SESSION_ID/fill_missing \
  -H "Content-Type: application/json" \
  -d '{"column": "amount", "method": "mean"}'

# 3. Filter valid rows
curl -X POST http://localhost:8000/session/SESSION_ID/filter \
  -H "Content-Type: application/json" \
  -d '{"filters": [{"column": "amount", "operator": "gt", "value": 0}]}'
```

---

## ✅ Validation Checklist

- [ ] Server starts without errors
- [ ] Health check returns OK
- [ ] File upload succeeds
- [ ] Transform catalog loads
- [ ] Suggestions work for each column type
- [ ] Preview shows correct transformations
- [ ] Apply transform creates new column
- [ ] Batch transforms all succeed
- [ ] Group by produces aggregated results
- [ ] Pivot/unpivot reshapes correctly
- [ ] Merge joins tables properly
- [ ] Filter reduces row count
- [ ] Export downloads files
- [ ] Stats analysis runs without errors

---

## 🐛 Troubleshooting

### Error: "Session not found"
- Check if session_id is correct
- Sessions expire after 24 hours
- Use fresh session_id from recent upload

### Error: "Column not found"
- Verify column name exactly matches (case-sensitive)
- Check session profile: `GET /session/{id}/profile`

### Error: "Invalid transform parameters"
- Check transform definition: `GET /transforms`
- Preview transform first: `POST /transform/preview`

### Error: Connection refused
- Ensure server is running: `python main.py`
- Check port 8000 is not in use
- Try: `lsof -i :8000` to see what's using it

---

## 📚 More Examples

See `README.md` for complete API documentation and use cases.

Interactive API docs: `http://localhost:8000/docs`
