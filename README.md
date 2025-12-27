# AI Data Lab - Stats & Transform Engine v2.0

**Advanced data transformation engine with 60+ intelligent transforms + statistical analysis**

## 🚀 What's New in v2.0

### ✅ Power Query-Level Capabilities

Your engine now has **FULL Power Query equivalency** plus Python-powered extras:

#### Column-Level Transforms (60+)
- ✅ **DateTime**: Extract month, year, quarter, weekday, fiscal periods, age calculations
- ✅ **Numeric**: Binning, scaling, log transforms, z-scores, percentiles
- ✅ **Text**: Cleaning, extraction, parsing, case transforms
- ✅ **Categorical**: Encoding, grouping, frequency-based transforms
- ✅ **Smart/ML**: Outlier detection, fuzzy matching, pattern recognition

#### Table-Level Operations
- ✅ **Group By & Aggregate**: Multi-column grouping with custom aggregations
- ✅ **Pivot/Unpivot**: Reshape data structures
- ✅ **Merge/Join**: Inner, outer, left, right joins
- ✅ **Filter**: Complex multi-condition filtering
- ✅ **Remove Duplicates**: Configurable duplicate handling
- ✅ **Fill Missing**: Forward fill, backward fill, mean/median/mode imputation

#### Advanced Features
- ✅ **Transform Discovery**: AI-suggested transforms per column
- ✅ **Transform Preview**: See results before applying
- ✅ **Transform Chains**: Combine multiple transforms
- ✅ **Batch Transforms**: Apply many transforms at once
- ✅ **Session Management**: Multiple datasets with TTL
- ✅ **Export**: CSV and JSON export

---

## 📖 Complete API Reference

### Base URL
```
http://localhost:8000
```

---

## 🔧 Core Endpoints

### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "version": "2.0.0",
  "features": ["transforms", "stats", "query", "table_ops"]
}
```

### System Stats
```http
GET /stats
```
**Response:**
```json
{
  "active_sessions": 3,
  "cached_transforms": 12,
  "total_memory_mb": 45.2
}
```

---

## 📁 Session Management

### Upload File
```http
POST /upload
Content-Type: multipart/form-data

file: [CSV or Excel file]
```
**Response:**
```json
{
  "session_id": "abc-123-def",
  "n_rows": 1000,
  "n_cols": 15,
  "columns": [...],
  "schema": [...],
  "descriptives": [...]
}
```

### Get Session Profile
```http
GET /session/{session_id}/profile
```

### Delete Session
```http
DELETE /session/{session_id}
```

---

## 🔍 Transform Discovery

### Get All Available Transforms
```http
GET /transforms
```
**Response:**
```json
{
  "transforms": {
    "month": {
      "input_types": ["datetime"],
      "output_type": "categorical",
      "params": {
        "format": {
          "type": "string",
          "values": ["name", "number", "short"],
          "default": "name"
        }
      },
      "description": "Extract month from datetime",
      "examples": ["January", "1", "Jan"]
    },
    ...
  }
}
```

### Get Transforms for Column Type
```http
GET /transforms/for/{column_type}

Examples:
GET /transforms/for/datetime
GET /transforms/for/numeric
GET /transforms/for/text
```

### Get AI-Powered Suggestions
```http
POST /session/{session_id}/suggest/{column}?limit=5
```
**Response:**
```json
{
  "column": "birth_date",
  "detected_type": "datetime",
  "suggested_transforms": [
    {
      "transform": "age_from_date",
      "usefulness_score": 0.95,
      "reason": "Calculate age for demographic analysis",
      "preview": [32, 45, 28, 51, 39],
      "params": {"reference_date": "today"}
    },
    {
      "transform": "year",
      "usefulness_score": 0.82,
      "reason": "Group by birth year for cohort analysis",
      "preview": [1992, 1979, 1996, 1973, 1985],
      "params": {}
    }
  ]
}
```

---

## ⚙️ Transform Application

### Preview Transform (No Changes)
```http
POST /session/{session_id}/transform/preview

Body:
{
  "column": "order_date",
  "transform_type": "month",
  "params": {"format": "name"},
  "n_rows": 100
}
```
**Response:**
```json
{
  "original": ["2024-01-15", "2024-02-20", ...],
  "transformed": ["January", "February", ...],
  "null_count": 0,
  "unique_values": 12,
  "sample_values": ["January", "February", "March", ...]
}
```

### Apply Single Transform
```http
POST /session/{session_id}/transform/apply

Body:
{
  "column": "order_date",
  "transforms": [
    {
      "type": "month",
      "params": {"format": "name"}
    }
  ],
  "new_column_name": "order_month",
  "replace_original": false
}
```
**Response:**
```json
{
  "success": true,
  "column_created": "order_month",
  "metadata": [
    {
      "source_column": "order_date",
      "transform_type": "month",
      "null_count": 0,
      "unique_values": 12,
      "sample_output": ["January", "February", "March", "April", "May"]
    }
  ],
  "n_rows": 1000
}
```

### Apply Transform Chain
```http
POST /session/{session_id}/transform/apply

Body:
{
  "column": "price",
  "transforms": [
    {
      "type": "log_transform",
      "params": {"base": 10}
    },
    {
      "type": "z_score",
      "params": {}
    },
    {
      "type": "bin",
      "params": {
        "bins": [-3, -1, 1, 3],
        "labels": ["low", "medium", "high"]
      }
    }
  ],
  "new_column_name": "price_category"
}
```

### Batch Apply Multiple Transforms
```http
POST /session/{session_id}/transform/batch

Body:
{
  "transforms": {
    "order_month": {
      "column": "order_date",
      "transforms": [{"type": "month", "params": {"format": "name"}}]
    },
    "order_year": {
      "column": "order_date",
      "transforms": [{"type": "year", "params": {}}]
    },
    "price_log": {
      "column": "price",
      "transforms": [{"type": "log_transform", "params": {"base": 10}}]
    }
  }
}
```
**Response:**
```json
{
  "success": true,
  "results": {
    "order_month": {"success": true, "metadata": [...]},
    "order_year": {"success": true, "metadata": [...]},
    "price_log": {"success": true, "metadata": [...]}
  },
  "errors": null
}
```

---

## 📊 Table Operations (Power Query Level)

### Group By & Aggregate
```http
POST /session/{session_id}/group_by

Body:
{
  "group_by": ["region", "product_category"],
  "aggregations": {
    "total_sales": "amount:sum",
    "avg_price": "price:mean",
    "num_orders": "order_id:count",
    "max_quantity": "quantity:max"
  },
  "create_new_session": true
}
```
**Response:**
```json
{
  "success": true,
  "session_id": "new-xyz-789",
  "n_rows": 48,
  "n_cols": 6
}
```

**Supported Aggregation Functions:**
- `sum`, `mean`, `median`, `min`, `max`
- `count`, `std`, `var`
- `first`, `last`

### Pivot Table
```http
POST /session/{session_id}/pivot

Body:
{
  "index": ["region"],
  "columns": "product_category",
  "values": "sales",
  "aggfunc": "sum"
}
```

### Unpivot (Melt)
```http
POST /session/{session_id}/unpivot

Body:
{
  "id_vars": ["date", "store_id"],
  "value_vars": ["product_a_sales", "product_b_sales", "product_c_sales"],
  "var_name": "product",
  "value_name": "sales_amount"
}
```

### Merge/Join Tables
```http
POST /session/{session_id}/merge/{other_session_id}

Body:
{
  "on": ["customer_id"],
  "how": "inner"
}

OR for different column names:
{
  "left_on": ["cust_id"],
  "right_on": ["customer_id"],
  "how": "left"
}
```

**Join Types:** `inner`, `left`, `right`, `outer`

### Remove Duplicates
```http
POST /session/{session_id}/remove_duplicates

Body:
{
  "subset": ["email", "phone"],
  "keep": "first"
}
```

### Fill Missing Values
```http
POST /session/{session_id}/fill_missing

Body:
{
  "column": "age",
  "method": "mean"
}
```

**Fill Methods:**
- `ffill` - Forward fill
- `bfill` - Backward fill
- `mean` - Fill with mean
- `median` - Fill with median
- `mode` - Fill with mode
- Or provide `value` directly: `{"value": 0}`

### Filter Rows
```http
POST /session/{session_id}/filter

Body:
{
  "filters": [
    {
      "column": "age",
      "operator": "gte",
      "value": 18
    },
    {
      "column": "country",
      "operator": "in",
      "value": ["USA", "Canada", "Mexico"]
    },
    {
      "column": "email",
      "operator": "contains",
      "value": "@company.com"
    }
  ],
  "create_new_session": false
}
```

**Filter Operators:**
- `eq`, `ne` - Equal, Not equal
- `gt`, `lt`, `gte`, `lte` - Greater/Less than (or equal)
- `in`, `not_in` - In list
- `contains` - String contains
- `starts_with` - String starts with

---

## 📈 Statistical Analysis

### Full Analysis
```http
GET /session/{session_id}/analysis
```
**Response:**
```json
{
  "session_id": "abc-123",
  "correlation": {
    "matrix": {
      "age": {"age": 1.0, "income": 0.45, "expenses": 0.32},
      "income": {"age": 0.45, "income": 1.0, "expenses": 0.78},
      "expenses": {"age": 0.32, "income": 0.78, "expenses": 1.0}
    }
  },
  "tests": [
    {
      "test_type": "t-test",
      "target": "salary",
      "group_col": "department",
      "p_value": 0.003,
      "statistic": 2.89,
      "interpretation": "Difference between the two groups is statistically significant."
    }
  ],
  "regression": {
    "target": "sales",
    "predictors": ["marketing_spend", "seasonality_index"],
    "r_squared": 0.72,
    "adj_r_squared": 0.71,
    "coefficients": {
      "const": 1250.5,
      "marketing_spend": 0.45,
      "seasonality_index": 320.8
    }
  }
}
```

### Correlation Only
```http
GET /session/{session_id}/correlation
```

---

## 🔬 Advanced Query Engine

### Complex Query with Everything
```http
POST /session/{session_id}/query

Body:
{
  "operation": "aggregate",
  "filters": [
    {"column": "status", "operator": "eq", "value": "active"},
    {"column": "revenue", "operator": "gt", "value": 1000}
  ],
  "transforms": {
    "signup_date": {
      "type": "month_year",
      "params": {"format": "short"}
    }
  },
  "virtual_columns": {
    "profit": {
      "type": "expression",
      "params": {
        "expression": "df['revenue'] - df['cost']"
      }
    }
  },
  "group_by": ["region", "signup_date_transformed"],
  "aggregations": {
    "total_revenue": "revenue:sum",
    "avg_profit": "profit:mean",
    "customer_count": "customer_id:count"
  },
  "limit": 100
}
```

---

## 💾 Data Export

### Export as CSV
```http
GET /session/{session_id}/export?format=csv
```

### Export as JSON
```http
GET /session/{session_id}/export?format=json
```

---

## 🎯 Common Use Cases

### Use Case 1: Date Analysis Pipeline
```javascript
// 1. Upload file
POST /upload

// 2. Get suggestions for date column
POST /session/abc-123/suggest/order_date

// 3. Create time-based features
POST /session/abc-123/transform/batch
{
  "transforms": {
    "order_month": {"column": "order_date", "transforms": [{"type": "month"}]},
    "order_quarter": {"column": "order_date", "transforms": [{"type": "quarter", "params": {"include_year": true}}]},
    "is_weekend": {"column": "order_date", "transforms": [{"type": "time_features", "params": {"features": ["is_weekend"]}}]}
  }
}

// 4. Aggregate by time period
POST /session/abc-123/group_by
{
  "group_by": ["order_month"],
  "aggregations": {
    "total_sales": "amount:sum",
    "order_count": "order_id:count"
  }
}
```

### Use Case 2: Customer Segmentation
```javascript
// 1. Create age groups from birthdate
POST /session/abc-123/transform/apply
{
  "column": "birth_date",
  "transforms": [
    {"type": "age_from_date"},
    {"type": "bin", "params": {
      "bins": [0, 25, 45, 65, 100],
      "labels": ["18-24", "25-44", "45-64", "65+"]
    }}
  ],
  "new_column_name": "age_group"
}

// 2. Create spending category
POST /session/abc-123/transform/apply
{
  "column": "total_spent",
  "transforms": [
    {"type": "quantile_bin", "params": {
      "quantiles": [0, 0.33, 0.67, 1.0],
      "labels": ["low", "medium", "high"]
    }}
  ],
  "new_column_name": "spending_tier"
}

// 3. Group and analyze
POST /session/abc-123/group_by
{
  "group_by": ["age_group", "spending_tier"],
  "aggregations": {
    "customer_count": "customer_id:count",
    "avg_purchase": "total_spent:mean"
  }
}
```

### Use Case 3: Sales Performance Analysis
```javascript
// 1. Create fiscal quarter
POST /session/abc-123/transform/apply
{
  "column": "sale_date",
  "transforms": [
    {"type": "fiscal_quarter", "params": {
      "fiscal_start_month": 4,
      "include_year": true
    }}
  ],
  "new_column_name": "fiscal_quarter"
}

// 2. Calculate growth metrics
POST /session/abc-123/query
{
  "operation": "aggregate",
  "virtual_columns": {
    "profit_margin": {
      "type": "expression",
      "params": {"expression": "(df['revenue'] - df['cost']) / df['revenue'] * 100"}
    }
  },
  "group_by": ["region", "fiscal_quarter"],
  "aggregations": {
    "total_revenue": "revenue:sum",
    "avg_margin": "profit_margin:mean"
  }
}
```

---

## 📋 Transform Catalog

### DateTime Transforms
- `month` - Extract month (name, number, or short)
- `month_year` - Month and year combined
- `year` - Extract year
- `quarter` - Extract quarter (Q1, Q2, Q3, Q4)
- `fiscal_quarter` - Fiscal quarter with custom year start
- `weekday` - Day of week (name, number, or short)
- `week` - Week number
- `day` - Day of month
- `hour` - Hour of day
- `date_only` - Remove time component
- `time_features` - Boolean features (is_weekend, is_month_end, etc.)
- `age_from_date` - Calculate age from birthdate

### Numeric Transforms
- `bin` - Custom binning with labels
- `quantile_bin` - Percentile-based binning
- `equal_width_bin` - Equal-width intervals
- `z_score` - Standardize to z-scores
- `min_max_scale` - Scale to 0-1 range
- `log_transform` - Logarithmic transformation
- `power_transform` - Power transformation
- `percentile_rank` - Convert to percentile ranks
- `round` - Round to decimals
- `abs` - Absolute value
- `clip` - Clip to min/max range

### Text Transforms
- `lowercase` - Convert to lowercase
- `uppercase` - Convert to UPPERCASE
- `title_case` - Convert to Title Case
- `trim` - Remove whitespace
- `extract_pattern` - Extract using regex
- `replace` - Find and replace
- `split` - Split on delimiter
- `length` - String length
- `contains` - Check if contains substring
- `starts_with` / `ends_with` - Prefix/suffix checks

### Categorical Transforms
- `one_hot_encode` - Create binary columns
- `label_encode` - Convert to numeric codes
- `frequency_encode` - Encode by frequency
- `target_encode` - Encode by target mean
- `group_rare` - Group infrequent categories
- `map_values` - Custom value mapping

### Smart/ML Transforms
- `detect_outliers` - Flag outliers (IQR or Z-score)
- `fuzzy_match` - Group similar strings
- `extract_entities` - Extract entities (email, phone, URL)
- `sentiment` - Text sentiment analysis
- `rolling_aggregate` - Time-series rolling windows

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run Server
```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

### Test API
```bash
# Health check
curl http://localhost:8000/health

# Upload file
curl -X POST http://localhost:8000/upload \
  -F "file=@sales_data.csv"

# Get transforms
curl http://localhost:8000/transforms
```

---

## 📦 Dependencies

See `requirements.txt`:
- FastAPI
- pandas
- numpy
- scipy
- statsmodels
- scikit-learn
- python-multipart
- openpyxl

---

## 🎓 API Philosophy

**Two Data Versions:**
1. **Staging** - Raw uploaded data, untouched
2. **Curated** - Cleaned + transformed data for analysis

**Session-Based:**
- Each upload creates a session_id
- Sessions persist in memory (24hr TTL)
- Transforms can create new sessions or modify existing

**Transform Chains:**
- Combine multiple transforms sequentially
- Each transform's output feeds the next
- Great for complex data prep pipelines

**Power Query Parity:**
- All major Power Query operations supported
- Plus Python-specific features (ML, regex, etc.)
- RESTful API instead of GUI

---

## 🔐 Best Practices

1. **Preview Before Applying**: Always use `/transform/preview` first
2. **Use Suggestions**: Let AI recommend useful transforms
3. **Create New Sessions for Major Changes**: Keep original data intact
4. **Batch Transforms**: Apply multiple transforms in one call for efficiency
5. **Export Regularly**: Export transformed data for backups

---

## 📞 Support

For issues or questions:
- Check `/health` endpoint status
- Review `/transforms` catalog
- Use `/suggest/{column}` for guidance
- Check transform preview before applying

---

**Version:** 2.0.0  
**Last Updated:** December 2024  
**License:** MIT
