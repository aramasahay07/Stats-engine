# AI Data Lab - Advanced Data Transformation Engine

A comprehensive Python/FastAPI backend with intelligent data transformations, statistical analysis, and ML-enhanced features.

## 🚀 Features

### Core Capabilities
- **Smart Data Upload**: Automatic type detection and intelligent date parsing
- **60+ Transform Types**: Date/time, numeric, text, categorical, and ML-enhanced transforms
- **Chained Transformations**: Apply multiple transforms in sequence
- **Transform Discovery**: Auto-suggest appropriate transforms for any column
- **Caching System**: Efficient transform caching for repeated operations
- **Statistical Analysis**: Correlation, t-tests, ANOVA, and regression

### Transform Categories

#### 📅 Date/Time Transforms
- `month`, `year`, `quarter`, `weekday`, `week`, `day`, `hour`
- `month_year`, `fiscal_quarter`, `date_only`
- `age_from_date`, `time_features`, `seasonality`

#### 🔢 Numeric Transforms
- `bucket`, `percentile_bucket`, `bucket_distribution`, `bucket_smart`
- `round`, `normalize`, `log`, `absolute`, `sign`
- `rolling`, `difference`, `percent_change`
- `detect_outliers`, `binning_adaptive`

#### 📝 Text Transforms
- `lowercase`, `uppercase`, `title_case`, `trim`, `slugify`
- `length`, `word_count`, `first_word`
- `extract`, `extract_entity` (email, phone, URL, zip, currency)
- `replace`, `contains`, `standardize`

#### 🏷️ Categorical Transforms
- `remap`, `top_n`, `group_rare`, `binary`
- `null_fill`, `null_fill_smart` (mean, median, mode, forward_fill, interpolate)
- `encode` (label, frequency, ordinal), `onehot`
- `conditional` (SQL-like CASE WHEN)

#### 🧠 Smart/ML Transforms
- `cast_smart`: Intelligent type coercion with validation
- `outlier_detection`: IQR, z-score, or Isolation Forest
- `fuzzy_match`: Standardize using string similarity
- `window_aggregation`: Rolling calculations with grouping

## 📁 Project Structure

```
backend/
├── main.py                          # FastAPI app with all endpoints
├── models.py                        # Pydantic schemas
├── session_store.py                 # Session & cache management
├── requirements.txt                 # Python dependencies
├── transformers/
│   ├── __init__.py
│   ├── base.py                      # Base transformer class
│   ├── datetime_transforms.py      # Date/time transforms
│   ├── numeric_transforms.py       # Numeric transforms
│   ├── text_transforms.py          # Text transforms
│   ├── categorical_transforms.py   # Categorical transforms
│   ├── smart_transforms.py         # AI/ML transforms
│   ├── pipeline.py                 # Chain execution
│   └── registry.py                 # Transform discovery
└── utils/
    ├── __init__.py
    ├── type_inference.py           # Smart type detection
    ├── date_parser.py              # Intelligent date parsing
    └── validators.py               # Input validation
```

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Server
```bash
python main.py
```

Server will start at `http://localhost:8000`

## 📚 API Documentation

### Upload Data
```bash
POST /upload
Content-Type: multipart/form-data

Response:
{
  "session_id": "uuid",
  "n_rows": 1000,
  "n_cols": 15,
  "columns": [...],
  "descriptives": [...]
}
```

### Query with Transforms
```bash
POST /query/{session_id}
Content-Type: application/json

{
  "operation": "aggregate",
  "group_by": ["admission_date"],
  "transforms": {
    "admission_date": {
      "type": "month",
      "params": {"format": "name"}
    }
  },
  "aggregations": {
    "patient_count": "*:count"
  },
  "sort": {
    "column": "admission_date_month",
    "order": "chronological"
  }
}
```

### Discover Transforms
```bash
GET /transforms

Response:
{
  "transforms": {
    "month": {
      "input_types": ["datetime"],
      "output_type": "categorical",
      "description": "Extract month from datetime"
    },
    ...
  }
}
```

### Get Transform Suggestions
```bash
GET /suggest-transforms/{session_id}?column=admission_date

Response:
{
  "column": "admission_date",
  "detected_type": "datetime",
  "suggested_transforms": [
    {
      "transform": "month",
      "usefulness_score": 0.95,
      "reason": "High cardinality date - monthly aggregation recommended",
      "preview": ["January", "February", "March"]
    }
  ]
}
```

## 🎯 Usage Examples

### Example 1: Monthly Patient Trend
```python
# Request
{
  "operation": "aggregate",
  "group_by": ["admission_date"],
  "transforms": {
    "admission_date": {
      "type": "month",
      "params": {"format": "name"}
    }
  },
  "aggregations": {
    "patient_count": "*:count"
  },
  "sort": {
    "column": "admission_date_month",
    "order": "chronological"
  }
}

# Result: Monthly patient counts with proper chronological ordering
```

### Example 2: Age Group Distribution
```python
{
  "operation": "aggregate",
  "group_by": ["age"],
  "transforms": {
    "age": {
      "type": "bucket",
      "params": {
        "bins": [0, 18, 35, 50, 65, 100],
        "labels": ["Child", "Young Adult", "Adult", "Senior", "Elderly"]
      }
    }
  },
  "aggregations": {
    "count": "*:count"
  }
}
```

### Example 3: Chained Transformations
```python
{
  "transforms": {
    "lengthofstay_days": {
      "type": "conditional",
      "params": {
        "conditions": [
          {"when": "lengthofstay_days < 3", "then": "Short Stay"},
          {"when": "lengthofstay_days < 7", "then": "Medium Stay"},
          {"when": "lengthofstay_days >= 7", "then": "Long Stay"}
        ]
      }
    }
  }
}
```

### Example 4: Smart Outlier Detection
```python
{
  "transforms": {
    "cost": {
      "type": "detect_outliers",
      "params": {
        "method": "isolation_forest",
        "contamination": 0.1
      }
    }
  }
}
```

## 🔧 Advanced Features

### Transform Caching
Transforms are automatically cached per session to improve performance:
```python
# First request: Computes transform
# Subsequent requests: Uses cached result
```

### Smart Type Inference
The system automatically detects:
- Date formats (MM/DD/YY, YYYY-MM-DD, etc.)
- Numeric columns with currency symbols
- Categorical vs text based on cardinality
- Boolean-like values (Yes/No, T/F, 1/0)

### Distribution-Aware Binning
```python
{
  "type": "bucket_distribution",
  "params": {
    "method": "outlier_aware",  # Excludes outliers when calculating bins
    "n_bins": 5,
    "outlier_threshold": 1.5
  }
}
```

### Null Handling Strategies
```python
{
  "type": "null_fill_smart",
  "params": {
    "method": "group_median",  # Fill with group median
    "group_by": "department"
  }
}
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## 📊 Performance

- **Transform Execution**: < 500ms for 100K rows
- **Caching**: 10x faster for repeated operations
- **Memory**: < 2x original dataframe size
- **Concurrent Transforms**: Up to 5 per query

## 🔐 Security

- Session-based isolation
- Input validation on all endpoints
- SQL injection protection (no raw SQL)
- File size limits on upload

## 🚧 Roadmap

- [ ] Add more ML transforms (PCA, feature engineering)
- [ ] Support for time series analysis
- [ ] Real-time streaming transforms
- [ ] Multi-table joins
- [ ] Export transformed data
- [ ] Transform history and undo

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please submit PRs with:
1. New transformer classes in appropriate file
2. Registration in `registry.py`
3. Unit tests
4. Documentation updates

## 💬 Support

For issues or questions, please open a GitHub issue.

---

**Version**: 2.0.0  
**Last Updated**: December 2024
