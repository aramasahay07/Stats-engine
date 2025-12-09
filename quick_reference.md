# Transform Quick Reference

## 📅 Date/Time Transforms

| Transform | Input | Output | Example |
|-----------|-------|--------|---------|
| `month` | datetime | "January" | `{"type": "month", "params": {"format": "name"}}` |
| `year` | datetime | 2024 | `{"type": "year"}` |
| `quarter` | datetime | "Q1" | `{"type": "quarter"}` |
| `weekday` | datetime | "Monday" | `{"type": "weekday", "params": {"format": "name"}}` |
| `month_year` | datetime | "Jan 2024" | `{"type": "month_year", "params": {"format": "short"}}` |
| `age_from_date` | datetime | 42 | `{"type": "age_from_date", "params": {"reference_date": "today"}}` |

## 🔢 Numeric Transforms

| Transform | Input | Output | Example |
|-----------|-------|--------|---------|
| `bucket` | numeric | "0-10" | `{"type": "bucket", "params": {"bins": [0,10,20,50,100]}}` |
| `percentile_bucket` | numeric | "0-25%" | `{"type": "percentile_bucket", "params": {"quantiles": [0.25,0.5,0.75,1]}}` |
| `round` | numeric | 42.00 | `{"type": "round", "params": {"precision": 2}}` |
| `normalize` | numeric | 0.75 | `{"type": "normalize", "params": {"method": "minmax"}}` |
| `log` | numeric | 3.21 | `{"type": "log", "params": {"base": 10}}` |
| `detect_outliers` | numeric | "Outlier" | `{"type": "detect_outliers", "params": {"method": "iqr"}}` |
| `rolling` | numeric | 42.5 | `{"type": "rolling", "params": {"window": 7, "function": "mean"}}` |

## 📝 Text Transforms

| Transform | Input | Output | Example |
|-----------|-------|--------|---------|
| `lowercase` | string | "hello world" | `{"type": "lowercase"}` |
| `uppercase` | string | "HELLO WORLD" | `{"type": "uppercase"}` |
| `trim` | string | "hello" | `{"type": "trim"}` |
| `length` | string | 11 | `{"type": "length"}` |
| `extract_entity` | string | "user@email.com" | `{"type": "extract_entity", "params": {"entity_type": "email"}}` |
| `word_count` | string | 5 | `{"type": "word_count"}` |
| `replace` | string | "hello there" | `{"type": "replace", "params": {"find": "world", "replace": "there"}}` |

## 🏷️ Categorical Transforms

| Transform | Input | Output | Example |
|-----------|-------|--------|---------|
| `remap` | any | "New Value" | `{"type": "remap", "params": {"mapping": {"old": "new"}}}` |
| `top_n` | categorical | "Other" | `{"type": "top_n", "params": {"n": 10}}` |
| `null_fill` | any | "Unknown" | `{"type": "null_fill", "params": {"value": "Unknown"}}` |
| `null_fill_smart` | numeric | 42.5 | `{"type": "null_fill_smart", "params": {"method": "median"}}` |
| `binary` | any | "Yes" | `{"type": "binary", "params": {"condition": "> 50"}}` |
| `encode` | categorical | 3 | `{"type": "encode", "params": {"method": "label"}}` |

## 🧠 Smart Transforms

| Transform | Description | Example |
|-----------|-------------|---------|
| `bucket_smart` | Intelligent binning | `{"type": "bucket_smart", "params": {"method": "outlier_aware"}}` |
| `cast_smart` | Type coercion | `{"type": "cast_smart", "params": {"target_type": "numeric"}}` |
| `seasonality` | Extract seasons | `{"type": "seasonality", "params": {"feature": "season"}}` |
| `standardize` | Clean messy text | `{"type": "standardize", "params": {"method": "boolean"}}` |

## 🔗 Chained Transform Example

```json
{
  "transforms": {
    "admission_date": {
      "type": "month",
      "params": {"format": "name"}
    }
  }
}
```

## 📊 Common Use Cases

### Monthly Trend Analysis
```json
{
  "operation": "aggregate",
  "group_by": ["date_column"],
  "transforms": {
    "date_column": {"type": "month", "params": {"format": "name"}}
  },
  "aggregations": {"count": "*:count"},
  "sort": {"column": "date_column_month", "order": "chronological"}
}
```

### Age Groups
```json
{
  "transforms": {
    "age": {
      "type": "bucket",
      "params": {
        "bins": [0, 18, 35, 50, 65, 100],
        "labels": ["Child", "Young Adult", "Adult", "Senior", "Elderly"]
      }
    }
  }
}
```

### Outlier Detection
```json
{
  "transforms": {
    "cost": {
      "type": "detect_outliers",
      "params": {"method": "isolation_forest", "contamination": 0.1}
    }
  }
}
```

### Smart Null Filling
```json
{
  "transforms": {
    "revenue": {
      "type": "null_fill_smart",
      "params": {"method": "median"}
    }
  }
}
```

## 🎯 Parameter Options

### Date Format Options
- `"name"` → "January", "February"
- `"short"` → "Jan", "Feb"
- `"number"` → 1, 2, 3

### Bucketing Methods
- `"equal_frequency"` → Same count per bin
- `"equal_width"` → Same range per bin
- `"outlier_aware"` → Excludes outliers
- `"kmeans"` → Natural clustering

### Null Fill Methods
- `"mean"` → Average value
- `"median"` → Middle value
- `"mode"` → Most common
- `"forward_fill"` → Copy previous
- `"interpolate"` → Linear interpolation

### Outlier Detection
- `"iqr"` → Interquartile range (1.5x threshold)
- `"zscore"` → Standard deviations (3σ)
- `"isolation_forest"` → ML-based

## 💡 Tips

1. **Always check suggestions first**: `GET /suggest-transforms/{session_id}?column=your_column`
2. **Start simple**: Use basic transforms before chaining
3. **Test with small data**: Verify output before full dataset
4. **Cache is automatic**: Repeated queries are fast
5. **Check chronological sort**: Date-derived columns need `"order": "chronological"`

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "Transform failed" | Check column type matches transform input |
| "Column not found" | Verify column name spelling |
| Months in wrong order | Use `"order": "chronological"` in sort |
| Null values in output | Use `null_fill` or `null_fill_smart` first |
| "Log requires positive values" | Use `absolute` transform first |

## 📖 Full API Docs

Visit `http://localhost:8000/docs` when server is running for interactive API documentation.
