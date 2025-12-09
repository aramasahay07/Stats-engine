"""
Intelligent type inference and detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import re


def infer_column_role(series: pd.Series) -> str:
    """
    Classify column into a role with intelligent detection
    Returns: "numeric", "categorical", "datetime", "text"
    """
    # Already typed as datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Numeric types
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's actually categorical (low cardinality)
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if unique_ratio < 0.05 and series.nunique() < 20:
            return "categorical"
        return "numeric"
    
    # Try to parse as datetime
    if _could_be_datetime(series):
        return "datetime"
    
    # String/object types
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        unique_count = series.nunique(dropna=True)
        total_count = len(series)
        unique_ratio = unique_count / max(total_count, 1)
        
        # Very low cardinality suggests categorical
        if unique_ratio < 0.05 or unique_count < 20:
            return "categorical"
        
        # Check if it could be numeric
        if _could_be_numeric(series):
            return "numeric"
        
        # Check average length to distinguish text from categorical
        avg_length = series.dropna().astype(str).str.len().mean()
        if avg_length > 50:
            return "text"
        
        return "categorical"
    
    # Boolean
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    
    # Default to text
    return "text"


def _could_be_datetime(series: pd.Series) -> bool:
    """Check if string series could be parsed as datetime"""
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
    
    # Common date patterns
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD-MM-YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
        r'\d{1,2}[/-]\w{3}[/-]\d{4}',      # DD-Mon-YYYY
        r'\w{3}\s+\d{1,2},?\s+\d{4}',      # Mon DD, YYYY
    ]
    
    sample_str = sample.astype(str)
    
    for pattern in date_patterns:
        matches = sample_str.str.contains(pattern, regex=True, na=False)
        if matches.sum() / len(sample) > 0.8:  # 80% match threshold
            return True
    
    return False


def _could_be_numeric(series: pd.Series) -> bool:
    """Check if string series could be parsed as numeric"""
    sample = series.dropna().head(100).astype(str)
    if len(sample) == 0:
        return False
    
    # Remove common formatting
    cleaned = sample.str.replace(',', '').str.replace('$', '').str.strip()
    
    # Try to convert to numeric
    try:
        converted = pd.to_numeric(cleaned, errors='coerce')
        success_rate = converted.notna().sum() / len(cleaned)
        return success_rate > 0.8  # 80% success threshold
    except Exception:
        return False


def detect_date_format(series: pd.Series) -> List[str]:
    """
    Detect likely date format(s) in a string series
    Returns list of format strings (e.g., ['%m/%d/%Y', '%Y-%m-%d'])
    """
    sample = series.dropna().head(100).astype(str)
    if len(sample) == 0:
        return []
    
    # Common formats to try
    formats = [
        '%m/%d/%y %H:%M',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y',
        '%m-%d-%Y',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d-%b-%Y',
        '%d/%m/%Y',
        '%d/%m/%y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%Y-%m-%d %H:%M:%S',
    ]
    
    detected = []
    
    for fmt in formats:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
            success_rate = parsed.notna().sum() / len(sample)
            if success_rate > 0.8:
                detected.append(fmt)
        except Exception:
            continue
    
    return detected


def smart_parse_datetime(series: pd.Series) -> pd.Series:
    """
    Intelligently parse datetime with multiple format attempts
    """
    # Try with infer_datetime_format first
    try:
        result = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        if result.notna().sum() / len(series) > 0.5:  # 50% success
            return result
    except Exception:
        pass
    
    # Try detected formats
    formats = detect_date_format(series)
    for fmt in formats:
        try:
            result = pd.to_datetime(series, format=fmt, errors='coerce')
            if result.notna().sum() / len(series) > 0.8:
                return result
        except Exception:
            continue
    
    # Last resort: let pandas infer
    return pd.to_datetime(series, errors='coerce')


def detect_encoding(file_bytes: bytes) -> str:
    """
    Detect file encoding
    """
    try:
        import chardet
        result = chardet.detect(file_bytes)
        return result['encoding'] or 'utf-8'
    except Exception:
        return 'utf-8'


def analyze_column_patterns(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze patterns in column data
    """
    analysis = {
        "has_nulls": series.isna().any(),
        "null_percentage": float(series.isna().sum() / len(series) * 100),
        "unique_count": int(series.nunique()),
        "cardinality": "high" if series.nunique() / len(series) > 0.5 else "low",
    }
    
    if pd.api.types.is_numeric_dtype(series):
        analysis.update({
            "has_negatives": (series < 0).any(),
            "has_zeros": (series == 0).any(),
            "range": (float(series.min()), float(series.max())),
            "potential_outliers": int((np.abs((series - series.mean()) / series.std()) > 3).sum())
        })
    
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        sample_str = series.dropna().astype(str)
        if len(sample_str) > 0:
            analysis.update({
                "avg_length": float(sample_str.str.len().mean()),
                "max_length": int(sample_str.str.len().max()),
                "has_whitespace": any(s != s.strip() for s in sample_str.head(100)),
                "has_special_chars": any(re.search(r'[^a-zA-Z0-9\s]', s) for s in sample_str.head(100))
            })
    
    return analysis


def suggest_column_transforms(series: pd.Series) -> List[str]:
    """
    Suggest appropriate transforms based on column analysis
    """
    suggestions = []
    role = infer_column_role(series)
    patterns = analyze_column_patterns(series)
    
    if role == "datetime":
        if series.nunique() > 50:
            suggestions.extend(["month", "year", "quarter", "weekday"])
    
    elif role == "numeric":
        if patterns.get("cardinality") == "high":
            suggestions.extend(["bucket", "percentile_bucket"])
        if patterns.get("has_negatives"):
            suggestions.append("absolute")
        if patterns.get("potential_outliers", 0) > 0:
            suggestions.append("detect_outliers")
    
    elif role == "categorical":
        if patterns.get("unique_count", 0) > 10:
            suggestions.append("top_n")
        if patterns.get("has_whitespace"):
            suggestions.append("trim")
    
    elif role == "text":
        if patterns.get("has_whitespace"):
            suggestions.extend(["trim", "lowercase"])
        suggestions.append("length")
    
    if patterns.get("has_nulls"):
        suggestions.insert(0, "null_fill_smart")
    
    return suggestions
