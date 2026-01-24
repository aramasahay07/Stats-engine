"""
Shared utilities for AI Analyst agents.

Provides common functionality like JSON serialization of numpy types,
column role detection, and statistical test selection helpers.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


def json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy/scipy/statsmodels outputs to JSON-safe native types.

    Handles:
    - numpy scalars (int64, float64, bool_, etc.)
    - numpy arrays -> lists
    - pandas Series/DataFrame -> dict/list
    - scipy sparse matrices -> dict representation
    - nested dicts and lists
    - NaN/Inf -> None
    """
    if obj is None:
        return None

    # Handle numpy types
    if hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]

    if hasattr(np, 'integer') and isinstance(obj, np.integer):
        return int(obj)

    if hasattr(np, 'floating') and isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    if hasattr(np, 'bool_') and isinstance(obj, np.bool_):
        return bool(obj)

    if hasattr(np, 'str_') and isinstance(obj, np.str_):
        return str(obj)

    # Handle pandas types if available
    try:
        import pandas as pd
        if isinstance(obj, pd.Series):
            return json_safe(obj.to_dict())
        if isinstance(obj, pd.DataFrame):
            return json_safe(obj.to_dict(orient='records'))
        if pd.isna(obj):
            return None
    except ImportError:
        pass

    # Handle native Python types
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (int, str, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]

    if isinstance(obj, set):
        return [json_safe(x) for x in obj]

    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return None

    # Fallback: try to convert to string
    try:
        return str(obj)
    except Exception:
        return None


def detect_column_role(
    dtype: str,
    sample_values: Optional[List[Any]] = None,
    unique_count: Optional[int] = None,
    n_rows: Optional[int] = None,
) -> str:
    """
    Detect the role of a column based on its dtype and characteristics.

    Returns one of: "numeric", "datetime", "categorical", "text", "unknown"
    """
    dtype_lower = dtype.lower()

    # Datetime types
    if any(t in dtype_lower for t in ['date', 'time', 'timestamp']):
        return "datetime"

    # Numeric types
    if any(t in dtype_lower for t in ['int', 'float', 'double', 'decimal', 'numeric', 'real', 'bigint', 'smallint', 'tinyint']):
        return "numeric"

    # Boolean
    if 'bool' in dtype_lower:
        return "categorical"

    # String types - determine if categorical or text
    if any(t in dtype_lower for t in ['varchar', 'char', 'text', 'string', 'str', 'object']):
        # If unique ratio is low, it's categorical
        if unique_count is not None and n_rows is not None and n_rows > 0:
            unique_ratio = unique_count / n_rows
            if unique_ratio < 0.5 or unique_count < 50:
                return "categorical"
            else:
                return "text"
        # Check sample values length
        if sample_values:
            avg_len = sum(len(str(v)) for v in sample_values if v is not None) / max(len(sample_values), 1)
            if avg_len > 100:
                return "text"
            return "categorical"
        return "categorical"

    return "unknown"


def infer_analysis_from_question(
    question: str,
    columns: List[Dict[str, Any]],
    selected_columns: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Infer the appropriate statistical analysis from a natural language question.

    Returns:
        (analysis_slug, params, reasoning)
    """
    q = question.lower()
    reasoning = []
    params: Dict[str, Any] = {}

    # Get column info by role
    numeric_cols = [c['name'] for c in columns if c.get('role') == 'numeric']
    categorical_cols = [c['name'] for c in columns if c.get('role') == 'categorical']
    datetime_cols = [c['name'] for c in columns if c.get('role') == 'datetime']

    # Apply selected columns if provided
    x_col = selected_columns.get('x') if selected_columns else None
    y_col = selected_columns.get('y') if selected_columns else None
    group_col = selected_columns.get('group') if selected_columns else None
    time_col = selected_columns.get('time') if selected_columns else None

    # Time series / trend analysis
    if any(kw in q for kw in ['trend', 'over time', 'time series', 'temporal', 'forecast', 'seasonal']):
        reasoning.append("Question mentions time-related analysis")
        if time_col or datetime_cols:
            time_column = time_col or datetime_cols[0]
            measure_column = y_col or (numeric_cols[0] if numeric_cols else None)
            if measure_column:
                params = {"time_column": time_column, "measure_column": measure_column}
                return "moving-average", params, reasoning + ["Using moving average for trend analysis"]
        reasoning.append("No datetime column found for time series analysis")

    # Correlation / relationship
    if any(kw in q for kw in ['correlat', 'relationship', 'association', 'related']):
        reasoning.append("Question asks about correlation or relationship")
        if x_col and y_col:
            params = {"x": x_col, "y": y_col}
            return "pearson-correlation", params, reasoning + ["Using Pearson correlation for relationship analysis"]
        if len(numeric_cols) >= 2:
            params = {"x": numeric_cols[0], "y": numeric_cols[1]}
            return "pearson-correlation", params, reasoning + [f"Using first two numeric columns: {numeric_cols[0]}, {numeric_cols[1]}"]
        reasoning.append("Need at least 2 numeric columns for correlation")

    # Regression / prediction
    if any(kw in q for kw in ['predict', 'regression', 'impact', 'effect on', 'influence']):
        reasoning.append("Question asks about prediction or impact")
        if x_col and y_col:
            params = {"x": x_col, "y": y_col}
            return "simple-linear-regression", params, reasoning + ["Using simple linear regression"]
        if len(numeric_cols) >= 2:
            params = {"x": numeric_cols[0], "y": numeric_cols[1]}
            return "simple-linear-regression", params, reasoning + [f"Using first numeric as predictor, second as outcome"]

    # Comparison between groups
    if any(kw in q for kw in ['compar', 'differ', 'versus', 'vs', 'between groups', 'significant']):
        reasoning.append("Question asks about comparison or difference")

        if group_col and (y_col or numeric_cols):
            measure = y_col or numeric_cols[0]
            # Check number of groups
            params = {"measure_column": measure, "group_column": group_col}

            # If we can detect exactly 2 groups, use t-test
            if any(kw in q for kw in ['two group', '2 group', 'two-group', 'between two']):
                return "two-sample-t-test", params, reasoning + ["Two-group comparison -> t-test"]

            # Default to ANOVA for multiple groups
            return "anova-one-way", params, reasoning + ["Multi-group comparison -> ANOVA"]

        # If we have categorical and numeric, infer grouping
        if categorical_cols and numeric_cols:
            params = {"measure_column": numeric_cols[0], "group_column": categorical_cols[0]}
            return "anova-one-way", params, reasoning + ["Found categorical and numeric columns -> ANOVA"]

    # Chi-square / categorical analysis
    if any(kw in q for kw in ['chi-square', 'chi square', 'categorical', 'contingency', 'independence', 'counts']):
        reasoning.append("Question mentions chi-square or categorical analysis")
        if x_col and y_col:
            params = {"x": x_col, "y": y_col}
            return "chi-square-test", params, reasoning + ["Using chi-square test for independence"]
        if len(categorical_cols) >= 2:
            params = {"x": categorical_cols[0], "y": categorical_cols[1]}
            return "chi-square-test", params, reasoning + [f"Using categorical columns: {categorical_cols[0]}, {categorical_cols[1]}"]

    # Distribution / normality
    if any(kw in q for kw in ['distribution', 'normal', 'skew', 'shape', 'histogram']):
        reasoning.append("Question asks about distribution")
        col = x_col or (numeric_cols[0] if numeric_cols else None)
        if col:
            params = {"column": col}
            return "normality-test", params, reasoning + [f"Testing normality of {col}"]

    # Descriptive statistics (default for simple questions)
    if any(kw in q for kw in ['average', 'mean', 'median', 'summary', 'describe', 'statistics', 'stats']):
        reasoning.append("Question asks for descriptive statistics")
        col = x_col or (numeric_cols[0] if numeric_cols else None)
        if col:
            params = {"column": col}
            return "mean", params, reasoning + [f"Computing descriptive statistics for {col}"]

    # Variance / spread
    if any(kw in q for kw in ['variance', 'spread', 'variability', 'std', 'standard deviation']):
        reasoning.append("Question asks about variance/spread")
        col = x_col or (numeric_cols[0] if numeric_cols else None)
        if col:
            params = {"column": col}
            return "variance", params, reasoning + [f"Computing variance for {col}"]

    # Default: descriptive statistics
    reasoning.append("No specific analysis detected, defaulting to descriptive statistics")
    if numeric_cols:
        params = {"column": numeric_cols[0]}
        return "mean", params, reasoning + ["Using mean as default descriptive measure"]

    return "descriptives", params, reasoning + ["Unable to determine specific analysis"]


def get_test_alternatives(analysis_slug: str) -> List[Tuple[str, str]]:
    """
    Get alternative tests that could be used instead of the selected one.

    Returns list of (test_slug, reason_not_chosen) tuples.
    """
    alternatives: Dict[str, List[Tuple[str, str]]] = {
        "two-sample-t-test": [
            ("mann-whitney-u", "Non-parametric alternative if normality assumption violated"),
            ("welch-t-test", "Does not assume equal variances"),
            ("permutation-test", "Distribution-free but computationally intensive"),
        ],
        "anova-one-way": [
            ("kruskal-wallis", "Non-parametric alternative if normality violated"),
            ("welch-anova", "Does not assume equal variances"),
            ("two-sample-t-test", "Only for exactly 2 groups"),
        ],
        "pearson-correlation": [
            ("spearman-correlation", "Non-parametric, works with monotonic relationships"),
            ("kendall-tau", "More robust to outliers, smaller samples"),
            ("simple-linear-regression", "When interested in prediction, not just association"),
        ],
        "chi-square-test": [
            ("fisher-exact-test", "Better for small samples (n < 20)"),
            ("g-test", "Alternative likelihood-ratio test"),
        ],
        "simple-linear-regression": [
            ("pearson-correlation", "When only interested in relationship strength"),
            ("polynomial-regression", "If relationship appears non-linear"),
            ("robust-regression", "If outliers are present"),
        ],
        "normality-test": [
            ("shapiro-wilk", "More powerful for small samples"),
            ("kolmogorov-smirnov", "Better for larger samples"),
            ("anderson-darling", "More sensitive to tails"),
        ],
    }
    return alternatives.get(analysis_slug, [])


def format_p_value(p: Optional[float]) -> str:
    """Format p-value for display."""
    if p is None:
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def interpret_p_value(p: Optional[float], alpha: float = 0.05) -> str:
    """Get interpretation of p-value."""
    if p is None:
        return "Unable to compute p-value"
    if p < alpha:
        return f"Statistically significant (p = {format_p_value(p)}, α = {alpha})"
    return f"Not statistically significant (p = {format_p_value(p)}, α = {alpha})"


def interpret_effect_size(
    effect: Optional[float],
    effect_type: str = "cohens_d",
) -> str:
    """Interpret effect size magnitude."""
    if effect is None:
        return "Effect size not computed"

    abs_effect = abs(effect)

    if effect_type in ["cohens_d", "d"]:
        if abs_effect < 0.2:
            return f"Negligible effect (d = {effect:.2f})"
        elif abs_effect < 0.5:
            return f"Small effect (d = {effect:.2f})"
        elif abs_effect < 0.8:
            return f"Medium effect (d = {effect:.2f})"
        else:
            return f"Large effect (d = {effect:.2f})"

    elif effect_type in ["r", "correlation"]:
        if abs_effect < 0.1:
            return f"Negligible correlation (r = {effect:.2f})"
        elif abs_effect < 0.3:
            return f"Weak correlation (r = {effect:.2f})"
        elif abs_effect < 0.5:
            return f"Moderate correlation (r = {effect:.2f})"
        elif abs_effect < 0.7:
            return f"Strong correlation (r = {effect:.2f})"
        else:
            return f"Very strong correlation (r = {effect:.2f})"

    elif effect_type in ["r_squared", "r2"]:
        pct = effect * 100
        if effect < 0.01:
            return f"Negligible variance explained (R² = {effect:.3f}, {pct:.1f}%)"
        elif effect < 0.09:
            return f"Small variance explained (R² = {effect:.3f}, {pct:.1f}%)"
        elif effect < 0.25:
            return f"Medium variance explained (R² = {effect:.3f}, {pct:.1f}%)"
        else:
            return f"Large variance explained (R² = {effect:.3f}, {pct:.1f}%)"

    elif effect_type in ["eta_squared", "eta2"]:
        if abs_effect < 0.01:
            return f"Negligible effect (η² = {effect:.3f})"
        elif abs_effect < 0.06:
            return f"Small effect (η² = {effect:.3f})"
        elif abs_effect < 0.14:
            return f"Medium effect (η² = {effect:.3f})"
        else:
            return f"Large effect (η² = {effect:.3f})"

    return f"Effect size = {effect:.3f}"


def get_required_columns_for_analysis(analysis_slug: str) -> Dict[str, str]:
    """
    Get required column parameters for a given analysis type.

    Returns dict of {param_name: description}
    """
    requirements: Dict[str, Dict[str, str]] = {
        "two-sample-t-test": {
            "measure_column": "Numeric column to compare",
            "group_column": "Categorical column defining 2 groups",
        },
        "anova-one-way": {
            "measure_column": "Numeric column to compare",
            "group_column": "Categorical column defining groups",
        },
        "pearson-correlation": {
            "x": "First numeric column",
            "y": "Second numeric column",
        },
        "spearman-correlation": {
            "x": "First numeric column",
            "y": "Second numeric column",
        },
        "chi-square-test": {
            "x": "First categorical column",
            "y": "Second categorical column",
        },
        "simple-linear-regression": {
            "x": "Predictor (independent) column",
            "y": "Outcome (dependent) column",
        },
        "mean": {
            "column": "Numeric column",
        },
        "variance": {
            "column": "Numeric column",
        },
        "normality-test": {
            "column": "Numeric column to test",
        },
        "moving-average": {
            "time_column": "Datetime column",
            "measure_column": "Numeric column to analyze",
        },
    }
    return requirements.get(analysis_slug, {})
