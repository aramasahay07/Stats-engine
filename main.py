from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any
from uuid import uuid4
from io import BytesIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import logging

# ---------------------------------------------------
# Logging setup
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------
app = FastAPI(title="AI Data Lab Stats Engine", version="0.4.0")

# CORS so Lovable (browser frontend) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# In-memory "session store" with timestamps
# (For MVP only; wiped on restart / cold start)
# ---------------------------------------------------
SESSIONS: Dict[str, Tuple[pd.DataFrame, datetime]] = {}


# ---------------------------------------------------
# Pydantic models for API schemas
# ---------------------------------------------------
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    type: str          # ✅ What Lovable frontend expects (same as role)
    role: str          # "numeric" | "categorical" | "datetime" | "text"
    missing_pct: float


class DescriptiveStats(BaseModel):
    column: str
    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None


class TestResult(BaseModel):
    test_type: str          # "t-test", "anova"
    target: str
    group_col: str
    p_value: float
    statistic: float
    df: Optional[float] = None
    interpretation: str


class RegressionResult(BaseModel):
    target: str
    predictors: List[str]
    r_squared: float
    adj_r_squared: float
    coefficients: Dict[str, float]


class ProfileResponse(BaseModel):
    session_id: str
    n_rows: int
    n_cols: int
    columns: List[ColumnInfo]
    schema: List[ColumnInfo]
    descriptives: List[DescriptiveStats]
    missing_summary: Dict[str, float]  # ✅ Required by Lovable
    cleaning_suggestions: List[str]    # ✅ Required by Lovable


class UploadResponse(BaseModel):
    session_id: str
    n_rows: int
    n_cols: int
    columns: List[ColumnInfo]
    schema: List[ColumnInfo]
    descriptives: List[DescriptiveStats]
    missing_summary: Dict[str, float]  # ✅ Required by Lovable
    cleaning_suggestions: List[str]    # ✅ Required by Lovable
    sample_rows: List[Dict]


class CorrelationResponse(BaseModel):
    matrix: Dict[str, Dict[str, float]]


class AnalysisResponse(BaseModel):
    session_id: str
    correlation: Optional[CorrelationResponse] = None
    tests: List[TestResult] = []
    regression: Optional[RegressionResult] = None


class SessionStats(BaseModel):
    total_sessions: int
    memory_used_mb: float
    oldest_session_age_minutes: Optional[float]
    session_ids: List[str]


class SchemaResponse(BaseModel):
    session_id: str
    n_rows: int
    n_cols: int
    schema: List[ColumnInfo]
    missing_summary: Dict[str, float]


class SampleResponse(BaseModel):
    session_id: str
    sample_rows: List[Dict]
    total_rows: int
    returned_rows: int


class QueryMetric(BaseModel):
    column: str
    agg: str  # "count", "sum", "mean", "median", "min", "max", "std", "var", "nunique", "mode", "iqr", "mad", "skew", "kurt"
    alias: Optional[str] = None


class QueryFilter(BaseModel):
    column: str
    op: str  # "==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "is_null", "is_not_null", "between"
    value: Optional[Any] = None
    values: Optional[List[Any]] = None  # For "in", "not_in", "between"


class QueryOrderBy(BaseModel):
    column: str
    direction: str = "asc"  # "asc" or "desc"


class QueryRequest(BaseModel):
    operation: str  # "aggregate", "filter", "describe", "correlation", "percentile", "distinct", "crosstab", "rank", "binning", "sample"
    group_by: Optional[List[str]] = None
    metrics: Optional[List[QueryMetric]] = None
    filters: Optional[List[QueryFilter]] = None
    order_by: Optional[List[QueryOrderBy]] = None
    limit: Optional[int] = None
    options: Optional[Dict] = None


class QueryResponse(BaseModel):
    success: bool
    operation: str
    result: Dict
    execution_time_ms: float
    query_summary: str
    error: Optional[str] = None


# ---------------------------------------------------
# Session Management Functions
# ---------------------------------------------------
def _clean_old_sessions(max_age_hours: int = 24) -> int:
    """
    Remove sessions older than max_age_hours.
    Returns number of sessions cleaned.
    """
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    expired = [
        sid for sid, (_, timestamp) in SESSIONS.items() 
        if timestamp < cutoff
    ]
    for sid in expired:
        del SESSIONS[sid]
        logger.info(f"Cleaned expired session: {sid}")
    
    return len(expired)


def _get_session_dataframe(session_id: str) -> pd.DataFrame:
    """
    Retrieve dataframe from session store.
    Raises HTTPException if not found.
    """
    session_data = SESSIONS.get(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    df, _ = session_data
    return df


def _store_session(session_id: str, df: pd.DataFrame):
    """Store dataframe with current timestamp."""
    SESSIONS[session_id] = (df, datetime.now())


# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------
def _infer_role(series: pd.Series) -> str:
    """Classify column into a simple role for UI logic."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # heuristic: low cardinality → categorical
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    if unique_ratio < 0.2:
        return "categorical"
    return "text"


def _auto_detect_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert text columns that look like dates into datetime.
    Safe - returns original if conversion fails.
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # Only check text columns
            # Sample a few values to see if they look like dates
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
                
            # Check if values contain date-like patterns
            try:
                date_like = sample.astype(str).str.contains(
                    r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', 
                    regex=True,
                    na=False
                ).sum()
                
                # If >50% look like dates, try to convert
                if date_like / len(sample) > 0.5:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Converted column '{col}' to datetime")
                    except Exception as e:
                        logger.debug(f"Could not convert {col} to datetime: {e}")
            except Exception as e:
                logger.debug(f"Error checking {col} for dates: {e}")
    
    return df


def _load_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read CSV or Excel file into a pandas DataFrame, with safe fallbacks."""
    content = file.file.read()
    file.file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    buffer = BytesIO(content)
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".csv"):
            # Try default engine first
            try:
                df = pd.read_csv(buffer)
            except Exception:
                # Fallback: more tolerant parser
                buffer.seek(0)
                df = pd.read_csv(buffer, engine="python", on_bad_lines="skip")
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buffer)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload CSV or Excel.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse file as CSV/Excel: {e}",
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="File contained no rows.")
    return df


def _build_profile(df: pd.DataFrame, session_id: str) -> ProfileResponse:
    """Create profile summary for the UI."""
    cols: List[ColumnInfo] = []
    descriptives: List[DescriptiveStats] = []

    for col in df.columns:
        s = df[col]
        role = _infer_role(s)
        missing_pct = float(s.isna().mean() * 100)  # Keep as percentage for display

        cols.append(
            ColumnInfo(
                name=col,
                dtype=str(s.dtype),
                type=role,  # ✅ Map role to type for Lovable frontend
                role=role,
                missing_pct=round(missing_pct, 2),
            )
        )

        if role == "numeric":
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            descriptives.append(
                DescriptiveStats(
                    column=col,
                    count=int(desc.get("count", 0)),
                    mean=float(desc.get("mean", np.nan))
                    if not np.isnan(desc.get("mean", np.nan))
                    else None,
                    std=float(desc.get("std", np.nan))
                    if not np.isnan(desc.get("std", np.nan))
                    else None,
                    min=float(desc.get("min", np.nan))
                    if not np.isnan(desc.get("min", np.nan))
                    else None,
                    q25=float(desc.get("25%", np.nan))
                    if not np.isnan(desc.get("25%", np.nan))
                    else None,
                    median=float(desc.get("50%", np.nan))
                    if not np.isnan(desc.get("50%", np.nan))
                    else None,
                    q75=float(desc.get("75%", np.nan))
                    if not np.isnan(desc.get("75%", np.nan))
                    else None,
                    max=float(desc.get("max", np.nan))
                    if not np.isnan(desc.get("max", np.nan))
                    else None,
                )
            )

    # Build missing_summary as decimal (0-1) for Lovable frontend
    # Frontend multiplies by 100 to display as percentage
    missing_summary = {c.name: round(c.missing_pct / 100, 4) for c in cols}
    
    # Generate cleaning_suggestions
    cleaning_suggestions = _generate_cleaning_suggestions(df, cols)

    return ProfileResponse(
        session_id=session_id,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=cols,
        schema=cols,
        descriptives=descriptives,
        missing_summary=missing_summary,
        cleaning_suggestions=cleaning_suggestions,
    )


def _build_correlation(df: pd.DataFrame) -> Optional[CorrelationResponse]:
    """Build correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    corr_dict: Dict[str, Dict[str, float]] = {}
    for col in corr.columns:
        corr_dict[col] = {idx: float(val) for idx, val in corr[col].items()}
    return CorrelationResponse(matrix=corr_dict)


def _auto_tests(df: pd.DataFrame, profile: ProfileResponse) -> List[TestResult]:
    """Run simple t-test or ANOVA if data supports it."""
    tests: List[TestResult] = []

    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    cat_cols = [c.name for c in profile.columns if c.role == "categorical"]

    if not numeric_cols or not cat_cols:
        return tests

    target = numeric_cols[0]
    group_col = cat_cols[0]
    s_target = df[target]
    s_group = df[group_col].astype("category")

    # Drop rows with missing values in relevant columns
    data = pd.DataFrame({"target": s_target, "group": s_group}).dropna()
    if data["group"].nunique() < 2:
        return tests

    groups = [g["target"].values for _, g in data.groupby("group")]

    try:
        if len(groups) == 2:
            # two-sample t-test (unequal variance)
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            interpretation = (
                "Difference between the two groups is statistically significant."
                if p < 0.05
                else "No statistically significant difference between the two groups."
            )
            tests.append(
                TestResult(
                    test_type="t-test",
                    target=target,
                    group_col=group_col,
                    p_value=float(p),
                    statistic=float(stat),
                    df=None,
                    interpretation=interpretation,
                )
            )
        elif len(groups) > 2:
            # one-way ANOVA
            stat, p = stats.f_oneway(*groups)
            interpretation = (
                "At least one group mean differs significantly from the others."
                if p < 0.05
                else "No statistically significant difference among group means."
            )
            tests.append(
                TestResult(
                    test_type="anova",
                    target=target,
                    group_col=group_col,
                    p_value=float(p),
                    statistic=float(stat),
                    df=None,
                    interpretation=interpretation,
                )
            )
    except Exception:
        return tests

    return tests


def _auto_regression(df: pd.DataFrame, profile: ProfileResponse) -> Optional[RegressionResult]:
    """Fit a simple OLS regression: first numeric as y, others as X."""
    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    if len(numeric_cols) < 2:
        return None

    target = numeric_cols[0]
    predictors = numeric_cols[1:]

    data = df[numeric_cols].dropna()
    if len(data) < 10:
        return None

    y = data[target]
    X = data[predictors]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    coeffs = {name: float(val) for name, val in model.params.items()}
    return RegressionResult(
        target=target,
        predictors=predictors,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        coefficients=coeffs,
    )


def _prepare_sample_rows(df: pd.DataFrame, max_rows: int = 200) -> List[Dict]:
    """
    Prepare sample rows for JSON serialization.
    Handles NaN, datetime, and other non-JSON-serializable types.
    
    Args:
        df: DataFrame to sample
        max_rows: Maximum rows to return (capped at 500 for safety)
    """
    # Safety cap
    max_rows = min(max_rows, 500)
    
    try:
        # Take sample
        sample_df = df.head(max_rows).copy()
        
        # Replace NaN with None for JSON serialization
        sample_df = sample_df.where(pd.notnull(sample_df), None)
        
        # Convert datetime columns to ISO format strings
        for col in sample_df.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                sample_df[col] = sample_df[col].apply(
                    lambda x: x.isoformat() if pd.notnull(x) else None
                )
        
        # Convert to records
        return sample_df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error preparing sample rows: {e}")
        return []


def _generate_cleaning_suggestions(df: pd.DataFrame, cols: List[ColumnInfo]) -> List[str]:
    """
    Generate data cleaning suggestions based on data analysis.
    Returns a list of human-readable suggestions for the UI.
    """
    suggestions = []
    
    # Check for high missing values
    for col in cols:
        if col.missing_pct > 50:
            suggestions.append(f"Column '{col.name}' has {col.missing_pct:.1f}% missing values - consider dropping or imputing")
        elif col.missing_pct > 20:
            suggestions.append(f"Column '{col.name}' has {col.missing_pct:.1f}% missing values - may need imputation")
    
    # Check for potential duplicates
    if df.duplicated().sum() > 0:
        dup_count = df.duplicated().sum()
        dup_pct = (dup_count / len(df)) * 100
        suggestions.append(f"Found {dup_count} duplicate rows ({dup_pct:.1f}%) - consider removing duplicates")
    
    # Check for single-value columns (constant)
    for col in cols:
        if col.name in df.columns:
            if df[col.name].nunique() == 1:
                suggestions.append(f"Column '{col.name}' has only one unique value - consider dropping")
    
    # Check for high cardinality text columns
    for col in cols:
        if col.type == "text" and col.name in df.columns:
            unique_ratio = df[col.name].nunique() / len(df)
            if unique_ratio > 0.95:
                suggestions.append(f"Column '{col.name}' has very high cardinality ({unique_ratio*100:.1f}%) - may not be useful for analysis")
    
    # Check for numeric columns that might be categorical
    for col in cols:
        if col.type == "numeric" and col.name in df.columns:
            unique_count = df[col.name].nunique()
            if unique_count <= 10 and unique_count > 1:
                suggestions.append(f"Column '{col.name}' has only {unique_count} unique values - consider treating as categorical")
    
    # If no issues found, add a positive message
    if not suggestions:
        suggestions.append("Data looks clean! No major issues detected.")
    
    return suggestions


def _apply_filters(df: pd.DataFrame, filters: List[QueryFilter]) -> pd.DataFrame:
    """Apply filters to a DataFrame based on QueryFilter objects."""
    result_df = df.copy()
    
    for f in filters:
        col = f.column
        if col not in result_df.columns:
            continue
            
        if f.op == "==":
            result_df = result_df[result_df[col] == f.value]
        elif f.op == "!=":
            result_df = result_df[result_df[col] != f.value]
        elif f.op == ">":
            result_df = result_df[result_df[col] > f.value]
        elif f.op == ">=":
            result_df = result_df[result_df[col] >= f.value]
        elif f.op == "<":
            result_df = result_df[result_df[col] < f.value]
        elif f.op == "<=":
            result_df = result_df[result_df[col] <= f.value]
        elif f.op == "in" and f.values:
            result_df = result_df[result_df[col].isin(f.values)]
        elif f.op == "not_in" and f.values:
            result_df = result_df[~result_df[col].isin(f.values)]
        elif f.op == "contains":
            result_df = result_df[result_df[col].astype(str).str.contains(str(f.value), case=False, na=False)]
        elif f.op == "starts_with":
            result_df = result_df[result_df[col].astype(str).str.startswith(str(f.value), na=False)]
        elif f.op == "is_null":
            result_df = result_df[result_df[col].isna()]
        elif f.op == "is_not_null":
            result_df = result_df[result_df[col].notna()]
        elif f.op == "between" and f.values and len(f.values) == 2:
            result_df = result_df[result_df[col].between(f.values[0], f.values[1])]
    
    return result_df


def _execute_aggregate_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute aggregate operation with grouping and metrics."""
    # Apply filters first
    if query.filters:
        df = _apply_filters(df, query.filters)
    
    # Build aggregation dictionary
    agg_dict = {}
    aliases = {}
    
    if query.metrics:
        for metric in query.metrics:
            col = metric.column
            agg = metric.agg
            alias = metric.alias or f"{col}_{agg}"
            
            if col == "*":
                # Count all rows
                if query.group_by:
                    agg_dict[query.group_by[0]] = "count"
                    aliases[query.group_by[0]] = alias
            else:
                if col not in df.columns:
                    continue
                agg_dict[col] = agg
                aliases[col] = alias
    
    # Perform aggregation
    if query.group_by:
        result = df.groupby(query.group_by).agg(agg_dict).reset_index()
        # Rename columns to aliases
        for old_col, new_col in aliases.items():
            if old_col in result.columns:
                result = result.rename(columns={old_col: new_col})
    else:
        # Global aggregation
        result = df.agg(agg_dict).to_frame().T
        for old_col, new_col in aliases.items():
            if old_col in result.columns:
                result = result.rename(columns={old_col: new_col})
    
    # Apply ordering
    if query.order_by:
        for order in query.order_by:
            if order.column in result.columns:
                ascending = order.direction.lower() == "asc"
                result = result.sort_values(by=order.column, ascending=ascending)
    
    # Apply limit
    if query.limit:
        result = result.head(query.limit)
    
    # Convert to dict
    data = result.to_dict(orient="records")
    
    return {
        "data": data,
        "row_count": len(data),
        "columns": list(result.columns)
    }


def _execute_filter_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute filter operation and return matching rows."""
    if query.filters:
        df = _apply_filters(df, query.filters)
    
    # Apply ordering
    if query.order_by:
        for order in query.order_by:
            if order.column in df.columns:
                ascending = order.direction.lower() == "asc"
                df = df.sort_values(by=order.column, ascending=ascending)
    
    # Apply limit
    if query.limit:
        df = df.head(query.limit)
    
    # Prepare for JSON serialization
    result_df = df.where(pd.notnull(df), None)
    data = result_df.to_dict(orient="records")
    
    return {
        "data": data,
        "row_count": len(data),
        "columns": list(df.columns)
    }


def _execute_describe_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute describe operation on specified columns."""
    columns = query.options.get("columns", []) if query.options else []
    
    if not columns:
        # Describe all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result = {}
    for col in columns:
        if col in df.columns:
            desc = df[col].describe()
            result[col] = {
                "count": int(desc.get("count", 0)),
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "25%": float(desc.get("25%", 0)),
                "50%": float(desc.get("50%", 0)),
                "75%": float(desc.get("75%", 0)),
                "max": float(desc.get("max", 0)),
            }
    
    return {"statistics": result}


def _execute_correlation_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute correlation operation."""
    columns = query.options.get("columns", []) if query.options else []
    method = query.options.get("method", "pearson") if query.options else "pearson"
    
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to requested columns
    df_subset = df[columns].dropna()
    
    if len(df_subset.columns) < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}
    
    corr = df_subset.corr(method=method)
    corr_dict = {}
    for col in corr.columns:
        corr_dict[col] = {idx: float(val) for idx, val in corr[col].items()}
    
    return {"correlation_matrix": corr_dict, "method": method}


def _execute_distinct_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute distinct operation to get unique values."""
    columns = query.options.get("columns", []) if query.options else []
    
    result = {}
    for col in columns:
        if col in df.columns:
            value_counts = df[col].value_counts()
            result[col] = {
                "unique_count": int(df[col].nunique()),
                "values": df[col].unique().tolist()[:100],  # Limit to 100 for safety
                "value_counts": {str(k): int(v) for k, v in value_counts.head(50).items()}
            }
    
    return {"distinct_values": result}


def _execute_percentile_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute percentile calculation."""
    columns = query.options.get("columns", []) if query.options else []
    percentiles = query.options.get("percentiles", [0.25, 0.5, 0.75, 0.95]) if query.options else [0.25, 0.5, 0.75, 0.95]
    
    result = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            result[col] = {}
            for p in percentiles:
                result[col][f"p{int(p*100)}"] = float(df[col].quantile(p))
    
    return {"percentiles": result}


def _execute_crosstab_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute crosstab/pivot table operation."""
    options = query.options or {}
    row_column = options.get("row_column")
    col_column = options.get("col_column")
    value_column = options.get("value_column")
    value_agg = options.get("value_agg", "count")
    
    if not row_column or not col_column:
        return {"error": "crosstab requires row_column and col_column"}
    
    # Apply filters first
    if query.filters:
        df = _apply_filters(df, query.filters)
    
    # Create crosstab
    if value_column and value_column in df.columns:
        # Aggregate a specific column
        result = pd.crosstab(
            df[row_column],
            df[col_column],
            values=df[value_column],
            aggfunc=value_agg
        )
    else:
        # Count occurrences
        result = pd.crosstab(df[row_column], df[col_column])
    
    # Convert to dict format
    crosstab_data = result.reset_index().to_dict(orient="records")
    
    return {
        "crosstab": crosstab_data,
        "row_column": row_column,
        "col_column": col_column,
        "columns": list(result.columns),
        "row_count": len(result)
    }


def _execute_rank_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute ranking operation."""
    options = query.options or {}
    rank_by = options.get("rank_by")  # Column to rank by
    ascending = options.get("ascending", False)
    method = options.get("method", "dense")  # "average", "min", "max", "dense", "first"
    
    if not rank_by or rank_by not in df.columns:
        return {"error": "rank operation requires valid rank_by column"}
    
    # Apply filters first
    if query.filters:
        df = _apply_filters(df, query.filters)
    
    # Add rank column
    df_copy = df.copy()
    df_copy["rank"] = df_copy[rank_by].rank(ascending=ascending, method=method)
    
    # Add percentile rank (0-100)
    df_copy["percentile"] = df_copy[rank_by].rank(pct=True, ascending=ascending) * 100
    
    # Apply ordering
    df_copy = df_copy.sort_values(by=rank_by, ascending=ascending)
    
    # Apply limit
    if query.limit:
        df_copy = df_copy.head(query.limit)
    
    # Prepare for JSON
    result_df = df_copy.where(pd.notnull(df_copy), None)
    data = result_df.to_dict(orient="records")
    
    return {
        "data": data,
        "row_count": len(data),
        "columns": list(df_copy.columns),
        "ranked_by": rank_by
    }


def _execute_binning_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute binning/bucketing operation."""
    options = query.options or {}
    column = options.get("column")
    bins = options.get("bins")  # Can be int (number of bins) or list (bin edges)
    labels = options.get("labels")  # Custom labels for bins
    
    if not column or column not in df.columns:
        return {"error": "binning requires valid column"}
    
    if not bins:
        return {"error": "binning requires bins parameter"}
    
    # Apply filters first
    if query.filters:
        df = _apply_filters(df, query.filters)
    
    df_copy = df.copy()
    
    # Create bins
    try:
        if isinstance(bins, int):
            # Equal-width bins
            df_copy["bin"] = pd.cut(df_copy[column], bins=bins, labels=labels)
        else:
            # Custom bin edges
            df_copy["bin"] = pd.cut(df_copy[column], bins=bins, labels=labels)
        
        # Count by bin
        bin_counts = df_copy["bin"].value_counts().sort_index()
        
        # Also calculate statistics per bin if requested
        if query.group_by or query.metrics:
            grouped = df_copy.groupby("bin")
            
            if query.metrics:
                agg_dict = {}
                for metric in query.metrics:
                    if metric.column in df_copy.columns:
                        agg_dict[metric.column] = metric.agg
                
                stats = grouped.agg(agg_dict).reset_index()
                data = stats.to_dict(orient="records")
            else:
                data = bin_counts.reset_index().to_dict(orient="records")
        else:
            data = bin_counts.reset_index().to_dict(orient="records")
        
        return {
            "data": data,
            "column": column,
            "bin_count": len(bin_counts),
            "total_rows": len(df_copy)
        }
    except Exception as e:
        return {"error": f"Binning failed: {str(e)}"}


def _execute_sample_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Execute sampling operation."""
    options = query.options or {}
    n = options.get("n", 100)  # Number of samples
    method = options.get("method", "random")  # "random", "stratified"
    stratify_column = options.get("stratify_column")  # For stratified sampling
    random_state = options.get("random_state", 42)
    
    # Apply filters first
    if query.filters:
        df = _apply_filters(df, query.filters)
    
    if len(df) == 0:
        return {"error": "No data after filtering"}
    
    # Ensure n doesn't exceed dataframe size
    n = min(n, len(df))
    
    try:
        if method == "stratified" and stratify_column and stratify_column in df.columns:
            # Stratified sampling - sample proportionally from each group
            sampled = df.groupby(stratify_column, group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(n * len(x) / len(df))), random_state=random_state)
            )
        else:
            # Simple random sampling
            sampled = df.sample(n=n, random_state=random_state)
        
        # Prepare for JSON
        result_df = sampled.where(pd.notnull(sampled), None)
        data = result_df.to_dict(orient="records")
        
        return {
            "data": data,
            "sample_size": len(data),
            "total_population": len(df),
            "sampling_method": method
        }
    except Exception as e:
        return {"error": f"Sampling failed: {str(e)}"}


def _execute_advanced_describe_query(df: pd.DataFrame, query: QueryRequest) -> Dict:
    """Enhanced describe with additional statistics."""
    columns = query.options.get("columns", []) if query.options else []
    
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            desc = series.describe()
            
            # Basic stats
            stats = {
                "count": int(desc.get("count", 0)),
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "q25": float(desc.get("25%", 0)),
                "median": float(desc.get("50%", 0)),
                "q75": float(desc.get("75%", 0)),
                "max": float(desc.get("max", 0)),
            }
            
            # Additional quantiles
            stats["q10"] = float(series.quantile(0.10))
            stats["q90"] = float(series.quantile(0.90))
            stats["q95"] = float(series.quantile(0.95))
            
            # IQR (Interquartile Range)
            stats["iqr"] = float(stats["q75"] - stats["q25"])
            
            # MAD (Median Absolute Deviation)
            mad = (series - series.median()).abs().median()
            stats["mad"] = float(mad)
            
            # Skewness and Kurtosis
            from scipy import stats as scipy_stats
            stats["skewness"] = float(scipy_stats.skew(series))
            stats["kurtosis"] = float(scipy_stats.kurtosis(series))
            
            # Coefficient of Variation (CV)
            if stats["mean"] != 0:
                stats["cv"] = float(stats["std"] / stats["mean"])
            else:
                stats["cv"] = None
            
            result[col] = stats
    
    return {"statistics": result}


# ---------------------------------------------------
# API endpoints
# ---------------------------------------------------
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": "0.4.0"}


@app.get("/sessions/stats", response_model=SessionStats)
async def session_stats():
    """
    Get statistics about current sessions.
    Useful for monitoring memory usage and session lifecycle.
    """
    total_sessions = len(SESSIONS)
    
    # Calculate total memory usage
    total_memory = 0
    oldest_timestamp = None
    
    for df, timestamp in SESSIONS.values():
        total_memory += df.memory_usage(deep=True).sum()
        if oldest_timestamp is None or timestamp < oldest_timestamp:
            oldest_timestamp = timestamp
    
    # Calculate age of oldest session
    oldest_age_minutes = None
    if oldest_timestamp:
        oldest_age_minutes = (datetime.now() - oldest_timestamp).total_seconds() / 60
    
    return SessionStats(
        total_sessions=total_sessions,
        memory_used_mb=round(total_memory / 1024 / 1024, 2),
        oldest_session_age_minutes=round(oldest_age_minutes, 2) if oldest_age_minutes else None,
        session_ids=list(SESSIONS.keys())
    )


@app.post("/sessions/cleanup")
async def cleanup_sessions(max_age_hours: int = 24):
    """
    Clean up expired sessions older than max_age_hours.
    Returns the number of sessions cleaned.
    """
    cleaned = _clean_old_sessions(max_age_hours)
    remaining = len(SESSIONS)
    
    logger.info(f"Cleaned {cleaned} sessions, {remaining} remaining")
    
    return {
        "cleaned": cleaned,
        "remaining": remaining,
        "max_age_hours": max_age_hours
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV/Excel file, store a cleaned DataFrame in memory,
    and return a profile summary + session_id + real sample_rows.
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        # Clean old sessions before processing new upload
        cleaned = _clean_old_sessions(max_age_hours=24)
        if cleaned > 0:
            logger.info(f"Auto-cleaned {cleaned} expired sessions")
        
        # 1) Load the file
        df = _load_dataframe(file)
        logger.info(f"Successfully loaded dataframe with shape: {df.shape}")
        
        # 2) Auto-detect and convert date columns
        df = _auto_detect_dates(df)
        
        # 3) Basic cleaning: drop columns that are all missing
        df = df.dropna(axis=1, how="all")
        logger.info(f"After cleaning, shape: {df.shape}")
        
        # 4) Store in session with timestamp
        session_id = str(uuid4())
        _store_session(session_id, df)
        logger.info(f"Stored dataframe with session_id: {session_id}")
        
        # 5) Build profile
        profile = _build_profile(df, session_id)
        logger.info(f"Built profile with {len(profile.columns)} columns")
        
        # 6) Prepare sample rows with proper serialization (200 rows, max 500)
        sample_rows = _prepare_sample_rows(df, max_rows=200)
        logger.info(f"Prepared {len(sample_rows)} sample rows")
        
        # 7) Return response using the UploadResponse model
        # Note: profile already contains missing_summary and cleaning_suggestions
        return UploadResponse(
            session_id=profile.session_id,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            columns=profile.columns,
            schema=profile.schema,
            descriptives=profile.descriptives,
            missing_summary=profile.missing_summary,
            cleaning_suggestions=profile.cleaning_suggestions,
            sample_rows=sample_rows
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Upload failed: {str(e)}"
        )


@app.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def run_analysis(session_id: str):
    """
    Run correlation, simple group tests, and linear regression
    for the given session_id.
    """
    try:
        logger.info(f"Running analysis for session: {session_id}")
        
        df = _get_session_dataframe(session_id)
        
        profile = _build_profile(df, session_id)
        correlation = _build_correlation(df)
        tests = _auto_tests(df, profile)
        regression = _auto_regression(df, profile)

        logger.info(f"Analysis complete for session: {session_id}")

        return AnalysisResponse(
            session_id=session_id,
            correlation=correlation,
            tests=tests,
            regression=regression,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/schema/{session_id}", response_model=SchemaResponse)
async def get_schema(session_id: str):
    """
    Get the schema (column information) for a previously uploaded dataset.
    This is useful for fetching column metadata without re-uploading the file.
    """
    try:
        logger.info(f"Fetching schema for session: {session_id}")
        
        df = _get_session_dataframe(session_id)
        profile = _build_profile(df, session_id)
        
        return SchemaResponse(
            session_id=session_id,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            schema=profile.schema,
            missing_summary=profile.missing_summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schema fetch error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch schema: {str(e)}"
        )


@app.get("/sample/{session_id}", response_model=SampleResponse)
async def get_sample(session_id: str, max_rows: int = 100):
    """
    Get sample rows from a previously uploaded dataset.
    
    Args:
        session_id: The session ID from the upload response
        max_rows: Maximum number of rows to return (default: 100, max: 500)
    """
    try:
        logger.info(f"Fetching sample for session: {session_id}, max_rows: {max_rows}")
        
        # Validate max_rows parameter
        if max_rows < 1:
            raise HTTPException(status_code=400, detail="max_rows must be at least 1")
        if max_rows > 500:
            raise HTTPException(status_code=400, detail="max_rows cannot exceed 500")
        
        df = _get_session_dataframe(session_id)
        sample_rows = _prepare_sample_rows(df, max_rows=max_rows)
        
        return SampleResponse(
            session_id=session_id,
            sample_rows=sample_rows,
            total_rows=len(df),
            returned_rows=len(sample_rows),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sample fetch error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch sample: {str(e)}"
        )


@app.post("/query/{session_id}", response_model=QueryResponse)
async def execute_query(session_id: str, query: QueryRequest):
    """
    Execute dynamic queries on uploaded dataset.
    Supports: aggregate, filter, describe, correlation, percentile, distinct operations.
    
    This endpoint enables AI agents to ask any question about the data.
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Executing {query.operation} query for session: {session_id}")
        
        df = _get_session_dataframe(session_id)
        
        # Route to appropriate query executor
        if query.operation == "aggregate":
            result = _execute_aggregate_query(df, query)
            summary = f"Grouped by {query.group_by}, calculated aggregations"
        elif query.operation == "filter":
            result = _execute_filter_query(df, query)
            summary = f"Filtered data with {len(query.filters or [])} conditions"
        elif query.operation == "describe":
            result = _execute_advanced_describe_query(df, query)
            summary = "Enhanced descriptive statistics"
        elif query.operation == "correlation":
            result = _execute_correlation_query(df, query)
            summary = "Correlation analysis"
        elif query.operation == "percentile":
            result = _execute_percentile_query(df, query)
            summary = "Percentile calculation"
        elif query.operation == "distinct":
            result = _execute_distinct_query(df, query)
            summary = "Distinct values analysis"
        elif query.operation == "crosstab":
            result = _execute_crosstab_query(df, query)
            summary = "Crosstab/pivot table"
        elif query.operation == "rank":
            result = _execute_rank_query(df, query)
            summary = "Ranking operation"
        elif query.operation == "binning":
            result = _execute_binning_query(df, query)
            summary = "Binning/bucketing operation"
        elif query.operation == "sample":
            result = _execute_sample_query(df, query)
            summary = "Sampling operation"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported operation: {query.operation}"
            )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return QueryResponse(
            success=True,
            operation=query.operation,
            result=result,
            execution_time_ms=round(execution_time, 2),
            query_summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}", exc_info=True)
        execution_time = (time.time() - start_time) * 1000
        return QueryResponse(
            success=False,
            operation=query.operation,
            result={},
            execution_time_ms=round(execution_time, 2),
            query_summary="Query failed",
            error=str(e)
        )
