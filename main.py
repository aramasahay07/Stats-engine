from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
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


class UploadResponse(BaseModel):
    session_id: str
    n_rows: int
    n_cols: int
    columns: List[ColumnInfo]
    schema: List[ColumnInfo]
    descriptives: List[DescriptiveStats]
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
        missing_pct = float(s.isna().mean() * 100)

        cols.append(
            ColumnInfo(
                name=col,
                dtype=str(s.dtype),
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

    return ProfileResponse(
        session_id=session_id,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=cols,
        schema=cols,
        descriptives=descriptives,
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
        return UploadResponse(
            session_id=profile.session_id,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            columns=profile.columns,
            schema=profile.schema,
            descriptives=profile.descriptives,
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
