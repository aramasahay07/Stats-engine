from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import uuid4
from io import BytesIO
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# ---------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------
app = FastAPI(title="AI Data Lab Stats Engine", version="0.2.0")

# CORS so Lovable (browser frontend) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# In-memory "session store"
# (For MVP only; wiped on restart / cold start)
# ---------------------------------------------------
SESSIONS: Dict[str, pd.DataFrame] = {}


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
    columns: List[ColumnInfo]        # original field
    schema: List[ColumnInfo]         # NEW: duplicate of columns for frontend
    descriptives: List[DescriptiveStats]


class CorrelationResponse(BaseModel):
    matrix: Dict[str, Dict[str, float]]  # col -> col -> corr


class AnalysisResponse(BaseModel):
    session_id: str
    correlation: Optional[CorrelationResponse] = None
    tests: List[TestResult] = []
    regression: Optional[RegressionResult] = None


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


def _load_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read CSV or Excel file into a pandas DataFrame."""
    content = file.file.read()
    file.file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    buffer = BytesIO(content)
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(buffer)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buffer)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload CSV or Excel.",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

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

    # NOTE: schema is just an alias of columns for frontend compatibility
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
        # Fail soft: if stats error, just skip tests
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


# ---------------------------------------------------
# API endpoints
# ---------------------------------------------------
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/upload", response_model=ProfileResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV/Excel file, store a cleaned DataFrame in memory,
    and return a profile summary + session_id.
    """
    df = _load_dataframe(file)

    # Basic cleaning: drop columns that are all missing
    df = df.dropna(axis=1, how="all")

    session_id = str(uuid4())
    SESSIONS[session_id] = df

    profile = _build_profile(df, session_id)
    return profile


@app.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def run_analysis(session_id: str):
    """
    Run correlation, simple group tests, and linear regression
    for the given session_id.
    """
    df = SESSIONS.get(session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    profile = _build_profile(df, session_id)
    correlation = _build_correlation(df)
    tests = _auto_tests(df, profile)
    regression = _auto_regression(df, profile)

    return AnalysisResponse(
        session_id=session_id,
        correlation=correlation,
        tests=tests,
        regression=regression,
    )
