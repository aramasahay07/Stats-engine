"""
AI Data Lab Backend v5.0
Complete Statistical Engine + Transform Engine
Combines v4.0 Minitab-level stats with v2.0 Transform capabilities
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from knowledge.routers.kb import router as kb_router
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from uuid import uuid4
from io import BytesIO, StringIO
import pandas as pd
from fastapi.encoders import jsonable_encoder
import numpy as np
from scipy import stats
from scipy.stats import shapiro, anderson, levene, bartlett, mannwhitneyu, kruskal
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from ai_agent.router import create_agent_router

# Transform engine imports
try:
    from transformers.registry import registry
    from transformers.base import TransformError
    TRANSFORMS_AVAILABLE = True
except ImportError:
    TRANSFORMS_AVAILABLE = False
    print("⚠️  Warning: Transform engine not available. Install transformers package.")

# Transform service
try:
    from transform_service import TransformService
    from session_store import SessionStore
    TRANSFORM_SERVICE_AVAILABLE = True
except ImportError:
    TRANSFORM_SERVICE_AVAILABLE = False
    print("⚠️  Warning: Transform service not available.")

# Type inference
try:
    from utils.type_inference import infer_column_role
    TYPE_INFERENCE_AVAILABLE = True
except ImportError:
    TYPE_INFERENCE_AVAILABLE = False

# ---------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------
app = FastAPI(
    title="AI Data Lab - Complete Analytics Platform",
    version="5.0.0",
    description="Minitab-level Statistics + 60+ Data Transforms + Table Operations"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    kb_router,
    prefix="/kb",
    tags=["Knowledge Base"]
)


# ---------------------------------------------------
# Data Storage
# ---------------------------------------------------
# Use SessionStore if available, otherwise fallback to dict
if TRANSFORM_SERVICE_AVAILABLE:
    session_store = SessionStore()
    transform_service = TransformService()
else:
    SESSIONS: Dict[str, pd.DataFrame] = {}

def _get_df(session_id: str):
    if TRANSFORM_SERVICE_AVAILABLE:
        return session_store.get(session_id)
    return SESSIONS.get(session_id)

# ---------------------------------------------------
# Agent Routes (NEW)
# ---------------------------------------------------
app.include_router(
    create_agent_router(_get_df),
    prefix="/agents"
)

# ---------------------------------------------------
# Pydantic Models - Statistical Analysis
# ---------------------------------------------------
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: str
    missing_pct: float
    unique_count: Optional[int] = None
    sample_values: Optional[List[Any]] = None


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
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


class NormalityTest(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    is_normal: bool
    interpretation: str


class TestResult(BaseModel):
    test_type: str
    target: Optional[str] = None
    group_col: Optional[str] = None
    p_value: float
    statistic: float
    df: Optional[float] = None
    effect_size: Optional[float] = None
    interpretation: str
    post_hoc: Optional[Dict[str, Any]] = None


class RegressionResult(BaseModel):
    target: str
    predictors: List[str]
    n_observations: int
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    aic: float
    bic: float
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    t_values: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]
    residuals: List[float]
    fitted_values: List[float]
    leverage: List[float]
    cooks_distance: List[float]
    vif: Optional[Dict[str, float]] = None
    heteroscedasticity_test: Optional[Dict[str, Any]] = None
    durbin_watson: Optional[float] = None


class ControlChartPoint(BaseModel):
    index: int
    value: float
    ucl: float
    lcl: float
    center: float
    out_of_control: bool
    rule_violations: List[str]


class ControlChartResult(BaseModel):
    chart_type: str
    data_points: List[ControlChartPoint]
    center_line: float
    ucl: float
    lcl: float
    sigma: float
    summary: Dict[str, Any]


class ProcessCapabilityResult(BaseModel):
    cp: float
    cpk: float
    pp: float
    ppk: float
    cpm: Optional[float] = None
    sigma_level: float
    dpmo: float
    expected_within_spec: float
    interpretation: str
    specifications: Dict[str, float]


class ProfileResponse(BaseModel):
    session_id: str
    n_rows: int
    n_cols: int
    columns: List[ColumnInfo]
    schema: List[ColumnInfo]  # Alias for frontend compatibility
    descriptives: List[DescriptiveStats]
    sample_rows: Optional[List[Dict[str, Any]]] = None  # For frontend data preview


class CorrelationResponse(BaseModel):
    matrix: Dict[str, Dict[str, float]]
    method: str = "pearson"


class AnalysisResponse(BaseModel):
    session_id: str
    correlation: Optional[CorrelationResponse] = None
    tests: List[TestResult] = []
    regression: Optional[RegressionResult] = None
    normality_tests: List[NormalityTest] = []


# ---------------------------------------------------
# Pydantic Models - Request Models
# ---------------------------------------------------
class ControlChartRequest(BaseModel):
    chart_type: Literal["xbar", "i", "p"]
    column: str
    subgroup_size: Optional[int] = None


class ProcessCapabilityRequest(BaseModel):
    column: str
    usl: float
    lsl: float
    target: Optional[float] = None


class RegressionRequest(BaseModel):
    target: str
    predictors: List[str]
    include_diagnostics: bool = True


class TransformRequest(BaseModel):
    column: str
    transform_type: str
    params: Optional[Dict[str, Any]] = None


class AdvancedAnalysisRequest(BaseModel):
    analysis_type: Literal["normality", "variance_test", "time_series", "pca", "cluster"]
    target: Optional[str] = None
    group_col: Optional[str] = None
    columns: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None


class AdvancedAnalysisResponse(BaseModel):
    session_id: str
    analysis_type: str
    result: Dict[str, Any]


class QueryRequest(BaseModel):
    """Flexible query request supporting multiple formats"""
    operation: Literal["aggregate", "filter", "distinct", "crosstab", "describe"]
    group_by: Optional[List[str]] = None
    metrics: Optional[List[Dict[str, str]]] = None  # [{"column": "x", "agg": "mean"}]
    aggregations: Optional[Dict[str, str]] = None  # {"total_sales": "sales:sum"} format
    filters: Optional[List[Dict[str, Any]]] = None
    limit: Optional[int] = None


class QueryResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------
# Pydantic Models - Transform Engine Models
# ---------------------------------------------------
class TransformSpec(BaseModel):
    type: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TransformRequestV2(BaseModel):
    """Transform request for v2 engine"""
    column: str
    transforms: List[TransformSpec]


class TransformMetadata(BaseModel):
    source_column: str
    transform_type: str
    null_count: int
    unique_values: int
    sample_output: Optional[List[Any]] = None


class FilterSpec(BaseModel):
    column: str
    operator: Literal["eq", "ne", "gt", "lt", "gte", "lte", "in", "not_in", "contains", "starts_with"]
    value: Any


class TransformSuggestion(BaseModel):
    transform: str
    usefulness_score: float
    reason: str
    preview: List[Any]
    params: Optional[Dict[str, Any]] = None


class SuggestTransformsResponse(BaseModel):
    column: str
    detected_type: str
    suggested_transforms: List[TransformSuggestion]


class TransformDefinition(BaseModel):
    input_types: List[str]
    output_type: str
    params: Dict[str, Any]
    description: str
    examples: Optional[List[str]] = None


class TransformDiscoveryResponse(BaseModel):
    transforms: Dict[str, TransformDefinition]


# ---------------------------------------------------
# Utility Functions - Type Inference
# ---------------------------------------------------
def _infer_role(series: pd.Series) -> str:
    """Fallback type inference if utils not available"""
    if TYPE_INFERENCE_AVAILABLE:
        return infer_column_role(series)
    
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return "categorical" if unique_ratio < 0.2 else "text"


# ---------------------------------------------------
# Utility Functions - Data Loading
# ---------------------------------------------------
def _load_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read CSV or Excel file into pandas DataFrame"""
    content = file.file.read()
    file.file.close()
    if not content:
        raise HTTPException(400, "Empty file")
    
    buffer = BytesIO(content)
    filename = file.filename.lower()
    
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(buffer)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buffer)
        else:
            raise HTTPException(400, "Unsupported file type. Please upload CSV or Excel.")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {e}")
    
    if df.empty:
        raise HTTPException(400, "File contained no rows.")
    
    return df


# ---------------------------------------------------
# Utility Functions - Profiling
# ---------------------------------------------------
def _build_profile(df: pd.DataFrame, session_id: str) -> ProfileResponse:
    """Create profile summary for the UI"""
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
                unique_count=int(s.nunique()),
                sample_values=s.dropna().head(3).tolist() if len(s.dropna()) > 0 else []
            )
        )
        
        if role == "numeric":
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            descriptives.append(
                DescriptiveStats(
                    column=col,
                    count=int(desc.get("count", 0)),
                    mean=float(desc["mean"]) if not np.isnan(desc["mean"]) else None,
                    std=float(desc["std"]) if not np.isnan(desc["std"]) else None,
                    min=float(desc["min"]) if not np.isnan(desc["min"]) else None,
                    q25=float(desc["25%"]) if not np.isnan(desc["25%"]) else None,
                    median=float(desc["50%"]) if not np.isnan(desc["50%"]) else None,
                    q75=float(desc["75%"]) if not np.isnan(desc["75%"]) else None,
                    max=float(desc["max"]) if not np.isnan(desc["max"]) else None,
                    skewness=float(s.skew()) if not np.isnan(s.skew()) else None,
                    kurtosis=float(s.kurtosis()) if not np.isnan(s.kurtosis()) else None
                )
            )
    
    # Generate sample rows for frontend data preview
    sample_rows = df.head(100).replace({np.nan: None, pd.NaT: None}).to_dict(orient='records')
    
    return ProfileResponse(
        session_id=session_id,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=cols,
        schema=cols,
        descriptives=descriptives,
        sample_rows=sample_rows
    )


def _build_correlation(df: pd.DataFrame, method: str = "pearson") -> Optional[CorrelationResponse]:
    """Build correlation matrix for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        return None
    
    corr = numeric_df.corr(method=method)
    corr_dict: Dict[str, Dict[str, float]] = {}
    for col in corr.columns:
        corr_dict[col] = {idx: float(val) for idx, val in corr[col].items()}
    
    return CorrelationResponse(matrix=corr_dict, method=method)


# ---------------------------------------------------
# Utility Functions - Statistical Tests
# ---------------------------------------------------
def _normality_tests(series: pd.Series) -> List[NormalityTest]:
    """Run normality tests on a series"""
    tests = []
    data = series.dropna()
    
    if len(data) < 3:
        return tests
    
    # Shapiro-Wilk test
    try:
        stat, p = shapiro(data)
        tests.append(NormalityTest(
            test_name="Shapiro-Wilk",
            statistic=float(stat),
            p_value=float(p),
            is_normal=bool(p > 0.05),
            interpretation=f"Data {'appears' if p > 0.05 else 'does not appear'} normally distributed (p={p:.4f})"
        ))
    except Exception:
        pass
    
    # Anderson-Darling test
    try:
        result = anderson(data)
        tests.append(NormalityTest(
            test_name="Anderson-Darling",
            statistic=float(result.statistic),
            p_value=0.0,  # Anderson doesn't return p-value directly
            is_normal=bool(result.statistic < result.critical_values[2]),
            interpretation=f"Test statistic: {result.statistic:.4f}, Critical value (5%): {result.critical_values[2]:.4f}"
        ))
    except Exception:
        pass
    
    return tests


def _auto_tests(df: pd.DataFrame, profile: ProfileResponse) -> List[TestResult]:
    """Run automatic statistical tests"""
    tests = []
    
    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    cat_cols = [c.name for c in profile.columns if c.role == "categorical"]
    
    if not numeric_cols or not cat_cols:
        return tests
    
    target = numeric_cols[0]
    group_col = cat_cols[0]
    
    data = pd.DataFrame({
        "target": df[target],
        "group": df[group_col].astype("category")
    }).dropna()
    
    if data["group"].nunique() < 2:
        return tests
    
    groups = [g["target"].values for _, g in data.groupby("group")]
    
    try:
        if len(groups) == 2:
            # T-test (Welch's)
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            
            # Calculate Cohen's d (effect size)
            pooled_std = np.sqrt(
                ((len(groups[0])-1)*np.var(groups[0]) + (len(groups[1])-1)*np.var(groups[1])) / 
                (len(groups[0])+len(groups[1])-2)
            )
            cohens_d = (np.mean(groups[0]) - np.mean(groups[1])) / pooled_std if pooled_std > 0 else 0
            
            tests.append(TestResult(
                test_type="t-test (Welch)",
                target=target,
                group_col=group_col,
                p_value=float(p),
                statistic=float(stat),
                effect_size=float(cohens_d),
                interpretation=f"{'Significant' if p < 0.05 else 'No significant'} difference between groups (p={p:.4f})"
            ))
            
            # Mann-Whitney U test (non-parametric alternative)
            stat_u, p_u = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            tests.append(TestResult(
                test_type="Mann-Whitney U",
                target=target,
                group_col=group_col,
                p_value=float(p_u),
                statistic=float(stat_u),
                interpretation=f"{'Significant' if p_u < 0.05 else 'No significant'} difference (non-parametric, p={p_u:.4f})"
            ))
            
        elif len(groups) > 2:
            # One-way ANOVA
            stat_f, p_f = stats.f_oneway(*groups)
            tests.append(TestResult(
                test_type="ANOVA",
                target=target,
                group_col=group_col,
                p_value=float(p_f),
                statistic=float(stat_f),
                df=float(len(groups) - 1),
                interpretation=f"{'Significant' if p_f < 0.05 else 'No significant'} difference among groups (p={p_f:.4f})"
            ))
            
            # Tukey HSD post-hoc test if ANOVA is significant
            if p_f < 0.05:
                try:
                    tukey = pairwise_tukeyhsd(data["target"], data["group"])
                    tests[-1].post_hoc = {"tukey": str(tukey)}
                except Exception:
                    pass
            
            # Kruskal-Wallis test (non-parametric alternative)
            stat_h, p_h = kruskal(*groups)
            tests.append(TestResult(
                test_type="Kruskal-Wallis",
                target=target,
                group_col=group_col,
                p_value=float(p_h),
                statistic=float(stat_h),
                interpretation=f"{'Significant' if p_h < 0.05 else 'No significant'} difference (non-parametric, p={p_h:.4f})"
            ))
            
    except Exception:
        pass
    
    return tests


# ---------------------------------------------------
# Utility Functions - Regression
# ---------------------------------------------------
def _calculate_regression_diagnostics(model, X, y) -> Dict[str, Any]:
    """Calculate comprehensive regression diagnostics"""
    diagnostics = {
        'aic': float(model.aic),
        'bic': float(model.bic),
        'durbin_watson': float(sm.stats.durbin_watson(model.resid)),
        'residuals': model.resid.tolist(),
        'fitted_values': model.fittedvalues.tolist()
    }
    
    # Influence measures
    influence = model.get_influence()
    diagnostics['leverage'] = influence.hat_matrix_diag.tolist()
    diagnostics['cooks_distance'] = influence.cooks_distance[0].tolist()
    
    # VIF (multicollinearity)
    try:
        if X.shape[1] > 1:
            vif_data = {}
            for i, col in enumerate(X.columns):
                if col != 'const':
                    vif_val = variance_inflation_factor(X.values, i)
                    if not np.isinf(vif_val):
                        vif_data[col] = float(vif_val)
            diagnostics['vif'] = vif_data if vif_data else None
    except Exception:
        diagnostics['vif'] = None
    
    # Heteroscedasticity test (Breusch-Pagan)
    try:
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        diagnostics['heteroscedasticity_test'] = {
            'breusch_pagan': {
                'statistic': float(bp_test[0]),
                'p_value': float(bp_test[1]),
                'heteroscedastic': bp_test[1] < 0.05
            }
        }
    except Exception:
        diagnostics['heteroscedasticity_test'] = None
    
    return diagnostics


def _auto_regression(df: pd.DataFrame, profile: ProfileResponse) -> Optional[RegressionResult]:
    """Fit automatic OLS regression with diagnostics"""
    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    
    if len(numeric_cols) < 2:
        return None
    
    target = numeric_cols[0]
    predictors = numeric_cols[1:]
    
    data = df[numeric_cols].dropna()
    if len(data) < 10:
        return None
    
    y = data[target]
    X = sm.add_constant(data[predictors])
    
    try:
        model = sm.OLS(y, X).fit()
        diagnostics = _calculate_regression_diagnostics(model, X, y)
    except Exception:
        return None
    
    return RegressionResult(
        target=target,
        predictors=predictors,
        n_observations=int(model.nobs),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue),
        f_pvalue=float(model.f_pvalue),
        aic=diagnostics['aic'],
        bic=diagnostics['bic'],
        coefficients={n: float(v) for n, v in model.params.items()},
        std_errors={n: float(v) for n, v in model.bse.items()},
        t_values={n: float(v) for n, v in model.tvalues.items()},
        p_values={n: float(v) for n, v in model.pvalues.items()},
        confidence_intervals={
            n: [float(model.conf_int().loc[n, 0]), float(model.conf_int().loc[n, 1])]
            for n in model.params.index
        },
        residuals=diagnostics['residuals'],
        fitted_values=diagnostics['fitted_values'],
        leverage=diagnostics['leverage'],
        cooks_distance=diagnostics['cooks_distance'],
        vif=diagnostics.get('vif'),
        heteroscedasticity_test=diagnostics.get('heteroscedasticity_test'),
        durbin_watson=diagnostics['durbin_watson']
    )


# ---------------------------------------------------
# Utility Functions - Control Charts
# ---------------------------------------------------
def _check_control_rules(points: List[float], center: float, sigma: float) -> List[List[str]]:
    """Check Western Electric rules for control charts"""
    violations = [[] for _ in points]
    
    for i in range(len(points)):
        # Rule 1: Point beyond 3σ
        if abs(points[i] - center) > 3 * sigma:
            violations[i].append("Rule 1: Beyond 3σ")
        
        # Rule 2: 2 out of 3 consecutive points beyond 2σ
        if i >= 2:
            recent = [(points[j] - center) / sigma for j in range(i-2, i+1)]
            if sum(1 for x in recent if x > 2) >= 2:
                violations[i].append("Rule 2: 2/3 points > 2σ")
            if sum(1 for x in recent if x < -2) >= 2:
                violations[i].append("Rule 2: 2/3 points < -2σ")
        
        # Rule 3: 4 out of 5 consecutive points beyond 1σ
        if i >= 4:
            recent = [(points[j] - center) / sigma for j in range(i-4, i+1)]
            if sum(1 for x in recent if x > 1) >= 4:
                violations[i].append("Rule 3: 4/5 points > 1σ")
            if sum(1 for x in recent if x < -1) >= 4:
                violations[i].append("Rule 3: 4/5 points < -1σ")
        
        # Rule 4: 8 consecutive points on same side of center
        if i >= 7:
            recent = [points[j] - center for j in range(i-7, i+1)]
            if all(x > 0 for x in recent) or all(x < 0 for x in recent):
                violations[i].append("Rule 4: 8 consecutive points on same side")
    
    return violations


def _create_control_chart(data: pd.Series, chart_type: str, subgroup_size: Optional[int] = None) -> ControlChartResult:
    """Create control chart (X-bar, I, or P chart)"""
    
    if chart_type == "xbar":
        # X-bar chart (subgroup means)
        subgroup_size = subgroup_size or min(5, len(data) // 10)
        n_subgroups = len(data) // subgroup_size
        
        subgroups = [
            data.iloc[i*subgroup_size:(i+1)*subgroup_size].mean()
            for i in range(n_subgroups)
        ]
        
        xbar_bar = np.mean(subgroups)
        ranges = [
            data.iloc[i*subgroup_size:(i+1)*subgroup_size].max() - 
            data.iloc[i*subgroup_size:(i+1)*subgroup_size].min()
            for i in range(n_subgroups)
        ]
        r_bar = np.mean(ranges)
        
        # Constants for control limits
        d2_values = {2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534, 7:2.704, 8:2.847, 9:2.970, 10:3.078}
        a2_values = {2:1.880, 3:1.023, 4:0.729, 5:0.577, 6:0.483, 7:0.419, 8:0.373, 9:0.337, 10:0.308}
        
        d2 = d2_values.get(subgroup_size, 2.326)
        a2 = a2_values.get(subgroup_size, 0.577)
        
        sigma = r_bar / d2
        ucl = xbar_bar + a2 * r_bar
        lcl = xbar_bar - a2 * r_bar
        
        violations = _check_control_rules(subgroups, xbar_bar, sigma)
        
        points = [
            ControlChartPoint(
                index=i,
                value=float(val),
                ucl=float(ucl),
                lcl=float(lcl),
                center=float(xbar_bar),
                out_of_control=(val > ucl or val < lcl),
                rule_violations=violations[i]
            )
            for i, val in enumerate(subgroups)
        ]
        
        return ControlChartResult(
            chart_type="X-bar",
            data_points=points,
            center_line=float(xbar_bar),
            ucl=float(ucl),
            lcl=float(lcl),
            sigma=float(sigma),
            summary={
                "n_subgroups": n_subgroups,
                "subgroup_size": subgroup_size,
                "out_of_control": sum(1 for p in points if p.out_of_control)
            }
        )
    
    elif chart_type == "i":
        # I-chart (individuals)
        values = data.dropna().values
        x_bar = np.mean(values)
        mr = np.abs(np.diff(values))
        mr_bar = np.mean(mr)
        sigma = mr_bar / 1.128
        
        ucl = x_bar + 3 * sigma
        lcl = x_bar - 3 * sigma
        
        violations = _check_control_rules(values.tolist(), x_bar, sigma)
        
        points = [
            ControlChartPoint(
                index=i,
                value=float(val),
                ucl=float(ucl),
                lcl=float(lcl),
                center=float(x_bar),
                out_of_control=(val > ucl or val < lcl),
                rule_violations=violations[i]
            )
            for i, val in enumerate(values)
        ]
        
        return ControlChartResult(
            chart_type="I-chart",
            data_points=points,
            center_line=float(x_bar),
            ucl=float(ucl),
            lcl=float(lcl),
            sigma=float(sigma),
            summary={
                "n_points": len(values),
                "mr_bar": float(mr_bar),
                "out_of_control": sum(1 for p in points if p.out_of_control)
            }
        )
    
    elif chart_type == "p":
        # P-chart (proportions)
        subgroup_size = subgroup_size or min(50, len(data) // 10)
        n_subgroups = len(data) // subgroup_size
        
        proportions = [
            data.iloc[i*subgroup_size:(i+1)*subgroup_size].mean()
            for i in range(n_subgroups)
        ]
        
        p_bar = np.mean(proportions)
        sigma_p = np.sqrt(p_bar * (1 - p_bar) / subgroup_size)
        
        ucl = p_bar + 3 * sigma_p
        lcl = max(0, p_bar - 3 * sigma_p)
        
        points = [
            ControlChartPoint(
                index=i,
                value=float(val),
                ucl=float(ucl),
                lcl=float(lcl),
                center=float(p_bar),
                out_of_control=(val > ucl or val < lcl),
                rule_violations=[]
            )
            for i, val in enumerate(proportions)
        ]
        
        return ControlChartResult(
            chart_type="P-chart",
            data_points=points,
            center_line=float(p_bar),
            ucl=float(ucl),
            lcl=float(lcl),
            sigma=float(sigma_p),
            summary={
                "n_subgroups": n_subgroups,
                "subgroup_size": subgroup_size,
                "out_of_control": sum(1 for p in points if p.out_of_control)
            }
        )
    
    raise HTTPException(400, f"Chart type '{chart_type}' not implemented")


# ---------------------------------------------------
# Utility Functions - Process Capability
# ---------------------------------------------------
def _calculate_process_capability(
    data: pd.Series,
    usl: float,
    lsl: float,
    target: Optional[float] = None
) -> ProcessCapabilityResult:
    """Calculate process capability indices"""
    
    values = data.dropna().values
    
    if len(values) < 30:
        raise HTTPException(400, "Need at least 30 data points for capability analysis")
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    target = target or (usl + lsl) / 2
    
    # Capability indices
    cp = (usl - lsl) / (6 * std)
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    cpk = min(cpu, cpl)
    cpm = (usl - lsl) / (6 * np.sqrt(std**2 + (mean - target)**2))
    
    # Sigma level
    z_min = min((usl - mean) / std, (mean - lsl) / std)
    
    # DPMO (Defects Per Million Opportunities)
    dpmo = (1 - stats.norm.cdf(z_min)) * 1_000_000
    
    # Expected within spec
    expected = (stats.norm.cdf((usl - mean) / std) - stats.norm.cdf(-(mean - lsl) / std)) * 100
    
    # Interpretation
    if cpk >= 2.0:
        interp = "Excellent (6σ capable)"
    elif cpk >= 1.33:
        interp = "Adequate (4σ capable)"
    elif cpk >= 1.0:
        interp = "Marginal (3σ capable)"
    else:
        interp = "Poor (below 3σ)"
    
    return ProcessCapabilityResult(
        cp=float(cp),
        cpk=float(cpk),
        pp=float(cp),  # Simplified: using cp as pp
        ppk=float(cpk),  # Simplified: using cpk as ppk
        cpm=float(cpm),
        sigma_level=float(z_min),
        dpmo=float(dpmo),
        expected_within_spec=float(expected),
        interpretation=interp,
        specifications={
            "usl": float(usl),
            "lsl": float(lsl),
            "target": float(target),
            "mean": float(mean),
            "std": float(std)
        }
    )


# ---------------------------------------------------
# Helper Functions - Session Management
# ---------------------------------------------------
def _get_session(session_id: str) -> pd.DataFrame:
    """Get dataframe from session store"""
    if TRANSFORM_SERVICE_AVAILABLE:
        df = session_store.get(session_id)
    else:
        df = SESSIONS.get(session_id)
    
    if df is None:
        raise HTTPException(404, "Session not found")
    
    return df


def _set_session(session_id: str, df: pd.DataFrame, metadata: Optional[dict] = None):
    """Store dataframe in session"""
    if TRANSFORM_SERVICE_AVAILABLE:
        session_store.set(session_id, df, metadata)
    else:
        SESSIONS[session_id] = df


# ===================================================
# API ENDPOINTS - Core
# ===================================================
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "5.0.0",
        "features": {
            "statistics": True,
            "transforms": TRANSFORMS_AVAILABLE,
            "transform_service": TRANSFORM_SERVICE_AVAILABLE,
            "quality_control": True,
            "advanced_analytics": True
        }
    }

@app.get("/healthz")
def healthz():
    return health_check()

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Data Lab - Complete Analytics Platform",
        "version": "5.0.0",
        "description": "Minitab-level Statistics + 60+ Transforms + Table Operations",
        "endpoints": {
            "upload": "POST /upload - Upload CSV/Excel file",
            "analysis": "GET /analysis/{session_id} - Full statistical analysis",
            "query": "POST /query/{session_id} - Flexible data queries",
            "transforms": "GET /transforms - List available transforms",
            "control_chart": "POST /control-chart/{session_id} - Quality control charts",
            "capability": "POST /process-capability/{session_id} - Process capability analysis",
            "docs": "/docs - Interactive API documentation"
        },
        "capabilities": [
            "Descriptive Statistics (mean, median, std, skewness, kurtosis)",
            "Statistical Tests (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis)",
            "Regression Analysis (with diagnostics: VIF, Cook's D, Durbin-Watson)",
            "Control Charts (X-bar, I-chart, P-chart)",
            "Process Capability (Cp, Cpk, Six Sigma metrics)",
            "60+ Data Transforms (datetime, numeric, text, categorical, ML)",
            "Table Operations (group by, pivot, merge, filter)",
            "Advanced Analytics (PCA, clustering, time series)"
        ]
    }


@app.post("/upload", response_model=ProfileResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV/Excel file, store as STAGING data,
    and return a profile summary + session_id
    """
    df = _load_dataframe(file)
    
    # Basic cleaning: drop columns that are all missing
    df = df.dropna(axis=1, how="all")
    
    session_id = str(uuid4())
    _set_session(session_id, df, metadata={"filename": file.filename})
    
    profile = _build_profile(df, session_id)
    return profile


# ===================================================
# API ENDPOINTS - Statistical Analysis
# ===================================================
@app.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def run_analysis(session_id: str):
    """
    Run comprehensive statistical analysis:
    - Correlation matrix
    - Automatic group tests (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis)
    - Regression with diagnostics
    - Normality tests
    """
    df = _get_session(session_id)
    profile = _build_profile(df, session_id)

    correlation = _build_correlation(df)
    tests = _auto_tests(df, profile)
    regression = _auto_regression(df, profile)

    # Run normality tests on numeric columns
    normality_tests = []
    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        normality_tests.extend(_normality_tests(df[col]))

    analysis = AnalysisResponse(
        session_id=session_id,
        correlation=correlation,
        tests=tests,
        regression=regression,
        normality_tests=normality_tests
    )

    return jsonable_encoder(
        analysis,
        custom_encoder={
            np.bool_: bool,
            np.generic: lambda x: x.item(),
            np.ndarray: lambda x: x.tolist(),
        },
    )

@app.post("/advanced-analysis/{session_id}", response_model=AdvancedAnalysisResponse)
async def advanced_analysis(session_id: str, request: AdvancedAnalysisRequest):
    """
    Advanced analytics:
    - normality: Comprehensive normality testing
    - variance_test: Levene's and Bartlett's tests
    - time_series: Seasonal decomposition
    - pca: Principal Component Analysis
    - cluster: K-means clustering
    """
    df = _get_session(session_id)
    
    result = {}
    
    if request.analysis_type == "normality":
        # Normality tests on target column
        if not request.target or request.target not in df.columns:
            raise HTTPException(400, "Target column required for normality test")
        
        tests = _normality_tests(df[request.target])
        result = {"tests": [t.dict() for t in tests]}
    
    elif request.analysis_type == "variance_test":
        # Test for equal variances across groups
        if not request.target or not request.group_col:
            raise HTTPException(400, "Target and group_col required for variance test")
        
        data = df[[request.target, request.group_col]].dropna()
        groups = [g[request.target].values for _, g in data.groupby(request.group_col)]
        
        if len(groups) < 2:
            raise HTTPException(400, "Need at least 2 groups")
        
        # Levene's test (robust to non-normality)
        stat_levene, p_levene = levene(*groups)
        
        # Bartlett's test (assumes normality)
        stat_bartlett, p_bartlett = bartlett(*groups)
        
        result = {
            "levene": {
                "statistic": float(stat_levene),
                "p_value": float(p_levene),
                "equal_variance": p_levene > 0.05,
                "interpretation": f"Variances {'appear' if p_levene > 0.05 else 'do not appear'} equal (robust test)"
            },
            "bartlett": {
                "statistic": float(stat_bartlett),
                "p_value": float(p_bartlett),
                "equal_variance": p_bartlett > 0.05,
                "interpretation": f"Variances {'appear' if p_bartlett > 0.05 else 'do not appear'} equal (assumes normality)"
            }
        }
    
    elif request.analysis_type == "time_series":
        # Seasonal decomposition
        if not request.target or request.target not in df.columns:
            raise HTTPException(400, "Target column required for time series analysis")
        
        series = df[request.target].dropna()
        
        if len(series) < 24:
            raise HTTPException(400, "Need at least 24 observations for seasonal decomposition")
        
        period = request.params.get("period", 12) if request.params else 12
        
        decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
        
        result = {
            "trend": decomposition.trend.tolist(),
            "seasonal": decomposition.seasonal.tolist(),
            "residual": decomposition.resid.tolist(),
            "observed": decomposition.observed.tolist()
        }
    
    elif request.analysis_type == "pca":
        # Principal Component Analysis
        if not request.columns:
            # Use all numeric columns
            request.columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        data = df[request.columns].dropna()
        
        if len(data) < 10 or len(request.columns) < 2:
            raise HTTPException(400, "Need at least 10 rows and 2 numeric columns for PCA")
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # PCA
        n_components = min(len(request.columns), request.params.get("n_components", 3) if request.params else 3)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(scaled_data)
        
        result = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": components.tolist(),
            "n_components": n_components,
            "feature_names": request.columns
        }
    
    elif request.analysis_type == "cluster":
        # K-means clustering
        if not request.columns:
            request.columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        data = df[request.columns].dropna()
        
        if len(data) < 10 or len(request.columns) < 2:
            raise HTTPException(400, "Need at least 10 rows and 2 numeric columns for clustering")
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # K-means
        n_clusters = request.params.get("n_clusters", 3) if request.params else 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        
        result = {
            "labels": labels.tolist(),
            "centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_),
            "n_clusters": n_clusters,
            "feature_names": request.columns
        }
    
    return AdvancedAnalysisResponse(
        session_id=session_id,
        analysis_type=request.analysis_type,
        result=result
    )


@app.post("/control-chart/{session_id}", response_model=ControlChartResult)
async def create_control_chart(session_id: str, request: ControlChartRequest):
    """
    Create control chart for quality control monitoring
    - xbar: X-bar chart (subgroup means)
    - i: I-chart (individual values)
    - p: P-chart (proportions)
    """
    df = _get_session(session_id)
    
    if request.column not in df.columns:
        raise HTTPException(400, f"Column '{request.column}' not found")
    
    return _create_control_chart(df[request.column], request.chart_type, request.subgroup_size)


@app.post("/process-capability/{session_id}", response_model=ProcessCapabilityResult)
async def process_capability(session_id: str, request: ProcessCapabilityRequest):
    """
    Calculate process capability indices (Cp, Cpk, Pp, Ppk)
    and Six Sigma metrics (DPMO, Sigma level)
    """
    df = _get_session(session_id)
    
    if request.column not in df.columns:
        raise HTTPException(400, f"Column '{request.column}' not found")
    
    return _calculate_process_capability(
        df[request.column],
        request.usl,
        request.lsl,
        request.target
    )


@app.post("/regression/{session_id}", response_model=RegressionResult)
async def regression_analysis(session_id: str, request: RegressionRequest):
    """
    Advanced regression analysis with comprehensive diagnostics:
    - VIF (multicollinearity)
    - Cook's distance (influential points)
    - Leverage
    - Heteroscedasticity test
    - Durbin-Watson (autocorrelation)
    """
    df = _get_session(session_id)
    
    # Validate columns
    all_cols = [request.target] + request.predictors
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    
    data = df[all_cols].dropna()
    
    if len(data) < max(10, len(request.predictors) + 1):
        raise HTTPException(400, f"Need at least {max(10, len(request.predictors) + 1)} complete observations")
    
    y = data[request.target]
    X = sm.add_constant(data[request.predictors])
    
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        raise HTTPException(400, f"Regression failed: {e}")
    
    # Calculate diagnostics if requested
    diagnostics = _calculate_regression_diagnostics(model, X, y) if request.include_diagnostics else {}
    
    return RegressionResult(
        target=request.target,
        predictors=request.predictors,
        n_observations=int(model.nobs),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue),
        f_pvalue=float(model.f_pvalue),
        aic=diagnostics.get('aic', float(model.aic)),
        bic=diagnostics.get('bic', float(model.bic)),
        coefficients={n: float(v) for n, v in model.params.items()},
        std_errors={n: float(v) for n, v in model.bse.items()},
        t_values={n: float(v) for n, v in model.tvalues.items()},
        p_values={n: float(v) for n, v in model.pvalues.items()},
        confidence_intervals={
            n: [float(model.conf_int().loc[n, 0]), float(model.conf_int().loc[n, 1])]
            for n in model.params.index
        },
        residuals=diagnostics.get('residuals', []),
        fitted_values=diagnostics.get('fitted_values', []),
        leverage=diagnostics.get('leverage', []),
        cooks_distance=diagnostics.get('cooks_distance', []),
        vif=diagnostics.get('vif'),
        heteroscedasticity_test=diagnostics.get('heteroscedasticity_test'),
        durbin_watson=diagnostics.get('durbin_watson')
    )


# ===================================================
# API ENDPOINTS - Transform Engine (v1 style)
# ===================================================
@app.post("/transform/{session_id}")
async def apply_transform_v1(session_id: str, request: TransformRequest):
    """
    Apply a single transform (v1 style endpoint)
    """
    if not TRANSFORMS_AVAILABLE:
        raise HTTPException(501, "Transform engine not available")
    
    df = _get_session(session_id)
    
    if request.column not in df.columns:
        raise HTTPException(400, f"Column '{request.column}' not found")
    
    try:
        transformer = registry.create(request.transform_type, request.params or {})
        result = transformer.transform(df[request.column], request.column)
        
        # Create new column
        new_col_name = f"{request.column}_{request.transform_type}"
        df[new_col_name] = result
        _set_session(session_id, df)
        
        return {
            "success": True,
            "column_created": new_col_name,
            "null_count": int(result.isna().sum()),
            "sample": result.dropna().head(5).tolist()
        }
    except TransformError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Transform failed: {e}")


@app.get("/transform/{session_id}/suggest")
async def suggest_transforms_v1(session_id: str, column: Optional[str] = None):
    """
    Get transform suggestions for a column (v1 style endpoint)
    """
    if not TRANSFORMS_AVAILABLE:
        raise HTTPException(501, "Transform engine not available")
    
    df = _get_session(session_id)
    
    if column and column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")
    
    if column:
        suggestions = registry.suggest_transforms(df[column], column)
        return {"column": column, "suggestions": suggestions}
    else:
        # Suggest for all columns
        all_suggestions = {}
        for col in df.columns:
            all_suggestions[col] = registry.suggest_transforms(df[col], col)
        return all_suggestions


# ===================================================
# API ENDPOINTS - Session Management
# ===================================================
@app.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """Get session metadata"""
    df = _get_session(session_id)
    
    if TRANSFORM_SERVICE_AVAILABLE:
        metadata = session_store.get_metadata(session_id) or {}
    else:
        metadata = {}
    
    return {
        "session_id": session_id,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        **metadata
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory"""
    if TRANSFORM_SERVICE_AVAILABLE:
        if not session_store.exists(session_id):
            raise HTTPException(404, "Session not found")
        session_store.delete(session_id)
    else:
        if session_id not in SESSIONS:
            raise HTTPException(404, "Session not found")
        del SESSIONS[session_id]
    
    return {"message": "Session deleted successfully"}


# ===================================================
# API ENDPOINTS - Data Access
# ===================================================
@app.get("/sample/{session_id}")
async def get_sample_data(session_id: str, n: int = 100):
    """
    Get sample rows from the session
    Used by frontend for data preview
    """
    df = _get_session(session_id)
    
    sample_rows = df.head(n).replace({np.nan: None, pd.NaT: None}).to_dict(orient="records")
    
    return {
        "session_id": session_id,
        "sample_rows": sample_rows,
        "total_rows": len(df),
        "returned_rows": len(sample_rows)
    }


@app.get("/schema/{session_id}")
async def get_schema(session_id: str):
    """Return column schema for a session"""
    df = _get_session(session_id)
    
    schema = []
    missing_summary = {}
    
    for col in df.columns:
        s = df[col]
        role = _infer_role(s)
        missing_pct = float(s.isna().mean() * 100)
        missing_summary[col] = round(missing_pct, 2)
        
        schema.append({
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "missing_pct": round(missing_pct, 2),
            "unique_count": int(s.nunique())
        })
    
    return {
        "session_id": session_id,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "schema": schema,
        "missing_summary": missing_summary
    }


@app.post("/query/{session_id}", response_model=QueryResponse)
async def query_dataset(session_id: str, request: QueryRequest):
    """
    Execute aggregation/filter queries on the dataset.
    CRITICAL for AI chat and chart generation in frontend.
    
    Supports:
    - aggregate: Group by and aggregate with metrics
    - filter: Filter rows based on conditions
    - distinct: Get unique rows
    - crosstab: Create pivot-style cross tabulation
    - describe: Statistical summary
    """
    df = _get_session(session_id)
    
    try:
        result_df = df.copy()
        
        # Apply filters first
        if request.filters:
            for f in request.filters:
                col = f.get("column")
                op = f.get("op") or f.get("operator", "==")
                val = f.get("value")
                
                if col not in result_df.columns:
                    continue
                
                if op == "==" or op == "eq":
                    result_df = result_df[result_df[col] == val]
                elif op == "!=" or op == "ne":
                    result_df = result_df[result_df[col] != val]
                elif op == ">" or op == "gt":
                    result_df = result_df[result_df[col] > val]
                elif op == "<" or op == "lt":
                    result_df = result_df[result_df[col] < val]
                elif op == ">=" or op == "gte":
                    result_df = result_df[result_df[col] >= val]
                elif op == "<=" or op == "lte":
                    result_df = result_df[result_df[col] <= val]
                elif op == "in":
                    result_df = result_df[result_df[col].isin(val if isinstance(val, list) else [val])]
                elif op == "not_in":
                    result_df = result_df[~result_df[col].isin(val if isinstance(val, list) else [val])]
                elif op == "contains":
                    result_df = result_df[result_df[col].astype(str).str.contains(str(val), case=False, na=False)]
                elif op == "starts_with":
                    result_df = result_df[result_df[col].astype(str).str.startswith(str(val), na=False)]
        
        # Handle different operations
        if request.operation == "aggregate":
            if request.group_by:
                # Handle aggregations - support both formats
                agg_dict = {}
                
                # Format 1: metrics list [{"column": "x", "agg": "mean"}]
                if request.metrics:
                    for m in request.metrics:
                        col = m.get("column")
                        agg = m.get("agg", "count")
                        
                        if col == "*" or agg == "count":
                            agg_dict["count"] = (result_df.columns[0], "size")
                        elif col in result_df.columns:
                            agg_dict[f"{col}_{agg}"] = (col, agg)
                
                # Format 2: aggregations dict {"total_sales": "sales:sum"}
                if request.aggregations:
                    for alias, spec in request.aggregations.items():
                        if ":" in spec:
                            col, agg = spec.split(":", 1)
                            if col == "*" or agg == "count":
                                agg_dict[alias] = (result_df.columns[0], "size")
                            elif col in result_df.columns:
                                agg_dict[alias] = (col, agg)
                
                # If no aggregations specified, default to count
                if not agg_dict:
                    agg_dict["count"] = (result_df.columns[0], "size")
                
                # Perform groupby
                result_df = result_df.groupby(request.group_by, as_index=False).agg(**agg_dict)
            else:
                # Aggregate without grouping (overall stats)
                agg_result = {}
                if request.metrics:
                    for m in request.metrics:
                        col = m.get("column")
                        agg = m.get("agg", "count")
                        if col in result_df.columns:
                            if agg == "mean":
                                agg_result[f"{col}_mean"] = float(result_df[col].mean())
                            elif agg == "sum":
                                agg_result[f"{col}_sum"] = float(result_df[col].sum())
                            elif agg == "count":
                                agg_result[f"{col}_count"] = int(result_df[col].count())
                            elif agg == "min":
                                agg_result[f"{col}_min"] = float(result_df[col].min())
                            elif agg == "max":
                                agg_result[f"{col}_max"] = float(result_df[col].max())
                            elif agg == "std":
                                agg_result[f"{col}_std"] = float(result_df[col].std())
                
                return QueryResponse(
                    success=True,
                    result={
                        "data": [agg_result],
                        "columns": list(agg_result.keys()),
                        "row_count": 1
                    }
                )
        
        elif request.operation == "distinct":
            if request.group_by:
                result_df = result_df[request.group_by].drop_duplicates()
            else:
                result_df = result_df.drop_duplicates()
        
        elif request.operation == "describe":
            desc = result_df.describe(include='all').to_dict()
            return QueryResponse(
                success=True,
                result={
                    "data": desc,
                    "columns": list(desc.keys()),
                    "row_count": len(desc)
                }
            )
        
        elif request.operation == "crosstab":
            if request.group_by and len(request.group_by) >= 2:
                # Create crosstab
                index_col = request.group_by[0]
                column_col = request.group_by[1]
                
                # Get value column from metrics or use count
                value_col = None
                agg_func = "count"
                if request.metrics and len(request.metrics) > 0:
                    value_col = request.metrics[0].get("column")
                    agg_func = request.metrics[0].get("agg", "count")
                
                if value_col and value_col in result_df.columns:
                    crosstab = pd.crosstab(
                        result_df[index_col],
                        result_df[column_col],
                        values=result_df[value_col],
                        aggfunc=agg_func
                    )
                else:
                    crosstab = pd.crosstab(result_df[index_col], result_df[column_col])
                
                # Convert to records format
                crosstab_reset = crosstab.reset_index()
                result_df = crosstab_reset
        
        # Apply limit
        if request.limit and request.limit > 0:
            result_df = result_df.head(request.limit)
        
        # Convert to JSON-safe format
        result_df = result_df.replace({np.nan: None, pd.NaT: None, np.inf: None, -np.inf: None})
        
        # Convert result to records
        result_data = result_df.to_dict(orient="records")
        
        return QueryResponse(
            success=True,
            result={
                "data": result_data,
                "columns": list(result_df.columns),
                "row_count": len(result_data)
            }
        )
    
    except Exception as e:
        return QueryResponse(success=False, error=str(e))


# ===================================================
# API ENDPOINTS - Transform Engine v2 (New Enhanced)
# ===================================================
if TRANSFORM_SERVICE_AVAILABLE:
    
    @app.get("/transforms", response_model=TransformDiscoveryResponse)
    def get_all_transforms():
        """Get complete catalog of available transforms"""
        all_transforms = transform_service.get_all_transforms()
        
        transforms_dict = {}
        for name, info in all_transforms.items():
            transforms_dict[name] = TransformDefinition(
                input_types=info.get("input_types", []),
                output_type=info.get("output_type", "unknown"),
                params=info.get("params", {}),
                description=info.get("description", ""),
                examples=info.get("examples", [])
            )
        
        return TransformDiscoveryResponse(transforms=transforms_dict)
    
    
    @app.get("/transforms/for/{column_type}")
    def get_transforms_for_type(column_type: str):
        """Get available transforms for a specific column type"""
        transforms = transform_service.get_available_transforms(column_type)
        return {"column_type": column_type, "transforms": transforms}
    
    
    @app.post("/session/{session_id}/suggest/{column}", response_model=SuggestTransformsResponse)
    def suggest_transforms_v2(session_id: str, column: str, limit: int = 5):
        """Get AI-powered transform suggestions for a column"""
        df = _get_session(session_id)
        
        try:
            suggestions = transform_service.suggest_transforms(df, column, limit)
            return suggestions
        except ValueError as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/transform/preview")
    def preview_transform(
        session_id: str,
        column: str,
        transform_type: str,
        params: Optional[Dict[str, Any]] = None,
        n_rows: int = 100
    ):
        """Preview transform without applying it"""
        df = _get_session(session_id)
        
        try:
            preview = transform_service.preview_transform(df, column, transform_type, params, n_rows)
            return preview
        except Exception as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/transform/apply")
    def apply_transform_v2(
        session_id: str,
        request: TransformRequestV2,
        new_column_name: Optional[str] = None,
        replace_original: bool = False
    ):
        """Apply transform(s) to create a new column or replace existing"""
        df = _get_session(session_id)
        
        try:
            # Apply transform chain
            result_series, metadata_list = transform_service.apply_transform_chain(df, request)
            
            # Determine column name
            if replace_original:
                col_name = request.column
            elif new_column_name:
                col_name = new_column_name
            else:
                # Auto-generate name
                transform_names = "_".join([spec.type for spec in request.transforms])
                col_name = f"{request.column}_{transform_names}"
            
            # Add to dataframe
            df[col_name] = result_series
            _set_session(session_id, df)
            
            return {
                "success": True,
                "column_created": col_name,
                "metadata": metadata_list,
                "n_rows": len(df)
            }
        except Exception as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/transform/batch")
    def apply_batch_transforms(
        session_id: str,
        transforms: Dict[str, TransformRequestV2]
    ):
        """Apply multiple transforms at once"""
        df = _get_session(session_id)
        
        results = {}
        errors = {}
        
        for new_col_name, request in transforms.items():
            try:
                result_series, metadata_list = transform_service.apply_transform_chain(df, request)
                df[new_col_name] = result_series
                results[new_col_name] = {"success": True, "metadata": metadata_list}
            except Exception as e:
                errors[new_col_name] = str(e)
        
        if results:
            _set_session(session_id, df)
        
        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors if errors else None
        }
    
    
    # Table Operations
    @app.post("/session/{session_id}/group_by")
    def group_by_aggregate(
        session_id: str,
        group_by: List[str],
        aggregations: Dict[str, str],
        create_new_session: bool = True
    ):
        """Group by and aggregate"""
        df = _get_session(session_id)
        
        try:
            result = transform_service.group_by(df, group_by, aggregations)
            
            if create_new_session:
                new_session_id = str(uuid4())
                _set_session(new_session_id, result, metadata={"parent_session": session_id})
                return {
                    "success": True,
                    "session_id": new_session_id,
                    "n_rows": len(result),
                    "n_cols": len(result.columns)
                }
            else:
                _set_session(session_id, result)
                return {
                    "success": True,
                    "session_id": session_id,
                    "n_rows": len(result),
                    "n_cols": len(result.columns)
                }
        except Exception as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/pivot")
    def create_pivot_table(
        session_id: str,
        index: List[str],
        columns: str,
        values: str,
        aggfunc: str = "sum"
    ):
        """Create pivot table"""
        df = _get_session(session_id)
        
        try:
            result = transform_service.pivot_table(df, index, columns, values, aggfunc)
            
            new_session_id = str(uuid4())
            _set_session(new_session_id, result, metadata={"parent_session": session_id})
            
            return {
                "success": True,
                "session_id": new_session_id,
                "n_rows": len(result),
                "n_cols": len(result.columns)
            }
        except Exception as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/unpivot")
    def unpivot_table(
        session_id: str,
        id_vars: List[str],
        value_vars: Optional[List[str]] = None,
        var_name: str = "variable",
        value_name: str = "value"
    ):
        """Unpivot/melt table"""
        df = _get_session(session_id)
        
        try:
            result = transform_service.unpivot(df, id_vars, value_vars, var_name, value_name)
            
            new_session_id = str(uuid4())
            _set_session(new_session_id, result, metadata={"parent_session": session_id})
            
            return {
                "success": True,
                "session_id": new_session_id,
                "n_rows": len(result),
                "n_cols": len(result.columns)
            }
        except Exception as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/merge/{other_session_id}")
    def merge_sessions(
        session_id: str,
        other_session_id: str,
        on: Optional[List[str]] = None,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None,
        how: str = "inner"
    ):
        """Merge two datasets (like SQL JOIN)"""
        left_df = _get_session(session_id)
        right_df = _get_session(other_session_id)
        
        try:
            result = transform_service.merge_tables(left_df, right_df, on, left_on, right_on, how)
            
            new_session_id = str(uuid4())
            _set_session(new_session_id, result, metadata={
                "left_session": session_id,
                "right_session": other_session_id,
                "join_type": how
            })
            
            return {
                "success": True,
                "session_id": new_session_id,
                "n_rows": len(result),
                "n_cols": len(result.columns)
            }
        except Exception as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/remove_duplicates")
    def remove_duplicates(
        session_id: str,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ):
        """Remove duplicate rows"""
        df = _get_session(session_id)
        
        original_count = len(df)
        result = transform_service.remove_duplicates(df, subset, keep)
        _set_session(session_id, result)
        
        return {
            "success": True,
            "rows_removed": original_count - len(result),
            "n_rows": len(result)
        }
    
    
    @app.post("/session/{session_id}/fill_missing")
    def fill_missing_values(
        session_id: str,
        column: str,
        method: str = "ffill",
        value: Optional[Any] = None
    ):
        """Fill missing values in a column"""
        df = _get_session(session_id)
        
        try:
            result_series = transform_service.fill_missing(df, column, method, value)
            df[column] = result_series
            _set_session(session_id, df)
            
            return {
                "success": True,
                "null_count": int(result_series.isna().sum())
            }
        except Exception as e:
            raise HTTPException(400, str(e))
    
    
    @app.post("/session/{session_id}/filter")
    def filter_rows(
        session_id: str,
        filters: List[FilterSpec],
        create_new_session: bool = False
    ):
        """Filter rows based on conditions"""
        df = _get_session(session_id)
        
        try:
            result = df
            for filter_spec in filters:
                result = transform_service.filter_rows(
                    result,
                    filter_spec.column,
                    filter_spec.operator,
                    filter_spec.value
                )
            
            if create_new_session:
                new_session_id = str(uuid4())
                _set_session(new_session_id, result, metadata={"parent_session": session_id})
                return {
                    "success": True,
                    "session_id": new_session_id,
                    "n_rows": len(result)
                }
            else:
                _set_session(session_id, result)
                return {
                    "success": True,
                    "session_id": session_id,
                    "n_rows": len(result)
                }
        except Exception as e:
            raise HTTPException(400, str(e))


# ===================================================
# API ENDPOINTS - Data Export
# ===================================================
@app.get("/session/{session_id}/export")
def export_data(session_id: str, format: str = "csv"):
    """Export current session data"""
    df = _get_session(session_id)
    
    if format == "csv":
        stream = StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=export_{session_id}.csv"}
        )
    elif format == "json":
        stream = StringIO()
        df.to_json(stream, orient="records", indent=2)
        stream.seek(0)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=export_{session_id}.json"}
        )
    else:
        raise HTTPException(400, "Unsupported format. Use 'csv' or 'json'")


# ===================================================
# Startup Message
# ===================================================
@app.on_event("startup")
async def startup_event():
    import os

    port = os.getenv("PORT", "8000")

    print("=" * 70)
    print("🚀 AI Data Lab v5.0 - Complete Analytics Platform")
    print("=" * 70)
    print("Features Loaded:")
    print(f"  ✅ Statistical Analysis (Minitab-level)")
    print(f"  ✅ Quality Control (Control Charts, Process Capability)")
    print(f"  ✅ Advanced Analytics (PCA, Clustering, Time Series)")
    print(f"  {'✅' if TRANSFORMS_AVAILABLE else '❌'} Transform Engine (60+ transforms)")
    print(f"  {'✅' if TRANSFORM_SERVICE_AVAILABLE else '❌'} Table Operations (group, pivot, merge)")
    print("=" * 70)
    print(f"📚 API Docs: http://0.0.0.0:{port}/docs")
    print("=" * 70)


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
