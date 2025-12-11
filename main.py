"""
AI Data Lab Backend v4.0
Complete Minitab-level Statistical Engine
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Literal
from uuid import uuid4
from io import BytesIO
import pandas as pd
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

try:
    from transformers.registry import registry
    from transformers.base import TransformError
    TRANSFORMS_AVAILABLE = True
except ImportError:
    TRANSFORMS_AVAILABLE = False

app = FastAPI(title="AI Data Lab", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
SESSIONS: Dict[str, pd.DataFrame] = {}

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: str
    missing_pct: float
    unique_count: Optional[int] = None

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
    schema: List[ColumnInfo]
    descriptives: List[DescriptiveStats]

class CorrelationResponse(BaseModel):
    matrix: Dict[str, Dict[str, float]]
    method: str = "pearson"

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

class AnalysisResponse(BaseModel):
    session_id: str
    correlation: Optional[CorrelationResponse] = None
    tests: List[TestResult] = []
    regression: Optional[RegressionResult] = None
    normality_tests: List[NormalityTest] = []

def _infer_role(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return "categorical" if unique_ratio < 0.2 else "text"

def _load_dataframe(file: UploadFile) -> pd.DataFrame:
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
            raise HTTPException(400, "Unsupported file type")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse: {e}")
    if df.empty:
        raise HTTPException(400, "Empty file")
    return df

def _build_profile(df: pd.DataFrame, session_id: str) -> ProfileResponse:
    cols, descriptives = [], []
    for col in df.columns:
        s = df[col]
        role = _infer_role(s)
        missing_pct = float(s.isna().mean() * 100)
        cols.append(ColumnInfo(name=col, dtype=str(s.dtype), role=role, missing_pct=round(missing_pct, 2), unique_count=int(s.nunique())))
        if role == "numeric":
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            descriptives.append(DescriptiveStats(
                column=col, count=int(desc["count"]),
                mean=float(desc["mean"]) if not np.isnan(desc["mean"]) else None,
                std=float(desc["std"]) if not np.isnan(desc["std"]) else None,
                min=float(desc["min"]) if not np.isnan(desc["min"]) else None,
                q25=float(desc["25%"]) if not np.isnan(desc["25%"]) else None,
                median=float(desc["50%"]) if not np.isnan(desc["50%"]) else None,
                q75=float(desc["75%"]) if not np.isnan(desc["75%"]) else None,
                max=float(desc["max"]) if not np.isnan(desc["max"]) else None,
                skewness=float(s.skew()) if not np.isnan(s.skew()) else None,
                kurtosis=float(s.kurtosis()) if not np.isnan(s.kurtosis()) else None
            ))
    return ProfileResponse(session_id=session_id, n_rows=len(df), n_cols=df.shape[1], columns=cols, schema=cols, descriptives=descriptives)

def _build_correlation(df: pd.DataFrame, method: str = "pearson") -> Optional[CorrelationResponse]:
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr(method=method)
    corr_dict = {col: {idx: float(val) for idx, val in corr[col].items()} for col in corr.columns}
    return CorrelationResponse(matrix=corr_dict, method=method)

def _normality_tests(series: pd.Series) -> List[NormalityTest]:
    tests = []
    data = series.dropna()
    if len(data) < 3:
        return tests
    try:
        stat, p = shapiro(data)
        tests.append(NormalityTest(test_name="Shapiro-Wilk", statistic=float(stat), p_value=float(p), is_normal=p > 0.05, interpretation=f"Data {'appears' if p > 0.05 else 'does not appear'} normally distributed"))
    except:
        pass
    try:
        result = anderson(data)
        tests.append(NormalityTest(test_name="Anderson-Darling", statistic=float(result.statistic), p_value=0.0, is_normal=result.statistic < result.critical_values[2], interpretation=f"Statistic: {result.statistic:.4f}"))
    except:
        pass
    return tests

def _auto_tests(df: pd.DataFrame, profile: ProfileResponse) -> List[TestResult]:
    tests = []
    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    cat_cols = [c.name for c in profile.columns if c.role == "categorical"]
    if not numeric_cols or not cat_cols:
        return tests
    target, group_col = numeric_cols[0], cat_cols[0]
    data = pd.DataFrame({"target": df[target], "group": df[group_col].astype("category")}).dropna()
    if data["group"].nunique() < 2:
        return tests
    groups = [g["target"].values for _, g in data.groupby("group")]
    try:
        if len(groups) == 2:
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            pooled_std = np.sqrt(((len(groups[0])-1)*np.var(groups[0]) + (len(groups[1])-1)*np.var(groups[1])) / (len(groups[0])+len(groups[1])-2))
            cohens_d = (np.mean(groups[0]) - np.mean(groups[1])) / pooled_std if pooled_std > 0 else 0
            tests.append(TestResult(test_type="t-test (Welch)", target=target, group_col=group_col, p_value=float(p), statistic=float(stat), effect_size=float(cohens_d), interpretation=f"{'Significant' if p < 0.05 else 'No significant'} difference"))
            stat_u, p_u = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            tests.append(TestResult(test_type="Mann-Whitney U", target=target, group_col=group_col, p_value=float(p_u), statistic=float(stat_u), interpretation=f"{'Significant' if p_u < 0.05 else 'No significant'} difference"))
        elif len(groups) > 2:
            stat_f, p_f = stats.f_oneway(*groups)
            tests.append(TestResult(test_type="ANOVA", target=target, group_col=group_col, p_value=float(p_f), statistic=float(stat_f), df=float(len(groups) - 1), interpretation=f"{'Significant' if p_f < 0.05 else 'No significant'} difference"))
            if p_f < 0.05:
                try:
                    tukey = pairwise_tukeyhsd(data["target"], data["group"])
                    tests[-1].post_hoc = {"tukey": str(tukey)}
                except:
                    pass
            stat_h, p_h = kruskal(*groups)
            tests.append(TestResult(test_type="Kruskal-Wallis", target=target, group_col=group_col, p_value=float(p_h), statistic=float(stat_h), interpretation=f"{'Significant' if p_h < 0.05 else 'No significant'} difference"))
    except:
        pass
    return tests
def _calculate_regression_diagnostics(model, X, y) -> Dict[str, Any]:
    diagnostics = {'aic': float(model.aic), 'bic': float(model.bic), 'durbin_watson': float(sm.stats.durbin_watson(model.resid)), 'residuals': model.resid.tolist(), 'fitted_values': model.fittedvalues.tolist()}
    influence = model.get_influence()
    diagnostics['leverage'] = influence.hat_matrix_diag.tolist()
    diagnostics['cooks_distance'] = influence.cooks_distance[0].tolist()
    try:
        if X.shape[1] > 1:
            vif_data = {col: float(variance_inflation_factor(X.values, i)) if not np.isinf(variance_inflation_factor(X.values, i)) else None for i, col in enumerate(X.columns) if col != 'const'}
            diagnostics['vif'] = vif_data
    except:
        diagnostics['vif'] = None
    try:
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        diagnostics['heteroscedasticity_test'] = {'breusch_pagan': {'statistic': float(bp_test[0]), 'p_value': float(bp_test[1]), 'heteroscedastic': bp_test[1] < 0.05}}
    except:
        diagnostics['heteroscedasticity_test'] = None
    return diagnostics

def _auto_regression(df: pd.DataFrame, profile: ProfileResponse) -> Optional[RegressionResult]:
    numeric_cols = [c.name for c in profile.columns if c.role == "numeric"]
    if len(numeric_cols) < 2:
        return None
    target, predictors = numeric_cols[0], numeric_cols[1:]
    data = df[numeric_cols].dropna()
    if len(data) < 10:
        return None
    y, X = data[target], sm.add_constant(data[predictors])
    try:
        model = sm.OLS(y, X).fit()
        diagnostics = _calculate_regression_diagnostics(model, X, y)
    except:
        return None
    return RegressionResult(
        target=target, predictors=predictors, n_observations=int(model.nobs), r_squared=float(model.rsquared), adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue), f_pvalue=float(model.f_pvalue), aic=diagnostics['aic'], bic=diagnostics['bic'],
        coefficients={n: float(v) for n, v in model.params.items()}, std_errors={n: float(v) for n, v in model.bse.items()},
        t_values={n: float(v) for n, v in model.tvalues.items()}, p_values={n: float(v) for n, v in model.pvalues.items()},
        confidence_intervals={n: [float(model.conf_int().loc[n, 0]), float(model.conf_int().loc[n, 1])] for n in model.params.index},
        residuals=diagnostics['residuals'], fitted_values=diagnostics['fitted_values'], leverage=diagnostics['leverage'],
        cooks_distance=diagnostics['cooks_distance'], vif=diagnostics.get('vif'), heteroscedasticity_test=diagnostics.get('heteroscedasticity_test'),
        durbin_watson=diagnostics['durbin_watson']
    )

def _check_control_rules(points: List[float], center: float, sigma: float) -> List[List[str]]:
    violations = [[] for _ in points]
    for i in range(len(points)):
        if abs(points[i] - center) > 3 * sigma:
            violations[i].append("Rule 1: Beyond 3σ")
        if i >= 2:
            recent = [(points[j] - center) / sigma for j in range(i-2, i+1)]
            if sum(1 for x in recent if x > 2) >= 2:
                violations[i].append("Rule 2: 2/3 > 2σ")
            if sum(1 for x in recent if x < -2) >= 2:
                violations[i].append("Rule 2: 2/3 < -2σ")
        if i >= 4:
            recent = [(points[j] - center) / sigma for j in range(i-4, i+1)]
            if sum(1 for x in recent if x > 1) >= 4:
                violations[i].append("Rule 3: 4/5 > 1σ")
            if sum(1 for x in recent if x < -1) >= 4:
                violations[i].append("Rule 3: 4/5 < -1σ")
        if i >= 7:
            recent = [points[j] - center for j in range(i-7, i+1)]
            if all(x > 0 for x in recent) or all(x < 0 for x in recent):
                violations[i].append("Rule 4: 8 consecutive")
    return violations

def _create_control_chart(data: pd.Series, chart_type: str, subgroup_size: Optional[int] = None) -> ControlChartResult:
    if chart_type == "xbar":
        subgroup_size = subgroup_size or min(5, len(data) // 10)
        n_subgroups = len(data) // subgroup_size
        subgroups = [data.iloc[i*subgroup_size:(i+1)*subgroup_size].mean() for i in range(n_subgroups)]
        xbar_bar = np.mean(subgroups)
        ranges = [data.iloc[i*subgroup_size:(i+1)*subgroup_size].max() - data.iloc[i*subgroup_size:(i+1)*subgroup_size].min() for i in range(n_subgroups)]
        r_bar = np.mean(ranges)
        d2, a2 = {2:1.128,3:1.693,4:2.059,5:2.326,6:2.534,7:2.704,8:2.847,9:2.970,10:3.078}.get(subgroup_size, 2.326), {2:1.880,3:1.023,4:0.729,5:0.577,6:0.483,7:0.419,8:0.373,9:0.337,10:0.308}.get(subgroup_size, 0.577)
        sigma, ucl, lcl = r_bar / d2, xbar_bar + a2 * r_bar, xbar_bar - a2 * r_bar
        violations = _check_control_rules(subgroups, xbar_bar, sigma)
        points = [ControlChartPoint(index=i, value=float(val), ucl=float(ucl), lcl=float(lcl), center=float(xbar_bar), out_of_control=(val > ucl or val < lcl), rule_violations=violations[i]) for i, val in enumerate(subgroups)]
        return ControlChartResult(chart_type="X-bar", data_points=points, center_line=float(xbar_bar), ucl=float(ucl), lcl=float(lcl), sigma=float(sigma), summary={"n_subgroups": n_subgroups, "subgroup_size": subgroup_size, "out_of_control": sum(1 for p in points if p.out_of_control)})
    elif chart_type == "i":
        values = data.dropna().values
        x_bar, mr = np.mean(values), np.abs(np.diff(values))
        mr_bar, sigma = np.mean(mr), np.mean(mr) / 1.128
        ucl, lcl = x_bar + 3 * sigma, x_bar - 3 * sigma
        violations = _check_control_rules(values.tolist(), x_bar, sigma)
        points = [ControlChartPoint(index=i, value=float(val), ucl=float(ucl), lcl=float(lcl), center=float(x_bar), out_of_control=(val > ucl or val < lcl), rule_violations=violations[i]) for i, val in enumerate(values)]
        return ControlChartResult(chart_type="I-chart", data_points=points, center_line=float(x_bar), ucl=float(ucl), lcl=float(lcl), sigma=float(sigma), summary={"n_points": len(values), "mr_bar": float(mr_bar), "out_of_control": sum(1 for p in points if p.out_of_control)})
    elif chart_type == "p":
        subgroup_size = subgroup_size or min(50, len(data) // 10)
        n_subgroups = len(data) // subgroup_size
        proportions = [data.iloc[i*subgroup_size:(i+1)*subgroup_size].mean() for i in range(n_subgroups)]
        p_bar, sigma_p = np.mean(proportions), np.sqrt(np.mean(proportions) * (1 - np.mean(proportions)) / subgroup_size)
        ucl, lcl = p_bar + 3 * sigma_p, max(0, p_bar - 3 * sigma_p)
        points = [ControlChartPoint(index=i, value=float(val), ucl=float(ucl), lcl=float(lcl), center=float(p_bar), out_of_control=(val > ucl or val < lcl), rule_violations=[]) for i, val in enumerate(proportions)]
        return ControlChartResult(chart_type="P-chart", data_points=points, center_line=float(p_bar), ucl=float(ucl), lcl=float(lcl), sigma=float(sigma_p), summary={"n_subgroups": n_subgroups, "subgroup_size": subgroup_size, "out_of_control": sum(1 for p in points if p.out_of_control)})
    raise HTTPException(400, f"Chart type '{chart_type}' not implemented")

def _calculate_process_capability(data: pd.Series, usl: float, lsl: float, target: Optional[float] = None) -> ProcessCapabilityResult:
    values = data.dropna().values
    if len(values) < 30:
        raise HTTPException(400, "Need at least 30 data points")
    mean, std = np.mean(values), np.std(values, ddof=1)
    target = target or (usl + lsl) / 2
    cp = (usl - lsl) / (6 * std)
    cpu, cpl, cpk = (usl - mean) / (3 * std), (mean - lsl) / (3 * std), min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    cpm = (usl - lsl) / (6 * np.sqrt(std**2 + (mean - target)**2))
    z_min = min((usl - mean) / std, (mean - lsl) / std)
    dpmo = (1 - stats.norm.cdf(z_min)) * 1_000_000
    expected = (stats.norm.cdf((usl - mean) / std) - stats.norm.cdf(-(mean - lsl) / std)) * 100
    interp = "Excellent (6σ)" if cpk >= 2.0 else "Adequate" if cpk >= 1.33 else "Marginal" if cpk >= 1.0 else "Poor"
    return ProcessCapabilityResult(cp=float(cp), cpk=float(cpk), pp=float(cp), ppk=float(cpk), cpm=float(cpm), sigma_level=float(z_min), dpmo=float(dpmo), expected_within_spec=float(expected), interpretation=interp, specifications={"usl": float(usl), "lsl": float(lsl), "target": float(target), "mean": float(mean), "std": float(std)})

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "4.0.0", "capabilities": "Complete Minitab-level", "transforms": TRANSFORMS_AVAILABLE}

# NOTE: /upload endpoint is in Part 3 (with sample_rows support)
# DO NOT add @app.post("/upload") here - it's been moved to Part 3

@app.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def run_analysis(session_id: str, correlation_method: str = "pearson"):
    df = SESSIONS.get(session_id)
    if df is None:  # FIXED: Was "if not df is not None" (double negative)
        raise HTTPException(404, "Session not found")
    profile = _build_profile(df, session_id)
    normality_tests = []
    for col_info in profile.columns:
        if col_info.role == "numeric":
            normality_tests.extend(_normality_tests(df[col_info.name]))
    return AnalysisResponse(session_id=session_id, correlation=_build_correlation(df, correlation_method), tests=_auto_tests(df, profile), regression=_auto_regression(df, profile), normality_tests=normality_tests)

@app.post("/advanced-analysis/{session_id}", response_model=AdvancedAnalysisResponse)
async def run_advanced_analysis(session_id: str, request: AdvancedAnalysisRequest):
    df = SESSIONS.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found")
    result = {}
    if request.analysis_type == "normality":
        result = {"tests": [t.dict() for t in _normality_tests(df[request.target])]} if request.target else {}
    elif request.analysis_type == "variance_test":
        if request.target and request.group_col:
            data = df[[request.target, request.group_col]].dropna()
            groups = [g[request.target].values for _, g in data.groupby(request.group_col)]
            stat_l, p_l = levene(*groups)
            stat_b, p_b = bartlett(*groups)
            result = {"levene": {"statistic": float(stat_l), "p_value": float(p_l)}, "bartlett": {"statistic": float(stat_b), "p_value": float(p_b)}}
    elif request.analysis_type == "time_series" and request.target:
        series, period = df[request.target].dropna(), request.params.get("period", 12) if request.params else 12
        decomp = seasonal_decompose(series, model='additive', period=period)
        result = {"trend": decomp.trend.dropna().tolist(), "seasonal": decomp.seasonal.dropna().tolist(), "residual": decomp.resid.dropna().tolist(), "period": period}
    elif request.analysis_type == "pca":
        cols = request.columns or df.select_dtypes(include=[np.number]).columns.tolist()
        data = StandardScaler().fit_transform(df[cols].dropna())
        n_comp = request.params.get("n_components", min(len(cols), 5)) if request.params else min(len(cols), 5)
        pca = PCA(n_components=n_comp).fit(data)
        result = {"n_components": n_comp, "explained_variance": pca.explained_variance_ratio_.tolist(), "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(), "components": pca.components_.tolist()}
    elif request.analysis_type == "cluster":
        cols = request.columns or df.select_dtypes(include=[np.number]).columns.tolist()
        data = StandardScaler().fit_transform(df[cols].dropna())
        n_clust = request.params.get("n_clusters", 3) if request.params else 3
        kmeans = KMeans(n_clusters=n_clust, random_state=42).fit(data)
        result = {"n_clusters": n_clust, "labels": kmeans.labels_.tolist(), "centroids": kmeans.cluster_centers_.tolist(), "inertia": float(kmeans.inertia_)}
    return AdvancedAnalysisResponse(session_id=session_id, analysis_type=request.analysis_type, result=result)

@app.post("/control-chart/{session_id}", response_model=ControlChartResult)
async def create_control_chart(session_id: str, request: ControlChartRequest):
    df = SESSIONS.get(session_id)
    if df is None or request.column not in df.columns:
        raise HTTPException(404, "Session/column not found")
    data = df[request.column].dropna()
    if len(data) < 10:
        raise HTTPException(400, "Need 10+ data points")
    return _create_control_chart(data, request.chart_type, request.subgroup_size)

@app.post("/process-capability/{session_id}", response_model=ProcessCapabilityResult)
async def calculate_process_capability(session_id: str, request: ProcessCapabilityRequest):
    df = SESSIONS.get(session_id)
    if df is None or request.column not in df.columns:
        raise HTTPException(404, "Session/column not found")
    return _calculate_process_capability(df[request.column], request.usl, request.lsl, request.target)

@app.post("/regression/{session_id}", response_model=RegressionResult)
async def run_regression(session_id: str, request: RegressionRequest):
    df = SESSIONS.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found")
    if request.target not in df.columns:
        raise HTTPException(400, f"Target not found")
    for pred in request.predictors:
        if pred not in df.columns:
            raise HTTPException(400, f"Predictor '{pred}' not found")
    data = df[[request.target] + request.predictors].dropna()
    if len(data) < len(request.predictors) + 10:
        raise HTTPException(400, f"Need {len(request.predictors) + 10}+ observations")
    y, X = data[request.target], sm.add_constant(data[request.predictors])
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        raise HTTPException(400, f"Regression failed: {e}")
    diag = _calculate_regression_diagnostics(model, X, y) if request.include_diagnostics else {'aic': model.aic, 'bic': model.bic, 'residuals': [], 'fitted_values': [], 'leverage': [], 'cooks_distance': []}
    return RegressionResult(target=request.target, predictors=request.predictors, n_observations=int(model.nobs), r_squared=float(model.rsquared), adj_r_squared=float(model.rsquared_adj), f_statistic=float(model.fvalue), f_pvalue=float(model.f_pvalue), aic=diag['aic'], bic=diag['bic'], coefficients={n: float(v) for n, v in model.params.items()}, std_errors={n: float(v) for n, v in model.bse.items()}, t_values={n: float(v) for n, v in model.tvalues.items()}, p_values={n: float(v) for n, v in model.pvalues.items()}, confidence_intervals={n: [float(model.conf_int().loc[n, 0]), float(model.conf_int().loc[n, 1])] for n in model.params.index}, residuals=diag.get('residuals', []), fitted_values=diag.get('fitted_values', []), leverage=diag.get('leverage', []), cooks_distance=diag.get('cooks_distance', []), vif=diag.get('vif'), heteroscedasticity_test=diag.get('heteroscedasticity_test'), durbin_watson=diag.get('durbin_watson'))

@app.post("/transform/{session_id}")
async def apply_transform(session_id: str, request: TransformRequest):
    if not TRANSFORMS_AVAILABLE:
        raise HTTPException(501, "Transforms not available")
    df = SESSIONS.get(session_id)
    if df is None or request.column not in df.columns:
        raise HTTPException(404, "Session/column not found")
    try:
        transformer = registry.create(request.transform_type, request.params)
        result = transformer.transform(df[request.column], request.column)
        return {"success": True, "preview": result.head(20).tolist(), "metadata": transformer.get_metadata(df[request.column], result)}
    except TransformError as e:
        raise HTTPException(400, f"Transform failed: {e.message}")

@app.get("/transform/{session_id}/suggest")
async def suggest_transforms(session_id: str, column: str):
    if not TRANSFORMS_AVAILABLE:
        raise HTTPException(501, "Transforms not available")
    df = SESSIONS.get(session_id)
    if df is None or column not in df.columns:
        raise HTTPException(404, "Session/column not found")
    return {"column": column, "suggestions": registry.suggest_transforms(df[column], column)}

@app.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    df = SESSIONS.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "n_rows": len(df), "n_cols": len(df.columns), "columns": df.columns.tolist(), "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {"message": "Deleted"}
    raise HTTPException(404, "Session not found")

@app.get("/")
def root():
    return {"name": "AI Data Lab", "version": "4.0.0", "docs": "/docs", "capabilities": ["Descriptive Stats", "Correlation", "Hypothesis Tests", "Normality Tests", "Variance Tests", "Full Regression Diagnostics", "Control Charts (X-bar, I, P)", "Process Capability (Cp, Cpk, Pp, Ppk, Cpm)", "Time Series", "PCA", "Clustering", "Tukey HSD", "Transforms (60+)" if TRANSFORMS_AVAILABLE else "Transforms (not loaded)"], "transforms": TRANSFORMS_AVAILABLE}
# =============================================================================
# MISSING ENDPOINTS FOR FRONTEND COMPATIBILITY
# Add these AFTER Part 2, BEFORE the final root() endpoint
# =============================================================================

# Additional Pydantic Models for Query Endpoint
class QueryRequest(BaseModel):
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

@app.get("/sample/{session_id}")
async def get_sample(session_id: str, max_rows: int = 100):
    """Return sample rows from dataset for UI preview"""
    df = SESSIONS.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found")
    
    # Get sample and convert NaN to None for JSON serialization
    sample_df = df.head(max_rows)
    sample_rows = sample_df.replace({np.nan: None, pd.NaT: None}).to_dict(orient="records")
    
    return {
        "session_id": session_id,
        "sample_rows": sample_rows,
        "total_rows": len(df),
        "returned_rows": len(sample_rows)
    }

@app.get("/schema/{session_id}")
async def get_schema(session_id: str):
    """Return column schema for a session"""
    df = SESSIONS.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found")
    
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
    """
    df = SESSIONS.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found")
    
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
                
                return QueryResponse(success=True, result={"data": [agg_result], "columns": list(agg_result.keys()), "row_count": 1})
        
        elif request.operation == "distinct":
            if request.group_by:
                result_df = result_df[request.group_by].drop_duplicates()
            else:
                result_df = result_df.drop_duplicates()
        
        elif request.operation == "describe":
            desc = result_df.describe(include='all').to_dict()
            return QueryResponse(success=True, result={"data": desc, "columns": list(desc.keys()), "row_count": len(desc)})
        
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

# =============================================================================
# UPDATE THE /upload ENDPOINT TO INCLUDE sample_rows
# Replace the existing @app.post("/upload") with this version:
# =============================================================================

@app.post("/upload", response_model=ProfileResponse)
async def upload_file_with_sample(file: UploadFile = File(...)):
    """Upload file with sample rows included in response"""
    df = _load_dataframe(file).dropna(axis=1, how="all")
    session_id = str(uuid4())
    SESSIONS[session_id] = df
    
    # Get profile
    profile = _build_profile(df, session_id)
    
    # Add sample rows to response
    sample_rows = df.head(100).replace({np.nan: None, pd.NaT: None}).to_dict(orient="records")
    
    # Create response dict and add sample_rows
    profile_dict = profile.dict()
    profile_dict["sample_rows"] = sample_rows
    
    return profile_dict