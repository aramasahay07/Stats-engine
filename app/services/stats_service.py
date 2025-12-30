"""
Enhanced Statistical Analysis Service
Provides comprehensive statistical analysis capabilities similar to Minitab.

Features:
- Descriptive Statistics
- Hypothesis Testing (t-tests, ANOVA, chi-square)
- Correlation & Regression
- Nonparametric Tests
- Time Series Analysis
- Normality Tests
- Smart caching for performance

Author: Stats Engine v2
"""

from __future__ import annotations
import hashlib
import json
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from app.config import settings
from app.db import registry
from app.engine.duckdb_engine import DuckDBEngine


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _fingerprint_file(path: Path) -> str:
    """
    Compute SHA256 checksum of a local file.
    Used to detect stale cached parquet files.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _hash_spec(
    dataset_id: str,
    analysis: str,
    params: dict,
    parquet_ref: str,
    parquet_sha: str,
    pipeline_hash: str,
    engine_version: str,
) -> str:
    """
    Create a unique hash for caching purposes.
    Same analysis on same data always produces same hash.
    """
    payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "analysis": analysis,
            "params": params,
            "parquet_ref": parquet_ref,
            "parquet_sha": parquet_sha,
            "pipeline_hash": pipeline_hash,
            "engine_version": engine_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quote(c: str) -> str:
    """Safely quote column names for SQL queries."""
    return f'"{c}"'


async def _get_parquet_local(user_id: str, dataset_id: str) -> Path:
    """
    Ensure the dataset parquet exists locally.

    Source of truth is Supabase Storage (datasets.parquet_ref).
    We download on-demand using the service role key (server-side)
    and cache under DATA_DIR.
    """

    p = (
        Path(settings.data_dir)
        / "datasets"
        / user_id
        / dataset_id
        / "data.parquet"
    )

    # ✅ Fast path: already cached locally— but verify integrity
    if p.exists():
        meta = await registry.fetchrow(
            """
            SELECT profile_json
            FROM datasets
            WHERE dataset_id = $1
              AND user_id = $2
            """,
            dataset_id,
            user_id,
        )

        expected_sha = None
        if meta and meta["profile_json"]:
            expected_sha = meta["profile_json"].get("parquet_sha")

        # If checksum matches → safe to use
        if expected_sha and _fingerprint_file(p) == expected_sha:
            return p

        # Otherwise stale parquet → delete and re-download
        p.unlink()

    # 🔍 Fetch parquet reference from DB
    row = await registry.fetchrow(
        """
        SELECT parquet_ref
        FROM datasets
        WHERE dataset_id = $1
          AND user_id = $2
        """,
        dataset_id,
        user_id,
    )

    if not row or not row["parquet_ref"]:
        raise FileNotFoundError(
            "Parquet artifact not found. Dataset parquet_ref is missing (still building?)."
        )

    # ⬇️ Download from Supabase Storage
    from app.services.storage_supabase import SupabaseStorage

    storage = SupabaseStorage()

    p.parent.mkdir(parents=True, exist_ok=True)

    file_bytes = await storage.download(row["parquet_ref"])
    p.write_bytes(file_bytes)

    return p


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

async def try_cache_get(
    user_id: str,
    dataset_id: str,
    analysis_hash: str
) -> Optional[Dict[str, Any]]:
    """
    Try to retrieve cached analysis results.
    Returns None if not found.
    """
    row = await registry.fetchrow(
        """
        SELECT result_json
        FROM analysis_results
        WHERE dataset_id = $1
          AND user_id = $2
          AND analysis_hash = $3
        """,
        dataset_id,
        user_id,
        analysis_hash,
    )

    # asyncpg returns JSONB as dict already
    return dict(row["result_json"]) if row else None


async def cache_put(
    user_id: str,
    dataset_id: str,
    analysis_hash: str,
    spec_json: Dict[str, Any],
    result_json: Dict[str, Any],
):
    """
    Store analysis results in cache for future reuse.
    """

    # ✅ Serialize Python dicts → JSON strings for asyncpg
    spec_payload = json.dumps(spec_json)
    result_payload = json.dumps(result_json)

    await registry.execute(
        """
        INSERT INTO analysis_results (
            id,
            dataset_id,
            user_id,
            analysis_hash,
            spec_json,
            result_json
        )
        VALUES (
            gen_random_uuid(),
            $1,
            $2,
            $3,
            $4::jsonb,
            $5::jsonb
        )
        """,
        dataset_id,
        user_id,
        analysis_hash,
        spec_payload,
        result_payload,
    )


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

async def descriptives(user_id: str, dataset_id: str, columns: List[str]) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for numeric columns.
    
    Returns: count, mean, std, min, max for each column
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    out: Dict[str, Any] = {}
    meta = await registry.fetchrow(
    """
    SELECT profile_json
    FROM datasets
    WHERE dataset_id = $1
      AND user_id = $2
    """,
    dataset_id,
    user_id,
)

    profile = meta["profile_json"] if meta else None
    numeric_summary = profile.get("numeric_summary") if profile else None

    for c in columns:
        # ------------------------------------------------------------
        # Reuse profile numeric summary if available (single source of truth)
        # ------------------------------------------------------------
        if numeric_summary and c in numeric_summary:
            out[c] = numeric_summary[c]
            continue

        qc = _quote(c)

        q = f"""
            SELECT
                COUNT(*) AS rows_total,
                COUNT({qc}) AS rows_used,
                COUNT(*) - COUNT({qc}) AS rows_missing,

                AVG({qc}) AS mean,
                STDDEV_SAMP({qc}) AS std,

                MIN({qc}) AS min,
                quantile_cont({qc}, 0.25) AS q1,
                quantile_cont({qc}, 0.50) AS median,
                quantile_cont({qc}, 0.75) AS q3,
                MAX({qc}) AS max,

                quantile_cont({qc}, 0.10) AS p10,
                quantile_cont({qc}, 0.90) AS p90
            FROM {view}
        """
        row = con.execute(q).fetchone()

        out[c] = {
            "rows_total": int(row[0]),
            "rows_used": int(row[1]),
            "rows_missing": int(row[2]),

            "mean": float(row[3]) if row[3] is not None else None,
            "std": float(row[4]) if row[4] is not None else None,

            "min": float(row[5]) if row[5] is not None else None,
            "q1": float(row[6]) if row[6] is not None else None,
            "median": float(row[7]) if row[7] is not None else None,
            "q3": float(row[8]) if row[8] is not None else None,
            "max": float(row[9]) if row[9] is not None else None,

            "p10": float(row[10]) if row[10] is not None else None,
            "p90": float(row[11]) if row[11] is not None else None,
        }

    con.close()
    return out


async def detailed_descriptives(user_id: str, dataset_id: str, columns: List[str]) -> Dict[str, Any]:
    """
    Extended descriptive statistics including quartiles, skewness, kurtosis.
    
    Returns: count, mean, std, min, q1, median, q3, max, skewness, kurtosis
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    out: Dict[str, Any] = {}
    for c in columns:
        q = f"SELECT {_quote(c)} FROM {view} WHERE {_quote(c)} IS NOT NULL"
        df = con.execute(q).fetchdf()
        
        if len(df) > 0:
            data = df[c].to_numpy()
            out[c] = {
                "count": int(len(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1)),
                "min": float(np.min(data)),
                "q1": float(np.percentile(data, 25)),
                "median": float(np.median(data)),
                "q3": float(np.percentile(data, 75)),
                "max": float(np.max(data)),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data))
            }
        else:
            out[c] = None
    
    con.close()
    return out


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

async def correlation(
    user_id: str,
    dataset_id: str,
    x: str,
    y: str,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Calculate correlation between two variables.
    
    Methods:
    - pearson: Linear correlation (default)
    - spearman: Rank correlation (nonparametric)
    - kendall: Tau correlation (nonparametric)
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(x)} AS x, {_quote(y)} AS y FROM {view} WHERE {_quote(x)} IS NOT NULL AND {_quote(y)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    if len(df) < 3:
        raise ValueError("Need at least 3 observations for correlation")
    
    x_data = df["x"].to_numpy()
    y_data = df["y"].to_numpy()
    
    if method == "pearson":
        corr, p_value = stats.pearsonr(x_data, y_data)
    elif method == "spearman":
        corr, p_value = stats.spearmanr(x_data, y_data)
    elif method == "kendall":
        corr, p_value = stats.kendalltau(x_data, y_data)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return {
        "correlation": float(corr),
        "p_value": float(p_value),
        "method": method,
        "n": int(len(df))
    }


async def correlation_matrix(
    user_id: str,
    dataset_id: str,
    columns: List[str],
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Calculate correlation matrix for multiple variables.
    
    Returns correlation matrix and p-values.
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    cols = ", ".join([_quote(c) for c in columns])
    df = con.execute(f"SELECT {cols} FROM {view}").fetchdf()
    con.close()
    
    df = df.dropna()
    
    if len(df) < 3:
        raise ValueError("Need at least 3 observations for correlation matrix")
    
    if method == "pearson":
        corr_matrix = df.corr(method='pearson')
    elif method == "spearman":
        corr_matrix = df.corr(method='spearman')
    elif method == "kendall":
        corr_matrix = df.corr(method='kendall')
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "method": method,
        "n": int(len(df)),
        "variables": columns
    }


# ============================================================================
# T-TESTS
# ============================================================================

async def ttest_1samp(user_id: str, dataset_id: str, column: str, pop_mean: float) -> Dict[str, Any]:
    """
    One-sample t-test: Test if sample mean differs from population mean.
    
    H0: sample mean = pop_mean
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(column)} AS col FROM {view} WHERE {_quote(column)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    data = df["col"].to_numpy()
    res = stats.ttest_1samp(data, pop_mean)
    
    return {
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "sample_mean": float(np.mean(data)),
        "pop_mean": float(pop_mean),
        "n": int(len(data))
    }


async def ttest_2samp(user_id: str, dataset_id: str, x: str, y: str) -> Dict[str, Any]:
    """
    Independent samples t-test: Compare means of two groups.
    
    H0: mean(x) = mean(y)
    Uses Welch's t-test (doesn't assume equal variances)
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(x)} AS x, {_quote(y)} AS y FROM {view} WHERE {_quote(x)} IS NOT NULL AND {_quote(y)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    res = stats.ttest_ind(
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        equal_var=False,
        nan_policy="omit"
    )
    
    return {
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_x": float(df["x"].mean()),
        "mean_y": float(df["y"].mean()),
        "n_x": int(df["x"].notna().sum()),
        "n_y": int(df["y"].notna().sum())
    }


async def paired_ttest(user_id: str, dataset_id: str, before: str, after: str) -> Dict[str, Any]:
    """
    Paired t-test: Compare two related samples.
    
    H0: mean(before) = mean(after)
    Use when same subjects measured twice.
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(before)} AS before, {_quote(after)} AS after FROM {view} WHERE {_quote(before)} IS NOT NULL AND {_quote(after)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    before_data = df["before"].to_numpy()
    after_data = df["after"].to_numpy()
    
    res = stats.ttest_rel(before_data, after_data)
    
    return {
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_before": float(np.mean(before_data)),
        "mean_after": float(np.mean(after_data)),
        "mean_difference": float(np.mean(after_data - before_data)),
        "n": int(len(df))
    }


# ============================================================================
# ANOVA
# ============================================================================

async def anova_oneway(user_id: str, dataset_id: str, value_col: str, group_col: str) -> Dict[str, Any]:
    """
    One-way ANOVA: Compare means across multiple groups.
    
    H0: All group means are equal
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(value_col)} AS v, {_quote(group_col)} AS g FROM {view} WHERE {_quote(value_col)} IS NOT NULL AND {_quote(group_col)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    groups = [grp["v"].to_numpy() for _, grp in df.groupby("g")]
    
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for ANOVA")
    
    fstat, p = stats.f_oneway(*groups)
    
    # Calculate group statistics
    group_stats = {}
    for name, grp in df.groupby("g"):
        group_stats[str(name)] = {
            "mean": float(grp["v"].mean()),
            "std": float(grp["v"].std()),
            "n": int(len(grp))
        }
    
    return {
        "f_stat": float(fstat),
        "p_value": float(p),
        "k_groups": int(len(groups)),
        "n_total": int(len(df)),
        "group_stats": group_stats
    }


# ============================================================================
# CHI-SQUARE TESTS
# ============================================================================

async def chi_square_goodness(
    user_id: str,
    dataset_id: str,
    observed_col: str,
    expected_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Chi-square goodness of fit test.
    
    H0: Observed frequencies match expected frequencies
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    if expected_col:
        df = con.execute(
            f"SELECT {_quote(observed_col)} AS obs, {_quote(expected_col)} AS exp FROM {view}"
        ).fetchdf()
        observed = df["obs"].to_numpy()
        expected = df["exp"].to_numpy()
    else:
        df = con.execute(f"SELECT {_quote(observed_col)} AS obs FROM {view}").fetchdf()
        observed = df["obs"].to_numpy()
        expected = None  # Uniform distribution assumed
    
    con.close()
    
    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    
    return {
        "chi2_stat": float(chi2),
        "p_value": float(p_value),
        "df": int(len(observed) - 1),
        "n": int(np.sum(observed))
    }


async def chi_square_independence(
    user_id: str,
    dataset_id: str,
    var1: str,
    var2: str
) -> Dict[str, Any]:
    """
    Chi-square test of independence.
    
    H0: Variables are independent (no association)
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(var1)} AS v1, {_quote(var2)} AS v2 FROM {view}"
    ).fetchdf()
    con.close()
    
    # Create contingency table
    contingency = pd.crosstab(df["v1"], df["v2"])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        "chi2_stat": float(chi2),
        "p_value": float(p_value),
        "df": int(dof),
        "contingency_table": contingency.to_dict(),
        "n": int(contingency.sum().sum())
    }


# ============================================================================
# NONPARAMETRIC TESTS
# ============================================================================

async def mann_whitney(user_id: str, dataset_id: str, x: str, y: str) -> Dict[str, Any]:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).
    
    Nonparametric alternative to independent samples t-test.
    H0: Distributions are equal
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(x)} AS x, {_quote(y)} AS y FROM {view} WHERE {_quote(x)} IS NOT NULL AND {_quote(y)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    u_stat, p_value = stats.mannwhitneyu(
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        alternative='two-sided'
    )
    
    return {
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "median_x": float(df["x"].median()),
        "median_y": float(df["y"].median()),
        "n_x": int(len(df["x"])),
        "n_y": int(len(df["y"]))
    }


async def wilcoxon_signed_rank(
    user_id: str,
    dataset_id: str,
    before: str,
    after: str
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test.
    
    Nonparametric alternative to paired t-test.
    H0: Median difference is zero
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(before)} AS before, {_quote(after)} AS after FROM {view} WHERE {_quote(before)} IS NOT NULL AND {_quote(after)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    stat, p_value = stats.wilcoxon(
        df["before"].to_numpy(),
        df["after"].to_numpy()
    )
    
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "median_before": float(df["before"].median()),
        "median_after": float(df["after"].median()),
        "n": int(len(df))
    }


async def kruskal_wallis(
    user_id: str,
    dataset_id: str,
    value_col: str,
    group_col: str
) -> Dict[str, Any]:
    """
    Kruskal-Wallis H test.
    
    Nonparametric alternative to one-way ANOVA.
    H0: All group distributions are equal
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(value_col)} AS v, {_quote(group_col)} AS g FROM {view} WHERE {_quote(value_col)} IS NOT NULL AND {_quote(group_col)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    groups = [grp["v"].to_numpy() for _, grp in df.groupby("g")]
    
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for Kruskal-Wallis test")
    
    h_stat, p_value = stats.kruskal(*groups)
    
    return {
        "h_statistic": float(h_stat),
        "p_value": float(p_value),
        "k_groups": int(len(groups)),
        "n": int(len(df))
    }


# ============================================================================
# NORMALITY TESTS
# ============================================================================

async def normality_test(user_id: str, dataset_id: str, column: str) -> Dict[str, Any]:
    """
    Test for normality using multiple methods.
    
    Tests:
    - Shapiro-Wilk: Best for small samples (n < 50)
    - Anderson-Darling: Good for all sample sizes
    - Kolmogorov-Smirnov: General goodness of fit
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(
        f"SELECT {_quote(column)} AS col FROM {view} WHERE {_quote(column)} IS NOT NULL"
    ).fetchdf()
    con.close()
    
    data = df["col"].to_numpy()
    
    if len(data) < 3:
        raise ValueError("Need at least 3 observations for normality test")
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    
    # Anderson-Darling test
    anderson_result = stats.anderson(data, dist='norm')
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    
    return {
        "shapiro_wilk": {
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p)
        },
        "anderson_darling": {
            "statistic": float(anderson_result.statistic),
            "critical_values": anderson_result.critical_values.tolist(),
            "significance_levels": anderson_result.significance_level.tolist()
        },
        "kolmogorov_smirnov": {
            "statistic": float(ks_stat),
            "p_value": float(ks_p)
        },
        "n": int(len(data)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data))
    }


# ============================================================================
# REGRESSION
# ============================================================================

async def regression_ols(user_id: str, dataset_id: str, y: str, X: List[str]) -> Dict[str, Any]:
    """
    Ordinary Least Squares (Linear) Regression.
    
    Predict Y from one or more X variables.
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    cols = ", ".join([_quote(y)] + [_quote(c) for c in X])
    df = con.execute(f"SELECT {cols} FROM {view}").fetchdf()
    con.close()
    
    df = df.dropna()
    
    if len(df) < len(X) + 2:
        raise ValueError(f"Need at least {len(X) + 2} observations for regression with {len(X)} predictors")
    
    yv = df[y].astype(float)
    Xv = df[X].astype(float)
    Xv = sm.add_constant(Xv)
    
    model = sm.OLS(yv, Xv).fit()
    
    return {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "f_stat": float(model.fvalue) if model.fvalue is not None else None,
        "f_p_value": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "params": {k: float(v) for k, v in model.params.to_dict().items()},
        "std_errors": {k: float(v) for k, v in model.bse.to_dict().items()},
        "t_values": {k: float(v) for k, v in model.tvalues.to_dict().items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.to_dict().items()},
        "residual_std_error": float(np.sqrt(model.mse_resid))
    }


async def logistic_regression(user_id: str, dataset_id: str, y: str, X: List[str]) -> Dict[str, Any]:
    """
    Logistic Regression for binary outcomes.
    
    Predict binary Y (0/1) from one or more X variables.
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    cols = ", ".join([_quote(y)] + [_quote(c) for c in X])
    df = con.execute(f"SELECT {cols} FROM {view}").fetchdf()
    con.close()
    
    df = df.dropna()
    
    if len(df) < len(X) + 10:
        raise ValueError(f"Need at least {len(X) + 10} observations for logistic regression")
    
    yv = df[y].astype(int)
    Xv = df[X].astype(float)
    Xv = sm.add_constant(Xv)
    
    model = sm.Logit(yv, Xv).fit(disp=0)
    
    return {
        "n": int(model.nobs),
        "pseudo_r2": float(model.prsquared),
        "log_likelihood": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "params": {k: float(v) for k, v in model.params.to_dict().items()},
        "std_errors": {k: float(v) for k, v in model.bse.to_dict().items()},
        "z_values": {k: float(v) for k, v in model.tvalues.to_dict().items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.to_dict().items()},
        "odds_ratios": {k: float(np.exp(v)) for k, v in model.params.to_dict().items()}
    }


# ============================================================================
# TIME SERIES
# ============================================================================

async def moving_average(
    user_id: str,
    dataset_id: str,
    column: str,
    window: int = 7
) -> Dict[str, Any]:
    """
    Calculate moving average (rolling mean).
    
    Smooths time series data by averaging over a window.
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(f"SELECT {_quote(column)} AS col FROM {view}").fetchdf()
    con.close()
    
    data = df["col"].to_numpy()
    ma = pd.Series(data).rolling(window=window, center=False).mean()
    
    return {
        "original": data.tolist(),
        "moving_average": ma.tolist(),
        "window": int(window),
        "n": int(len(data))
    }


async def trend_analysis(
    user_id: str,
    dataset_id: str,
    value_col: str,
    time_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Linear trend analysis.
    
    Fits a linear trend line to time series data.
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    if time_col:
        df = con.execute(
            f"SELECT {_quote(time_col)} AS time, {_quote(value_col)} AS value FROM {view}"
        ).fetchdf()
        x = np.arange(len(df))
    else:
        df = con.execute(f"SELECT {_quote(value_col)} AS value FROM {view}").fetchdf()
        x = np.arange(len(df))
    
    con.close()
    
    y = df["value"].to_numpy()
    
    # Fit linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate trend line
    trend_line = intercept + slope * x
    
    # Detrended data (residuals)
    detrended = y - trend_line
    
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "trend_direction": "increasing" if slope > 0 else "decreasing",
        "trend_line": trend_line.tolist(),
        "detrended": detrended.tolist(),
        "n": int(len(y))
    }


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

async def outlier_detection(
    user_id: str,
    dataset_id: str,
    column: str,
    method: str = "zscore",
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect outliers using various methods.
    
    Methods:
    - zscore: Z-score method (default, threshold=3.0)
    - iqr: Interquartile range method (threshold=1.5)
    - modified_z: Modified Z-score using median
    """
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    df = con.execute(f"SELECT {_quote(column)} AS col FROM {view}").fetchdf()
    con.close()
    
    data = df["col"].to_numpy()
    
    if method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        outlier_values = data[outliers].tolist()
        outlier_indices = np.where(outliers)[0].tolist()
        
    elif method == "iqr":
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)
        outlier_values = data[outliers].tolist()
        outlier_indices = np.where(outliers)[0].tolist()
        
    elif method == "modified_z":
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        outlier_values = data[outliers].tolist()
        outlier_indices = np.where(outliers)[0].tolist()
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return {
        "method": method,
        "threshold": float(threshold),
        "n_outliers": int(np.sum(outliers)),
        "outlier_percentage": float(100 * np.sum(outliers) / len(data)),
        "outlier_indices": outlier_indices,
        "outlier_values": outlier_values,
        "n_total": int(len(data))
    }


# ============================================================================
# MAIN ROUTING FUNCTION
# ============================================================================

async def run_stats(
    user_id: str,
    dataset_id: str,
    analysis: str,
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], bool]:
    """
    Main routing function for statistical analyses.
    
    Returns: (result_dict, is_cached)
    
    Supported analyses:
    - Descriptive: descriptives, detailed_descriptives
    - Correlation: correlation, correlation_matrix
    - T-tests: ttest_1samp, ttest_2samp, paired_ttest
    - ANOVA: anova_oneway
    - Chi-square: chi_square_goodness, chi_square_independence
    - Nonparametric: mann_whitney, wilcoxon_signed_rank, kruskal_wallis
    - Normality: normality_test
    - Regression: regression_ols, logistic_regression
    - Time Series: moving_average, trend_analysis
    - Outliers: outlier_detection
    """
    # --- Concept slug aliases ---
    # The preferred contract is to call this endpoint with a concept slug.
    # Internally we map slugs to concrete analysis implementations.
    alias = {
        # tests
        "one-sample-t-test": "ttest_1samp",
        "two-sample-t-test": "ttest_2samp",
        "paired-t-test": "paired_ttest",
        "one-way-anova": "anova_oneway",
        "anova-one-way": "anova_oneway",
        "pearson-correlation": "correlation",
        "spearman-correlation": "correlation",
        "linear-regression": "regression_ols",
        "simple-linear-regression": "regression_ols",
        # metrics/bundles
        "descriptive-statistics": "descriptives",
        "summary-statistics": "descriptives",
        "standard-deviation": "descriptives",
        "mean": "descriptives",
        "median": "detailed_descriptives",
        "quartiles": "detailed_descriptives",
    }
    analysis = alias.get(analysis, analysis)
    # ------------------------------------------------------------------
    # Check cache first (lineage-safe)
    # ------------------------------------------------------------------
    meta = await registry.fetchrow(
        """
        SELECT profile_json
        FROM datasets
        WHERE dataset_id = $1
          AND user_id = $2
        """,
        dataset_id,
        user_id,
    )

    if not meta or not meta["profile_json"]:
        raise ValueError("Dataset profile not available; cannot compute stats safely")

    profile = meta["profile_json"]
    snapshot_payload = {
        "dataset_id": dataset_id,
        "parquet_sha": profile.get("parquet_sha"),
        "pipeline_hash": profile.get("pipeline_hash", "__none__"),
        "engine_version": profile.get("engine_version"),
     }

    snapshot_id = hashlib.sha256(
               json.dumps(snapshot_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    h = _hash_spec(
        dataset_id=dataset_id,
        analysis=analysis,
        params=params,
        parquet_ref=profile.get("parquet_ref"),
        parquet_sha=profile.get("parquet_sha"),
        pipeline_hash=profile.get("pipeline_hash", "__none__"),
        engine_version=profile.get("engine_version"),
    )

    cached = await try_cache_get(user_id, dataset_id, h)
    if cached is not None:
        cached["_snapshot_id"] = snapshot_id
        return cached, True

    # Route to appropriate analysis function
    result = None
    
    # DESCRIPTIVE STATISTICS
    if analysis == "descriptives":
        result = await descriptives(user_id, dataset_id, params.get("columns", []))
    
    elif analysis == "detailed_descriptives":
        result = await detailed_descriptives(user_id, dataset_id, params.get("columns", []))
    
    # CORRELATION
    elif analysis == "correlation":
        result = await correlation(
            user_id, dataset_id,
            params["x"], params["y"],
            params.get("method", "pearson")
        )
    
    elif analysis == "correlation_matrix":
        result = await correlation_matrix(
            user_id, dataset_id,
            params["columns"],
            params.get("method", "pearson")
        )
    
    # T-TESTS
    elif analysis == "ttest_1samp":
        result = await ttest_1samp(
            user_id, dataset_id,
            params["column"],
            params["pop_mean"]
        )
    
    elif analysis == "ttest_2samp" or analysis == "ttest":
        result = await ttest_2samp(user_id, dataset_id, params["x"], params["y"])
    
    elif analysis == "paired_ttest":
        result = await paired_ttest(
            user_id, dataset_id,
            params["before"],
            params["after"]
        )
    
    # ANOVA
    elif analysis == "anova_oneway" or analysis == "anova":
        result = await anova_oneway(
            user_id, dataset_id,
            params["value_col"],
            params["group_col"]
        )
    
    # CHI-SQUARE
    elif analysis == "chi_square_goodness":
        result = await chi_square_goodness(
            user_id, dataset_id,
            params["observed_col"],
            params.get("expected_col")
        )
    
    elif analysis == "chi_square_independence":
        result = await chi_square_independence(
            user_id, dataset_id,
            params["var1"],
            params["var2"]
        )
    
    # NONPARAMETRIC TESTS
    elif analysis == "mann_whitney":
        result = await mann_whitney(user_id, dataset_id, params["x"], params["y"])
    
    elif analysis == "wilcoxon_signed_rank":
        result = await wilcoxon_signed_rank(
            user_id, dataset_id,
            params["before"],
            params["after"]
        )
    
    elif analysis == "kruskal_wallis":
        result = await kruskal_wallis(
            user_id, dataset_id,
            params["value_col"],
            params["group_col"]
        )
    
    # NORMALITY
    elif analysis == "normality_test":
        result = await normality_test(user_id, dataset_id, params["column"])
    
    # REGRESSION
    elif analysis == "regression_ols" or analysis == "regression":
        result = await regression_ols(user_id, dataset_id, params["y"], params["X"])
    
    elif analysis == "logistic_regression":
        result = await logistic_regression(user_id, dataset_id, params["y"], params["X"])
    
    # TIME SERIES
    elif analysis == "moving_average":
        result = await moving_average(
            user_id, dataset_id,
            params["column"],
            params.get("window", 7)
        )
    
    elif analysis == "trend_analysis":
        result = await trend_analysis(
            user_id, dataset_id,
            params["value_col"],
            params.get("time_col")
        )
    
    # OUTLIERS
    elif analysis == "outlier_detection":
        result = await outlier_detection(
            user_id, dataset_id,
            params["column"],
            params.get("method", "zscore"),
            params.get("threshold", 3.0)
        )
    
    else:
        raise ValueError(f"Unsupported analysis: {analysis}")
    
    # Cache the result
    # Cache the result
    result["_snapshot_id"] = snapshot_id

    await cache_put(
        user_id,
        dataset_id,
        h,
        {"analysis": analysis, "params": params, "snapshot_id": snapshot_id},
        result,
    )

    return result, False


# ============================================================================
# FUTURE EXPANSION GUIDE
# ============================================================================

"""
TO ADD A NEW STATISTICAL ANALYSIS:

1. Create the analysis function following this pattern:

async def your_new_analysis(user_id: str, dataset_id: str, ...params...) -> Dict[str, Any]:
    '''
    Brief description of what this analysis does.
    
    H0: State the null hypothesis if applicable
    '''
    # Get data
    parquet = await _get_parquet_local(user_id, dataset_id)
    eng = DuckDBEngine(user_id)
    con = eng.connect()
    view = eng.register_parquet(con, dataset_id, parquet)
    
    # Query needed columns
    df = con.execute(f"SELECT ... FROM {view}").fetchdf()
    con.close()
    
    # Perform analysis (use scipy, statsmodels, numpy, etc.)
    result = your_analysis_here(...)
    
    # Return formatted dictionary
    return {
        "key1": float(value1),
        "key2": float(value2),
        # ... more results
    }

2. Add to run_stats function:

elif analysis == "your_new_analysis":
    result = await your_new_analysis(user_id, dataset_id, params["param1"], params["param2"])

3. That's it! The caching, routing, and API integration all work automatically.

CATEGORIES TO CONSIDER ADDING:
- Two-way ANOVA
- Repeated measures ANOVA
- Multiple comparison tests (Tukey HSD, Bonferroni)
- ARIMA forecasting
- Seasonal decomposition
- Bootstrap methods
- Power analysis
- Sample size calculations
- Survival analysis (Kaplan-Meier, Cox regression)
- Factor analysis
- Principal Component Analysis (PCA)
- Cluster analysis (K-means, hierarchical)
- Discriminant analysis
"""
