"""Descriptive Statistics - 12 Minitab concepts."""

from typing import Any, Dict, List
import numpy as np
from scipy import stats

class DescriptiveStats:
    """
    Concepts covered:
    1. Mean, Median, Mode
    2. Standard Deviation, Variance
    3. Range, IQR
    4. Skewness, Kurtosis
    5. Percentiles (P10, P25, P50, P75, P90)
    6. Coefficient of Variation
    7. Standard Error
    8. Confidence Intervals
    9. Trimmed Mean
    10. Geometric Mean
    11. Harmonic Mean
    12. Summary Statistics (describe)
    """
    
    @staticmethod
    def full_descriptives(data: np.ndarray, confidence: float = 0.95) -> Dict[str, Any]:
        """Complete descriptive statistics for a numeric array."""
        clean = data[~np.isnan(data)]
        n = len(clean)
        
        if n == 0:
            return {"error": "No valid data points"}
        
        mean = float(np.mean(clean))
        std = float(np.std(clean, ddof=1)) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        
        # Confidence interval
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1) if n > 1 else 0
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        
        return {
            "n": n,
            "n_missing": int(np.sum(np.isnan(data))),
            "mean": mean,
            "median": float(np.median(clean)),
            "mode": float(stats.mode(clean, keepdims=True).mode[0]) if n > 0 else None,
            "std": std,
            "variance": float(np.var(clean, ddof=1)) if n > 1 else 0,
            "se_mean": se,
            "min": float(np.min(clean)),
            "max": float(np.max(clean)),
            "range": float(np.ptp(clean)),
            "q1": float(np.percentile(clean, 25)),
            "q3": float(np.percentile(clean, 75)),
            "iqr": float(np.percentile(clean, 75) - np.percentile(clean, 25)),
            "skewness": float(stats.skew(clean)) if n > 2 else None,
            "kurtosis": float(stats.kurtosis(clean)) if n > 3 else None,
            "cv": (std / mean * 100) if mean != 0 else None,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence,
            "trimmed_mean_5pct": float(stats.trim_mean(clean, 0.05)) if n > 2 else mean,
            "geometric_mean": float(stats.gmean(clean[clean > 0])) if np.all(clean > 0) else None,
            "harmonic_mean": float(stats.hmean(clean[clean > 0])) if np.all(clean > 0) else None,
            "percentiles": {
                "p10": float(np.percentile(clean, 10)),
                "p25": float(np.percentile(clean, 25)),
                "p50": float(np.percentile(clean, 50)),
                "p75": float(np.percentile(clean, 75)),
                "p90": float(np.percentile(clean, 90)),
                "p95": float(np.percentile(clean, 95)),
                "p99": float(np.percentile(clean, 99)),
            }
        }
    
    @staticmethod
    def normality_test(data: np.ndarray) -> Dict[str, Any]:
        """Test for normality using multiple methods."""
        clean = data[~np.isnan(data)]
        n = len(clean)
        
        results = {"n": n}
        
        # Shapiro-Wilk (best for n < 5000)
        if 3 <= n <= 5000:
            stat, p = stats.shapiro(clean)
            results["shapiro_wilk"] = {"statistic": float(stat), "p_value": float(p)}
        
        # Anderson-Darling
        if n >= 8:
            ad = stats.anderson(clean, dist='norm')
            results["anderson_darling"] = {
                "statistic": float(ad.statistic),
                "critical_values": {f"{int(sl)}%": float(cv) for sl, cv in zip(ad.significance_level, ad.critical_values)}
            }
        
        # D'Agostino-Pearson
        if n >= 20:
            stat, p = stats.normaltest(clean)
            results["dagostino_pearson"] = {"statistic": float(stat), "p_value": float(p)}
        
        # Jarque-Bera
        if n >= 20:
            stat, p = stats.jarque_bera(clean)
            results["jarque_bera"] = {"statistic": float(stat), "p_value": float(p)}
        
        return results
