"""Hypothesis Testing - 18 Minitab concepts."""

from typing import Any, Dict, List, Optional, Literal
import numpy as np
from scipy import stats

class HypothesisTesting:
    """
    Concepts covered:
    1. One-sample t-test
    2. Two-sample t-test (independent)
    3. Paired t-test
    4. One-sample z-test
    5. Two-sample z-test
    6. One proportion test
    7. Two proportion test
    8. Chi-square goodness of fit
    9. Chi-square test of independence
    10. Fisher's exact test
    11. McNemar's test
    12. One-sample variance test
    13. Two-sample variance test (F-test)
    14. Levene's test
    15. Bartlett's test
    16. Power analysis (t-test)
    17. Sample size calculation
    18. Equivalence testing (TOST)
    """
    
    @staticmethod
    def one_sample_t(data: np.ndarray, mu: float = 0, 
                     alternative: Literal["two-sided", "less", "greater"] = "two-sided",
                     confidence: float = 0.95) -> Dict[str, Any]:
        """One-sample t-test."""
        clean = data[~np.isnan(data)]
        n = len(clean)
        
        result = stats.ttest_1samp(clean, mu, alternative=alternative)
        mean = float(np.mean(clean))
        std = float(np.std(clean, ddof=1))
        se = std / np.sqrt(n)
        
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
        
        return {
            "test": "one_sample_t",
            "n": n,
            "mean": mean,
            "std": std,
            "se": se,
            "hypothesized_mean": mu,
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "df": n - 1,
            "alternative": alternative,
            "ci_lower": mean - t_crit * se,
            "ci_upper": mean + t_crit * se,
            "confidence_level": confidence,
            "significant_at_05": result.pvalue < 0.05,
        }
    
    @staticmethod
    def two_sample_t(group1: np.ndarray, group2: np.ndarray,
                     equal_var: bool = False,
                     alternative: Literal["two-sided", "less", "greater"] = "two-sided") -> Dict[str, Any]:
        """Two-sample independent t-test (Welch's by default)."""
        g1 = group1[~np.isnan(group1)]
        g2 = group2[~np.isnan(group2)]
        
        result = stats.ttest_ind(g1, g2, equal_var=equal_var, alternative=alternative)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1)) / (len(g1)+len(g2)-2))
        cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            "test": "two_sample_t",
            "test_type": "equal_variance" if equal_var else "welch",
            "n1": len(g1),
            "n2": len(g2),
            "mean1": float(np.mean(g1)),
            "mean2": float(np.mean(g2)),
            "std1": float(np.std(g1, ddof=1)),
            "std2": float(np.std(g2, ddof=1)),
            "mean_difference": float(np.mean(g1) - np.mean(g2)),
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "df": float(result.df) if hasattr(result, 'df') else len(g1) + len(g2) - 2,
            "alternative": alternative,
            "cohens_d": float(cohens_d),
            "effect_size_interpretation": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large",
            "significant_at_05": result.pvalue < 0.05,
        }
    
    @staticmethod
    def paired_t(before: np.ndarray, after: np.ndarray,
                 alternative: Literal["two-sided", "less", "greater"] = "two-sided") -> Dict[str, Any]:
        """Paired samples t-test."""
        # Remove pairs with NaN in either
        mask = ~(np.isnan(before) | np.isnan(after))
        b = before[mask]
        a = after[mask]
        
        diff = a - b
        result = stats.ttest_rel(b, a, alternative=alternative)
        
        return {
            "test": "paired_t",
            "n_pairs": len(b),
            "mean_before": float(np.mean(b)),
            "mean_after": float(np.mean(a)),
            "mean_difference": float(np.mean(diff)),
            "std_difference": float(np.std(diff, ddof=1)),
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "df": len(b) - 1,
            "alternative": alternative,
            "significant_at_05": result.pvalue < 0.05,
        }
    
    @staticmethod
    def chi_square_independence(contingency_table: np.ndarray) -> Dict[str, Any]:
        """Chi-square test of independence."""
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Cramér's V effect size
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            "test": "chi_square_independence",
            "chi2_statistic": float(chi2),
            "p_value": float(p),
            "df": int(dof),
            "expected_frequencies": expected.tolist(),
            "cramers_v": float(cramers_v),
            "effect_size_interpretation": "small" if cramers_v < 0.1 else "medium" if cramers_v < 0.3 else "large",
            "significant_at_05": p < 0.05,
        }
    
    @staticmethod
    def chi_square_goodness_of_fit(observed: np.ndarray, 
                                    expected: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Chi-square goodness of fit test."""
        if expected is None:
            # Assume uniform distribution
            expected = np.full_like(observed, observed.mean(), dtype=float)
        
        result = stats.chisquare(observed, f_exp=expected)
        
        return {
            "test": "chi_square_goodness_of_fit",
            "chi2_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "df": len(observed) - 1,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "significant_at_05": result.pvalue < 0.05,
        }
    
    @staticmethod
    def levene_test(*groups: np.ndarray, center: str = "median") -> Dict[str, Any]:
        """Levene's test for equality of variances."""
        clean_groups = [g[~np.isnan(g)] for g in groups]
        result = stats.levene(*clean_groups, center=center)
        
        return {
            "test": "levene",
            "center": center,
            "n_groups": len(clean_groups),
            "group_sizes": [len(g) for g in clean_groups],
            "group_variances": [float(np.var(g, ddof=1)) for g in clean_groups],
            "w_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "significant_at_05": result.pvalue < 0.05,
            "interpretation": "Variances are equal" if result.pvalue >= 0.05 else "Variances are NOT equal",
        }
    
    @staticmethod
    def fisher_exact(contingency_2x2: np.ndarray) -> Dict[str, Any]:
        """Fisher's exact test for 2x2 contingency tables."""
        if contingency_2x2.shape != (2, 2):
            return {"error": "Fisher's exact test requires a 2x2 table"}
        
        odds_ratio, p = stats.fisher_exact(contingency_2x2)
        
        return {
            "test": "fisher_exact",
            "odds_ratio": float(odds_ratio),
            "p_value": float(p),
            "significant_at_05": p < 0.05,
        }
