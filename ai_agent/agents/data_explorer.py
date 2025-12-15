"""
Data Explorer Agent - Profiles data and assesses quality
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ai_agent.core.orchestrator import BaseAgent, AgentRole


class DataExplorerAgent(BaseAgent):
    """
    Explores and profiles datasets
    - Analyzes column types and distributions
    - Assesses data quality
    - Identifies interesting features
    """
    
    def __init__(self):
        super().__init__(AgentRole.DATA_EXPLORER)
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Profile the dataset
        """
        df = inputs["data_context"]["dataframe"]
        
        self.log(f"Profiling dataset: {len(df)} rows, {len(df.columns)} columns")
        
        try:
            data_profile = self._build_profile(df)
            quality_report = self._assess_quality(df)
            interesting_features = self._find_interesting_features(df)
            preliminary_insights = self._generate_preliminary_insights(df, data_profile)
            
            return {
                "data_profile": data_profile,
                "quality_report": quality_report,
                "interesting_features": interesting_features,
                "preliminary_insights": preliminary_insights
            }
        except Exception as e:
            self.log(f"Error in execute: {str(e)}")
            return {
                "data_profile": {"shape": {"rows": len(df), "columns": len(df.columns)}, "columns": {}},
                "quality_report": {"overall_score": 0, "issues": [{"type": "error", "message": str(e)}], "passed_checks": 0},
                "interesting_features": [],
                "preliminary_insights": [f"Error during profiling: {str(e)}"]
            }
    
    def _build_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build comprehensive data profile"""
        profile = {
            "shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "columns": {}
        }
        
        for col in df.columns:
            try:
                col_profile = self._profile_column(df[col], col)
                profile["columns"][col] = col_profile
            except Exception as e:
                profile["columns"][col] = {
                    "dtype": "unknown",
                    "role": "unknown",
                    "missing_pct": 0,
                    "unique_count": 0,
                    "error": str(e)
                }
        
        return profile
    
    def _profile_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Profile individual column"""
        try:
            missing_pct = float(series.isna().mean() * 100)
        except Exception:
            missing_pct = 0.0
            
        try:
            unique_count = int(series.nunique())
        except Exception:
            unique_count = 0
            
        profile = {
            "dtype": str(series.dtype),
            "role": self._infer_role(series),
            "missing_pct": missing_pct,
            "unique_count": unique_count
        }
        
        # Add stats for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            try:
                # Drop NaN before calculating stats
                clean_series = series.dropna()
                if len(clean_series) > 0:
                    desc = clean_series.describe()
                    profile.update({
                        "mean": self._safe_float(desc.get("mean")),
                        "median": self._safe_float(clean_series.median()),
                        "std": self._safe_float(desc.get("std")),
                        "min": self._safe_float(desc.get("min")),
                        "max": self._safe_float(desc.get("max")),
                        "q25": self._safe_float(desc.get("25%")),
                        "q75": self._safe_float(desc.get("75%")),
                        "outlier_count": self._count_outliers(clean_series)
                    })
                else:
                    profile.update({
                        "mean": None, "median": None, "std": None,
                        "min": None, "max": None, "q25": None, "q75": None,
                        "outlier_count": 0
                    })
            except Exception as e:
                profile.update({
                    "mean": None, "median": None, "std": None,
                    "min": None, "max": None, "q25": None, "q75": None,
                    "outlier_count": 0,
                    "stats_error": str(e)
                })
        
        return profile
    
    def _safe_float(self, value) -> float:
        """Safely convert to float, returning None for NaN/Inf"""
        if value is None:
            return None
        try:
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None
    
    def _infer_role(self, series: pd.Series) -> str:
        """Infer semantic role of column"""
        try:
            if pd.api.types.is_numeric_dtype(series):
                return "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                return "datetime"
            elif series.nunique() / max(len(series), 1) < 0.05:
                return "categorical"
            else:
                return "text"
        except Exception:
            return "unknown"
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        try:
            clean = series.dropna()
            if len(clean) < 4:
                return 0
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                return 0
            outliers = ((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).sum()
            return int(outliers)
        except Exception:
            return 0
    
    def _assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        issues = []
        
        try:
            # Check for high missing data
            missing_pct = df.isna().mean() * 100
            high_missing = missing_pct[missing_pct > 50]
            if len(high_missing) > 0:
                issues.append({
                    "type": "high_missing_data",
                    "severity": "medium",
                    "message": f"{len(high_missing)} columns have >50% missing data",
                    "columns": list(high_missing.index)
                })
        except Exception:
            pass
        
        try:
            # Check for duplicates
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                issues.append({
                    "type": "duplicates",
                    "severity": "low",
                    "message": f"{dup_count} duplicate rows found",
                    "count": int(dup_count)
                })
        except Exception:
            pass
        
        # Check sample size
        if len(df) < 100:
            issues.append({
                "type": "small_sample",
                "severity": "high",
                "message": f"Small sample size (n={len(df)}) may limit analysis",
                "count": len(df)
            })
        
        # Calculate overall score
        score = 100 - (len(issues) * 15)
        score = max(0, min(100, score))
        
        return {
            "overall_score": score,
            "issues": issues,
            "passed_checks": 10 - len(issues)
        }
    
    def _find_interesting_features(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify potentially interesting features"""
        interesting = []
        
        # High variance numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                clean = df[col].dropna()
                if len(clean) < 2:
                    continue
                mean_val = clean.mean()
                std_val = clean.std()
                if pd.isna(mean_val) or pd.isna(std_val) or mean_val == 0:
                    continue
                cv = abs(std_val / mean_val)
                if cv > 1.0:
                    interesting.append({
                        "column": col,
                        "reason": "high_variance",
                        "detail": f"High variability (CV={cv:.2f})"
                    })
            except Exception:
                continue
        
        # Skewed distributions
        for col in numeric_cols:
            try:
                clean = df[col].dropna()
                if len(clean) < 3:
                    continue
                skew = clean.skew()
                if pd.isna(skew):
                    continue
                if abs(skew) > 2:
                    interesting.append({
                        "column": col,
                        "reason": "skewed_distribution",
                        "detail": f"{'Right' if skew > 0 else 'Left'} skewed (skew={skew:.2f})"
                    })
            except Exception:
                continue
        
        # Unbalanced categorical
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            try:
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 0 and value_counts.iloc[0] > 0.9:
                    interesting.append({
                        "column": col,
                        "reason": "imbalanced",
                        "detail": f"Highly imbalanced ({value_counts.iloc[0]*100:.1f}% in one category)"
                    })
            except Exception:
                continue
        
        return interesting[:10]
    
    def _generate_preliminary_insights(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any]
    ) -> List[str]:
        """Generate preliminary insights"""
        insights = []
        
        try:
            n_rows = profile["shape"]["rows"]
            if n_rows < 100:
                insights.append(f"Small dataset (n={n_rows}) - statistical power may be limited")
            elif n_rows > 10000:
                insights.append(f"Large dataset (n={n_rows}) - robust analysis possible")
        except Exception:
            pass
        
        try:
            missing_cols = [
                col for col, info in profile.get("columns", {}).items()
                if info.get("missing_pct", 0) > 10
            ]
            if missing_cols:
                insights.append(f"{len(missing_cols)} columns have >10% missing data")
        except Exception:
            pass
        
        try:
            numeric_count = sum(1 for c in profile.get("columns", {}).values() if c.get("role") == "numeric")
            categorical_count = sum(1 for c in profile.get("columns", {}).values() if c.get("role") == "categorical")
            insights.append(f"Dataset contains {numeric_count} numeric and {categorical_count} categorical features")
        except Exception:
            pass
        
        return insights
