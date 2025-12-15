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
        
        Args:
            inputs: {
                "data_context": {
                    "dataframe": pd.DataFrame,
                    "filename": str
                },
                "focus": List[str] (optional)
            }
        
        Returns:
            {
                "data_profile": {...},
                "quality_report": {...},
                "interesting_features": [...],
                "preliminary_insights": [...]
            }
        """
        df = inputs["data_context"]["dataframe"]
        
        self.log(f"Profiling dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Build data profile
        data_profile = self._build_profile(df)
        
        # Assess quality
        quality_report = self._assess_quality(df)
        
        # Find interesting features
        interesting_features = self._find_interesting_features(df)
        
        # Generate preliminary insights
        preliminary_insights = self._generate_preliminary_insights(df, data_profile)
        
        return {
            "data_profile": data_profile,
            "quality_report": quality_report,
            "interesting_features": interesting_features,
            "preliminary_insights": preliminary_insights
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
            col_profile = self._profile_column(df[col], col)
            profile["columns"][col] = col_profile
        
        return profile
    
    def _profile_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Profile individual column"""
        profile = {
            "dtype": str(series.dtype),
            "role": self._infer_role(series),
            "missing_pct": float(series.isna().mean() * 100),
            "unique_count": int(series.nunique())
        }
        
        # Add stats for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            profile.update({
                "mean": float(desc["mean"]) if not np.isnan(desc["mean"]) else None,
                "median": float(series.median()) if not np.isnan(series.median()) else None,
                "std": float(desc["std"]) if not np.isnan(desc["std"]) else None,
                "min": float(desc["min"]) if not np.isnan(desc["min"]) else None,
                "max": float(desc["max"]) if not np.isnan(desc["max"]) else None,
                "q25": float(desc["25%"]) if not np.isnan(desc["25%"]) else None,
                "q75": float(desc["75%"]) if not np.isnan(desc["75%"]) else None,
                "outlier_count": self._count_outliers(series)
            })
        
        return profile
    
    def _infer_role(self, series: pd.Series) -> str:
        """Infer semantic role of column"""
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif series.nunique() / max(len(series), 1) < 0.05:
            return "categorical"
        else:
            return "text"
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
        return int(outliers)
    
    def _assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        issues = []
        
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
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append({
                "type": "duplicates",
                "severity": "low",
                "message": f"{dup_count} duplicate rows found",
                "count": int(dup_count)
            })
        
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
            cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            if cv > 1.0:  # Coefficient of variation > 1
                interesting.append({
                    "column": col,
                    "reason": "high_variance",
                    "detail": f"High variability (CV={cv:.2f})"
                })
        
        # Skewed distributions
        for col in numeric_cols:
            skew = df[col].skew()
            if abs(skew) > 2:
                interesting.append({
                    "column": col,
                    "reason": "skewed_distribution",
                    "detail": f"{'Right' if skew > 0 else 'Left'} skewed (skew={skew:.2f})"
                })
        
        # Unbalanced categorical
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.9:  # Dominant category
                interesting.append({
                    "column": col,
                    "reason": "imbalanced",
                    "detail": f"Highly imbalanced ({value_counts.iloc[0]*100:.1f}% in one category)"
                })
        
        return interesting[:10]  # Limit to top 10
    
    def _generate_preliminary_insights(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any]
    ) -> List[str]:
        """Generate preliminary insights"""
        insights = []
        
        # Data size insight
        n_rows = profile["shape"]["rows"]
        if n_rows < 100:
            insights.append(f"Small dataset (n={n_rows}) - statistical power may be limited")
        elif n_rows > 10000:
            insights.append(f"Large dataset (n={n_rows}) - robust analysis possible")
        
        # Missing data insight
        missing_cols = [
            col for col, info in profile["columns"].items()
            if info["missing_pct"] > 10
        ]
        if missing_cols:
            insights.append(f"{len(missing_cols)} columns have >10% missing data")
        
        # Column types insight
        numeric_count = sum(1 for c in profile["columns"].values() if c["role"] == "numeric")
        categorical_count = sum(1 for c in profile["columns"].values() if c["role"] == "categorical")
        
        insights.append(f"Dataset contains {numeric_count} numeric and {categorical_count} categorical features")
        
        return insights
