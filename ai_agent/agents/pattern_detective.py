"""
Pattern Detective Agent - Finds patterns in data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from ai_agent.core.orchestrator import BaseAgent, AgentRole


class PatternDetectiveAgent(BaseAgent):
    """
    Detects patterns in datasets
    - Temporal trends
    - Clustering
    - Associations
    - Anomalies
    """
    
    def __init__(self):
        super().__init__(AgentRole.PATTERN_DETECTIVE)
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect patterns in data
        
        Args:
            inputs: {
                "data_context": {
                    "dataframe": pd.DataFrame
                },
                "exploration_insights": Dict (optional)
            }
        
        Returns:
            {
                "temporal_patterns": {...},
                "clustering_patterns": {...},
                "association_patterns": {...},
                "anomaly_patterns": {...},
                "summary_insights": [...]
            }
        """
        df = inputs["data_context"]["dataframe"]
        
        self.log(f"Detecting patterns in dataset")
        
        results = {
            "temporal_patterns": self._detect_temporal_patterns(df),
            "clustering_patterns": self._detect_clusters(df),
            "association_patterns": self._detect_associations(df),
            "anomaly_patterns": self._detect_anomalies(df),
            "summary_insights": []
        }
        
        # Generate summary
        results["summary_insights"] = self._summarize_patterns(results)
        
        return results
    
    def _detect_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect trends over time"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            return {"message": "No datetime columns found"}
        
        time_col = datetime_cols[0]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        trends = []
        for col in numeric_cols[:5]:  # Limit to first 5
            trend = self._analyze_trend(df, time_col, col)
            if trend:
                trends.append(trend)
        
        return {
            "trends_detected": len(trends) > 0,
            "trends": trends
        }
    
    def _analyze_trend(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_col: str
    ) -> Dict[str, Any]:
        """Analyze single trend"""
        data = df[[time_col, value_col]].dropna()
        if len(data) < 10:
            return None
        
        data = data.sort_values(time_col)
        x = np.arange(len(data))
        y = data[value_col].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        if p_value < 0.05:  # Significant trend
            return {
                "column": value_col,
                "direction": "increasing" if slope > 0 else "decreasing",
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "start_value": float(y[0]),
                "end_value": float(y[-1]),
                "percent_change": float((y[-1] - y[0]) / abs(y[0]) * 100),
                "n_observations": len(data)
            }
        
        return None
    
    def _detect_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect natural clusters"""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.shape[0] < 10 or numeric_df.shape[1] < 2:
            return {
                "clusters_detected": False,
                "message": "Insufficient data for clustering"
            }
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        
        # Find optimal k
        best_k = 2
        best_score = -1
        
        for k in range(2, min(6, len(X_scaled) // 3)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        cluster_sizes = {}
        for i in range(best_k):
            cluster_sizes[f"cluster_{i}"] = int((labels == i).sum())
        
        return {
            "clusters_detected": True,
            "n_clusters": best_k,
            "silhouette_score": float(best_score),
            "cluster_sizes": cluster_sizes,
            "cluster_details": [{
                "n_clusters": best_k,
                "method": "kmeans",
                "silhouette_score": float(best_score),
                "cluster_sizes": cluster_sizes
            }]
        }
    
    def _detect_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect associations between variables"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        categorical_associations = []
        numeric_categorical_associations = []
        
        # Chi-square tests for categorical
        for i, col1 in enumerate(cat_cols[:3]):
            for col2 in cat_cols[i+1:4]:
                if col1 != col2:
                    contingency = pd.crosstab(df[col1], df[col2])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    
                    if p_value < 0.05:
                        categorical_associations.append({
                            "var1": col1,
                            "var2": col2,
                            "test": "chi_square",
                            "statistic": float(chi2),
                            "p_value": float(p_value),
                            "significant": True
                        })
        
        # ANOVA for numeric vs categorical
        for num_col in numeric_cols[:3]:
            for cat_col in cat_cols[:2]:
                groups = [g[num_col].dropna().values for _, g in df.groupby(cat_col)]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    if p_value < 0.05:
                        numeric_categorical_associations.append({
                            "numeric_var": num_col,
                            "categorical_var": cat_col,
                            "test": "anova",
                            "f_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "significant": True
                        })
        
        return {
            "categorical_associations": categorical_associations,
            "numeric_categorical_associations": numeric_categorical_associations
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.shape[0] < 10:
            return {
                "anomalies_detected": False,
                "message": "Insufficient data"
            }
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        
        # Isolation Forest
        iso = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso.fit_predict(X_scaled)
        
        anomaly_count = (predictions == -1).sum()
        
        return {
            "anomalies_detected": anomaly_count > 0,
            "count": int(anomaly_count),
            "percentage": float(anomaly_count / len(predictions) * 100),
            "method": "isolation_forest"
        }
    
    def _summarize_patterns(self, results: Dict[str, Any]) -> List[str]:
        """Generate summary insights"""
        insights = []
        
        # Temporal
        if results["temporal_patterns"].get("trends_detected"):
            n_trends = len(results["temporal_patterns"].get("trends", []))
            insights.append(f"Detected {n_trends} significant temporal trends")
        
        # Clustering
        if results["clustering_patterns"].get("clusters_detected"):
            n_clusters = results["clustering_patterns"]["n_clusters"]
            insights.append(f"Found {n_clusters} natural clusters in the data")
        
        # Associations
        n_assoc = len(results["association_patterns"].get("categorical_associations", []))
        n_assoc += len(results["association_patterns"].get("numeric_categorical_associations", []))
        if n_assoc > 0:
            insights.append(f"Identified {n_assoc} significant associations between variables")
        
        # Anomalies
        if results["anomaly_patterns"].get("anomalies_detected"):
            pct = results["anomaly_patterns"]["percentage"]
            insights.append(f"Detected {pct:.1f}% anomalous observations")
        
        return insights
