"""
Pattern Detective Agent - Finds patterns in data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from ai_agent.core.orchestrator import BaseAgent, AgentRole


class PatternDetectiveAgent(BaseAgent):
    """
    Detects patterns in datasets
    """
    
    def __init__(self):
        super().__init__(AgentRole.PATTERN_DETECTIVE)
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns in data"""
        df = inputs["data_context"]["dataframe"]
        
        self.log(f"Detecting patterns in dataset")
        
        try:
            results = {
                "temporal_patterns": self._detect_temporal_patterns(df),
                "clustering_patterns": self._detect_clusters(df),
                "association_patterns": self._detect_associations(df),
                "anomaly_patterns": self._detect_anomalies(df),
                "summary_insights": []
            }
            results["summary_insights"] = self._summarize_patterns(results)
            return results
        except Exception as e:
            self.log(f"Error in execute: {str(e)}")
            return {
                "temporal_patterns": {"message": "Error analyzing temporal patterns"},
                "clustering_patterns": {"clusters_detected": False},
                "association_patterns": {},
                "anomaly_patterns": {"anomalies_detected": False},
                "summary_insights": [f"Error during pattern detection: {str(e)}"]
            }
    
    def _detect_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect trends over time"""
        try:
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(datetime_cols) == 0:
                return {"message": "No datetime columns found"}
            
            time_col = datetime_cols[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            trends = []
            for col in numeric_cols[:5]:
                try:
                    trend = self._analyze_trend(df, time_col, col)
                    if trend:
                        trends.append(trend)
                except Exception:
                    continue
            
            return {
                "trends_detected": len(trends) > 0,
                "trends": trends
            }
        except Exception as e:
            return {"message": f"Error: {str(e)}"}
    
    def _analyze_trend(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, Any]:
        """Analyze trend for a single variable"""
        try:
            data = df[[time_col, value_col]].dropna().copy()
            if len(data) < 10:
                return None
            
            data = data.sort_values(time_col)
            data['time_numeric'] = (data[time_col] - data[time_col].min()).dt.total_seconds()
            
            x = data['time_numeric'].values
            y = data[value_col].values
            
            if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                return None
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            if pd.isna(slope) or pd.isna(p_value):
                return None
            
            if p_value < 0.05:
                start_val = float(y[0]) if not np.isnan(y[0]) else 0
                end_val = float(y[-1]) if not np.isnan(y[-1]) else 0
                pct_change = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                
                return {
                    "column": value_col,
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "direction": "increasing" if slope > 0 else "decreasing",
                    "percent_change": float(pct_change),
                    "start_value": start_val,
                    "end_value": end_val,
                    "n_observations": len(data)
                }
            return None
        except Exception:
            return None
    
    def _detect_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect clusters in numeric data"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return {"clusters_detected": False, "message": "Need at least 2 numeric columns"}
            
            data = df[numeric_cols[:5]].dropna()
            if len(data) < 10:
                return {"clusters_detected": False, "message": "Insufficient data for clustering"}
            
            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Replace any remaining NaN/Inf with 0
            scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Try different k values
            best_k = 2
            best_score = -1
            
            for k in range(2, min(6, len(data))):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(scaled_data)
                    score = silhouette_score(scaled_data, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except Exception:
                    continue
            
            if best_score < 0:
                return {"clusters_detected": False, "message": "No meaningful clusters found"}
            
            # Final clustering
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_data)
            
            cluster_sizes = {}
            for i in range(best_k):
                cluster_sizes[f"cluster_{i}"] = int((labels == i).sum())
            
            return {
                "clusters_detected": True,
                "cluster_details": [{
                    "method": "kmeans",
                    "n_clusters": best_k,
                    "silhouette_score": float(best_score),
                    "cluster_sizes": cluster_sizes
                }]
            }
        except Exception as e:
            return {"clusters_detected": False, "message": f"Error: {str(e)}"}
    
    def _detect_associations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect associations between variables"""
        associations = {
            "categorical_associations": [],
            "numeric_categorical_associations": []
        }
        
        try:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Categorical associations (chi-square)
            for i, col1 in enumerate(cat_cols[:3]):
                for col2 in cat_cols[i+1:4]:
                    try:
                        contingency = pd.crosstab(df[col1], df[col2])
                        if contingency.size > 1:
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            if p_value < 0.05 and not np.isnan(chi2):
                                associations["categorical_associations"].append({
                                    "columns": [col1, col2],
                                    "chi_square": float(chi2),
                                    "p_value": float(p_value),
                                    "significant": True
                                })
                    except Exception:
                        continue
            
            # Numeric-categorical (ANOVA)
            for cat_col in cat_cols[:3]:
                for num_col in numeric_cols[:3]:
                    try:
                        groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col)]
                        groups = [g for g in groups if len(g) >= 2]
                        if len(groups) >= 2:
                            f_stat, p_value = stats.f_oneway(*groups)
                            if p_value < 0.05 and not np.isnan(f_stat):
                                associations["numeric_categorical_associations"].append({
                                    "categorical": cat_col,
                                    "numeric": num_col,
                                    "f_statistic": float(f_stat),
                                    "p_value": float(p_value),
                                    "significant": True
                                })
                    except Exception:
                        continue
        except Exception:
            pass
        
        return associations
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 1:
                return {"anomalies_detected": False, "message": "No numeric columns"}
            
            data = df[numeric_cols[:5]].dropna()
            if len(data) < 10:
                return {"anomalies_detected": False, "message": "Insufficient data"}
            
            # Standardize
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(scaled)
            
            anomaly_count = int((predictions == -1).sum())
            anomaly_pct = float(anomaly_count / len(data) * 100)
            
            return {
                "anomalies_detected": anomaly_count > 0,
                "anomaly_count": anomaly_count,
                "anomaly_percentage": anomaly_pct,
                "method": "isolation_forest"
            }
        except Exception as e:
            return {"anomalies_detected": False, "message": f"Error: {str(e)}"}
    
    def _summarize_patterns(self, results: Dict[str, Any]) -> List[str]:
        """Generate summary insights"""
        insights = []
        
        try:
            temporal = results.get("temporal_patterns", {})
            if temporal.get("trends"):
                insights.append(f"Found {len(temporal['trends'])} significant temporal trends")
        except Exception:
            pass
        
        try:
            clustering = results.get("clustering_patterns", {})
            if clustering.get("clusters_detected"):
                details = clustering.get("cluster_details", [{}])[0]
                insights.append(f"Identified {details.get('n_clusters', 0)} natural clusters in the data")
        except Exception:
            pass
        
        try:
            anomalies = results.get("anomaly_patterns", {})
            if anomalies.get("anomalies_detected"):
                insights.append(f"Detected {anomalies.get('anomaly_count', 0)} potential anomalies")
        except Exception:
            pass
        
        return insights
