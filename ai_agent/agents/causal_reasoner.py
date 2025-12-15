"""
Causal Reasoner Agent - Infers causal relationships
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import LinearRegression
from scipy import stats
from core.orchestrator import BaseAgent, AgentRole


class CausalReasonerAgent(BaseAgent):
    """
    Infers potential causal relationships
    - Generates hypotheses
    - Tests directionality
    - Builds causal graph
    """
    
    def __init__(self):
        super().__init__(AgentRole.CAUSAL_REASONER)
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer causal relationships
        
        Args:
            inputs: {
                "data_context": {
                    "dataframe": pd.DataFrame
                },
                "patterns": Dict (optional),
                "domain_knowledge": str (optional)
            }
        
        Returns:
            {
                "hypotheses": [...],
                "causal_graph": {...},
                "potential_confounders": [...],
                "causal_insights": [...],
                "recommendations_for_validation": [...]
            }
        """
        df = inputs["data_context"]["dataframe"]
        domain = inputs.get("domain_knowledge", "general")
        
        self.log(f"Inferring causal relationships (domain: {domain})")
        
        # Generate hypotheses
        hypotheses = self._generate_hypotheses(df)
        
        # Test directionality
        causal_graph = self._build_causal_graph(df, hypotheses)
        
        # Identify confounders
        confounders = self._identify_confounders(df, causal_graph)
        
        # Generate insights
        insights = self._generate_causal_insights(causal_graph, confounders)
        
        return {
            "hypotheses": hypotheses,
            "causal_graph": causal_graph,
            "potential_confounders": confounders,
            "causal_insights": insights,
            "recommendations_for_validation": self._recommend_validation(hypotheses)
        }
    
    def _generate_hypotheses(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate causal hypotheses from correlations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        hypotheses = []
        
        # Look at strong correlations
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i+1:6]:
                data = df[[col1, col2]].dropna()
                if len(data) < 30:
                    continue
                
                corr, p_value = stats.pearsonr(data[col1], data[col2])
                
                if abs(corr) > 0.5 and p_value < 0.05:
                    hypotheses.append({
                        "var1": col1,
                        "var2": col2,
                        "correlation": float(corr),
                        "p_value": float(p_value),
                        "hypothesis": f"{col1} may influence {col2}" if corr > 0 else f"{col1} may inversely affect {col2}",
                        "confidence": "medium",
                        "type": "correlation_based"
                    })
        
        return hypotheses[:10]  # Limit to top 10
    
    def _build_causal_graph(
        self,
        df: pd.DataFrame,
        hypotheses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build causal graph using directionality tests"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        nodes = []
        edges = []
        
        # Test directionality for each hypothesis
        for hyp in hypotheses:
            var1 = hyp["var1"]
            var2 = hyp["var2"]
            
            data = df[[var1, var2]].dropna()
            if len(data) < 30:
                continue
            
            # Test both directions
            r2_forward = self._test_direction(data[var1], data[var2])
            r2_reverse = self._test_direction(data[var2], data[var1])
            
            # If forward prediction is significantly better
            if r2_forward > r2_reverse * 1.3 and r2_forward > 0.3:
                if var1 not in nodes:
                    nodes.append(var1)
                if var2 not in nodes:
                    nodes.append(var2)
                
                confidence = "high" if r2_forward > r2_reverse * 2 else "medium"
                
                edges.append({
                    "from": var1,
                    "to": var2,
                    "strength": float(r2_forward),
                    "confidence": confidence,
                    "evidence": f"R² forward: {r2_forward:.3f}, R² reverse: {r2_reverse:.3f}"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "n_relationships": len(edges)
        }
    
    def _test_direction(self, cause: pd.Series, effect: pd.Series) -> float:
        """Test predictive power in one direction"""
        X = cause.values.reshape(-1, 1)
        y = effect.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.score(X, y)
    
    def _identify_confounders(
        self,
        df: pd.DataFrame,
        causal_graph: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential confounding variables"""
        confounders = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        edges = causal_graph.get("edges", [])
        
        for edge in edges[:5]:  # Check first 5 edges
            cause = edge["from"]
            effect = edge["to"]
            
            # Look for variables correlated with both
            for col in numeric_cols:
                if col == cause or col == effect:
                    continue
                
                data = df[[cause, effect, col]].dropna()
                if len(data) < 30:
                    continue
                
                try:
                    corr_cause, p1 = stats.pearsonr(data[col], data[cause])
                    corr_effect, p2 = stats.pearsonr(data[col], data[effect])
                    
                    # If correlated with both
                    if abs(corr_cause) > 0.4 and abs(corr_effect) > 0.4:
                        if p1 < 0.05 and p2 < 0.05:
                            confounders.append({
                                "variable": col,
                                "affects_cause": cause,
                                "affects_effect": effect,
                                "corr_with_cause": float(corr_cause),
                                "corr_with_effect": float(corr_effect),
                                "likelihood": "high" if min(abs(corr_cause), abs(corr_effect)) > 0.6 else "medium"
                            })
                except:
                    continue
        
        return confounders[:5]  # Top 5
    
    def _generate_causal_insights(
        self,
        causal_graph: Dict[str, Any],
        confounders: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from causal analysis"""
        insights = []
        
        n_relationships = causal_graph.get("n_relationships", 0)
        if n_relationships > 0:
            insights.append(f"Identified {n_relationships} potential causal relationships")
        
        high_conf = [e for e in causal_graph.get("edges", []) if e["confidence"] == "high"]
        if high_conf:
            insights.append(f"{len(high_conf)} relationships have high confidence")
        
        if confounders:
            insights.append(f"Found {len(confounders)} potential confounding variables")
        
        return insights
    
    def _recommend_validation(self, hypotheses: List[Dict[str, Any]]) -> List[str]:
        """Recommend validation approaches"""
        recommendations = [
            "Consider A/B testing to validate causal claims",
            "Collect time-series data to establish temporal precedence",
            "Control for identified confounders in regression analysis"
        ]
        
        if hypotheses:
            recommendations.append(f"Focus validation on top {min(3, len(hypotheses))} hypotheses")
        
        return recommendations
