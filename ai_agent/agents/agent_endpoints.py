"""
Agent Endpoints - Individual agent API wrappers
Provides clean interfaces for each agent to be called via REST API
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from ai_agent.agents.data_explorer import DataExplorerAgent
from ai_agent.agents.pattern_detective import PatternDetectiveAgent
from ai_agent.agents.causal_reasoner import CausalReasonerAgent
from ai_agent.core.orchestrator import AgentRole


class AgentEndpoints:
    """
    Wrapper class for individual agent endpoints
    Each method corresponds to a REST API endpoint
    """
    
    def __init__(self):
        self.explorer = DataExplorerAgent()
        self.detective = PatternDetectiveAgent()
        self.reasoner = CausalReasonerAgent()
    
    async def explore(
        self,
        df: pd.DataFrame,
        session_id: str,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Data exploration endpoint
        Returns: data profile, quality assessment, interesting features
        """
        inputs = {
            "data_context": {
                "dataframe": df,
                "filename": f"session_{session_id}"
            },
            "focus": focus_areas or ["exploratory"]
        }
        
        result = await self.explorer.execute(inputs)
        
        # Format for API response
        return {
            "agent": "data_explorer",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "data_profile": result.get("data_profile", {}),
            "quality_report": result.get("quality_report", {}),
            "interesting_features": result.get("interesting_features", []),
            "preliminary_insights": result.get("preliminary_insights", [])
        }
    
    async def detect_patterns(
        self,
        df: pd.DataFrame,
        session_id: str,
        exploration_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Pattern detection endpoint
        Returns: trends, clusters, associations, anomalies
        """
        inputs = {
            "data_context": {
                "dataframe": df,
                "filename": f"session_{session_id}"
            },
            "exploration_insights": exploration_results or {}
        }
        
        result = await self.detective.execute(inputs)
        
        # Format for API response
        return {
            "agent": "pattern_detective",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "temporal_patterns": result.get("temporal_patterns", {}),
            "clustering_patterns": result.get("clustering_patterns", {}),
            "association_patterns": result.get("association_patterns", {}),
            "anomaly_patterns": result.get("anomaly_patterns", {}),
            "interaction_patterns": result.get("interaction_patterns", {}),
            "summary_insights": result.get("summary_insights", [])
        }
    
    async def infer_causality(
        self,
        df: pd.DataFrame,
        session_id: str,
        patterns: Optional[Dict] = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Causal inference endpoint
        Returns: hypotheses, causal links, confounders
        """
        inputs = {
            "data_context": {
                "dataframe": df,
                "filename": f"session_{session_id}"
            },
            "patterns": patterns or {},
            "domain_knowledge": domain
        }
        
        result = await self.reasoner.execute(inputs)
        
        # Format for API response
        return {
            "agent": "causal_reasoner",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "hypotheses": result.get("hypotheses", []),
            "causal_graph": result.get("causal_graph", {}),
            "potential_confounders": result.get("potential_confounders", []),
            "causal_insights": result.get("causal_insights", []),
            "validation_recommendations": result.get("recommendations_for_validation", [])
        }


def format_findings_json(
    exploration: Optional[Dict] = None,
    patterns: Optional[Dict] = None,
    causality: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Combine agent results into unified Findings JSON format
    This is what gets sent to Edge Function → Claude
    """
    
    findings = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "agents_used": []
        },
        "statistics": {},
        "patterns": {},
        "tests": {}
    }
    
    # Add exploration data
    if exploration:
        findings["metadata"]["agents_used"].append("data_explorer")
        findings["metadata"]["session_id"] = exploration.get("session_id")
        
        # Data quality
        quality = exploration.get("quality_report", {})
        findings["metadata"]["quality_score"] = quality.get("overall_score", 0)
        findings["metadata"]["confidence"] = _confidence_from_quality(quality.get("overall_score", 0))
        findings["metadata"]["caveats"] = [
            issue.get("message", "") 
            for issue in quality.get("issues", [])
        ]
        
        # Data shape
        profile = exploration.get("data_profile", {})
        shape = profile.get("shape", {})
        findings["metadata"]["n_rows"] = shape.get("rows", 0)
        findings["metadata"]["n_cols"] = shape.get("columns", 0)
        
        # Descriptive stats from profile
        findings["statistics"]["descriptive"] = _extract_descriptive_stats(profile)
    
    # Add pattern detection data
    if patterns:
        findings["metadata"]["agents_used"].append("pattern_detective")
        
        # Temporal patterns
        temporal = patterns.get("temporal_patterns", {})
        if temporal.get("trends"):
            findings["patterns"]["trends"] = {
                "detected_trends": _format_trends(temporal.get("trends", []))
            }
        
        # Clustering
        clustering = patterns.get("clustering_patterns", {})
        if clustering.get("clusters_detected"):
            findings["patterns"]["clusters"] = _format_clusters(clustering)
        
        # Associations
        associations = patterns.get("association_patterns", {})
        findings["patterns"]["associations"] = {
            "categorical": associations.get("categorical_associations", []),
            "numeric_categorical": associations.get("numeric_categorical_associations", [])
        }
        
        # Anomalies
        anomalies = patterns.get("anomaly_patterns", {})
        if anomalies.get("anomalies_detected"):
            findings["patterns"]["anomalies"] = anomalies
    
    # Add causal inference data
    if causality:
        findings["metadata"]["agents_used"].append("causal_reasoner")
        
        # Causal links
        causal_graph = causality.get("causal_graph", {})
        edges = causal_graph.get("edges", [])
        
        if edges:
            findings["tests"]["causal_links"] = {
                "links": _format_causal_links(edges)
            }
        
        # Hypotheses
        hypotheses = causality.get("hypotheses", [])
        findings["tests"]["hypotheses"] = [
            h for h in hypotheses 
            if h.get("confidence") in ["high", "medium"]
        ]
    
    return findings


def _confidence_from_quality(quality_score: float) -> str:
    """Convert quality score to confidence level"""
    if quality_score > 80:
        return "high"
    elif quality_score > 60:
        return "medium"
    else:
        return "low"


def _extract_descriptive_stats(profile: Dict) -> Dict[str, Any]:
    """Extract descriptive stats from data profile"""
    descriptive = {}
    
    columns = profile.get("columns", {})
    for col_name, col_info in columns.items():
        if col_info.get("role") == "numeric":
            descriptive[col_name] = {
                "mean": col_info.get("mean"),
                "median": col_info.get("median"),
                "std": col_info.get("std"),
                "min": col_info.get("min"),
                "max": col_info.get("max"),
                "q25": col_info.get("q25"),
                "q75": col_info.get("q75"),
                "missing_pct": col_info.get("missing_pct", 0),
                "outlier_count": col_info.get("outlier_count", 0)
            }
    
    return descriptive


def _format_trends(trends: List[Dict]) -> List[Dict]:
    """Format trends for Findings JSON"""
    formatted = []
    
    for trend in trends:
        formatted.append({
            "variable": trend.get("column"),
            "slope": trend.get("slope"),
            "r_squared": trend.get("r_squared"),
            "p_value": trend.get("p_value"),
            "percent_change": trend.get("percent_change"),
            "direction": trend.get("direction"),
            "start_value": trend.get("start_value"),
            "end_value": trend.get("end_value"),
            "n_observations": trend.get("n_observations"),
            "is_significant": trend.get("p_value", 1) < 0.05
        })
    
    return formatted


def _format_clusters(clustering: Dict) -> Dict[str, Any]:
    """Format clustering results for Findings JSON"""
    details = clustering.get("cluster_details", [])
    
    if not details:
        return {}
    
    first_result = details[0]
    
    return {
        "n_clusters": first_result.get("n_clusters", 0),
        "method": first_result.get("method", "kmeans"),
        "silhouette_score": first_result.get("silhouette_score", 0),
        "cluster_sizes": first_result.get("cluster_sizes", {}),
        "distinguishing_features": first_result.get("cluster_characteristics", [])
    }


def _format_causal_links(edges: List[Dict]) -> List[Dict]:
    """Format causal links for Findings JSON"""
    formatted = []
    
    for edge in edges:
        formatted.append({
            "cause": edge.get("from"),
            "effect": edge.get("to"),
            "strength": edge.get("strength"),
            "confidence": edge.get("confidence"),
            "evidence": edge.get("evidence")
        })
    
    return formatted
