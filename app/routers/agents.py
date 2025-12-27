"""
AI Agents Router - AI-powered data analysis
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.services.compute import compute_service

router = APIRouter(prefix="/datasets", tags=["agents"])


class AgentRequest(BaseModel):
    """Request for AI agent analysis"""
    agent_type: str  # "explorer", "pattern", "causal", "storyteller", "validator", "recommender"
    params: Optional[Dict[str, Any]] = {}


class AgentResponse(BaseModel):
    """Response from AI agent"""
    agent_type: str
    insights: List[str]
    recommendations: List[str]
    data: Optional[Dict[str, Any]] = None


@router.post("/{dataset_id}/agents/analyze", response_model=AgentResponse)
def run_agent_analysis(
    dataset_id: str,
    user_id: str,
    request: AgentRequest
):
    """
    Run AI agent analysis on dataset
    
    Available agents:
    - **explorer**: Initial data exploration and profiling
    - **pattern**: Pattern and anomaly detection
    - **causal**: Causal relationship analysis
    - **storyteller**: Generate narrative insights
    - **validator**: Data quality validation
    - **recommender**: Suggest next analysis steps
    
    Example:
    ```json
    {
      "agent_type": "explorer",
      "params": {}
    }
    ```
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing user_id")
    
    try:
        # Load dataset
        df = compute_service.load_dataframe(dataset_id, user_id)
        
        # Route to appropriate agent
        if request.agent_type == "explorer":
            return _run_explorer_agent(df, dataset_id)
        elif request.agent_type == "pattern":
            return _run_pattern_agent(df, dataset_id, request.params)
        elif request.agent_type == "causal":
            return _run_causal_agent(df, dataset_id, request.params)
        elif request.agent_type == "storyteller":
            return _run_storyteller_agent(df, dataset_id)
        elif request.agent_type == "validator":
            return _run_validator_agent(df, dataset_id)
        elif request.agent_type == "recommender":
            return _run_recommender_agent(df, dataset_id)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown agent type: {request.agent_type}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")


def _run_explorer_agent(df, dataset_id: str) -> AgentResponse:
    """Data Explorer Agent - Initial profiling"""
    insights = []
    recommendations = []
    
    # Basic insights
    insights.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
    
    # Missing data
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        insights.append(f"Found missing values in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}")
        recommendations.append("Consider handling missing values before analysis")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        insights.append(f"Found {len(numeric_cols)} numeric columns suitable for statistical analysis")
        recommendations.append("Run descriptive statistics on numeric columns")
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        insights.append(f"Found {len(categorical_cols)} categorical columns")
        recommendations.append("Explore group-based analysis using categorical variables")
    
    return AgentResponse(
        agent_type="explorer",
        insights=insights,
        recommendations=recommendations,
        data={
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_columns": missing_cols
        }
    )


def _run_pattern_agent(df, dataset_id: str, params: Dict) -> AgentResponse:
    """Pattern Detective Agent - Find patterns and anomalies"""
    insights = []
    recommendations = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        insights.append("No numeric columns found for pattern analysis")
        return AgentResponse(
            agent_type="pattern",
            insights=insights,
            recommendations=["Add numeric data for pattern detection"],
            data={}
        )
    
    # Check for outliers using IQR method
    from scipy import stats
    import numpy as np
    
    outlier_info = {}
    for col in numeric_cols[:5]:  # Limit to first 5 columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
        
        if len(outliers) > 0:
            outlier_info[col] = len(outliers)
            insights.append(f"Found {len(outliers)} potential outliers in '{col}'")
    
    if outlier_info:
        recommendations.append("Review outliers - may indicate data quality issues or interesting patterns")
    else:
        insights.append("No significant outliers detected in numeric columns")
    
    return AgentResponse(
        agent_type="pattern",
        insights=insights,
        recommendations=recommendations,
        data={"outliers": outlier_info}
    )


def _run_causal_agent(df, dataset_id: str, params: Dict) -> AgentResponse:
    """Causal Reasoner Agent - Analyze relationships"""
    insights = []
    recommendations = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        insights.append("Need at least 2 numeric columns for correlation analysis")
        return AgentResponse(
            agent_type="causal",
            insights=insights,
            recommendations=["Add more numeric variables for causal analysis"],
            data={}
        )
    
    # Calculate correlations
    corr_matrix = df[numeric_cols].corr()
    
    # Find strong correlations (|r| > 0.7)
    strong_correlations = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                strong_correlations.append({
                    "var1": numeric_cols[i],
                    "var2": numeric_cols[j],
                    "correlation": round(float(corr), 3)
                })
    
    if strong_correlations:
        insights.append(f"Found {len(strong_correlations)} strong correlations")
        for sc in strong_correlations[:3]:  # Show top 3
            insights.append(
                f"Strong {'positive' if sc['correlation'] > 0 else 'negative'} "
                f"correlation between '{sc['var1']}' and '{sc['var2']}' (r={sc['correlation']})"
            )
        recommendations.append("Investigate causal relationships with regression analysis")
    else:
        insights.append("No strong correlations detected")
        recommendations.append("Variables appear relatively independent")
    
    return AgentResponse(
        agent_type="causal",
        insights=insights,
        recommendations=recommendations,
        data={"strong_correlations": strong_correlations}
    )


def _run_storyteller_agent(df, dataset_id: str) -> AgentResponse:
    """Storyteller Agent - Generate narrative"""
    insights = []
    
    insights.append(f"This dataset tells the story of {len(df)} observations across {len(df.columns)} dimensions.")
    
    # Add more narrative elements based on data characteristics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        insights.append(f"The numerical narrative unfolds through {len(numeric_cols)} quantitative measures.")
    
    return AgentResponse(
        agent_type="storyteller",
        insights=insights,
        recommendations=["Generate visualizations to support the narrative"],
        data={}
    )


def _run_validator_agent(df, dataset_id: str) -> AgentResponse:
    """Validator Agent - Quality checks"""
    insights = []
    recommendations = []
    issues = []
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
        recommendations.append("Consider removing duplicates")
    
    # Check for missing data
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = missing_pct[missing_pct > 50].to_dict()
    if high_missing:
        issues.append(f"{len(high_missing)} columns have >50% missing data")
        recommendations.append("Review columns with high missingness")
    
    # Data quality score
    score = 100
    if duplicates > 0:
        score -= 10
    if high_missing:
        score -= 20 * len(high_missing)
    score = max(0, score)
    
    insights.append(f"Data quality score: {score}/100")
    if issues:
        insights.extend(issues)
    else:
        insights.append("No major data quality issues detected")
    
    return AgentResponse(
        agent_type="validator",
        insights=insights,
        recommendations=recommendations,
        data={"quality_score": score, "issues": issues}
    )


def _run_recommender_agent(df, dataset_id: str) -> AgentResponse:
    """Recommender Agent - Suggest next steps"""
    recommendations = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_cols:
        recommendations.append("Run descriptive statistics to understand distributions")
        
        if len(numeric_cols) >= 2:
            recommendations.append("Perform correlation analysis between numeric variables")
            recommendations.append("Consider regression modeling")
    
    if categorical_cols and numeric_cols:
        recommendations.append("Run ANOVA or t-tests to compare groups")
        recommendations.append("Create group-based visualizations")
    
    if len(df) > 1000:
        recommendations.append("Dataset size suitable for machine learning")
    
    return AgentResponse(
        agent_type="recommender",
        insights=[f"Analyzed dataset structure and identified {len(recommendations)} recommended next steps"],
        recommendations=recommendations,
        data={}
    )


@router.get("/{dataset_id}/agents/available")
def list_available_agents():
    """Get list of available AI agents"""
    return {
        "agents": [
            {
                "type": "explorer",
                "name": "Data Explorer",
                "description": "Initial data profiling and exploration"
            },
            {
                "type": "pattern",
                "name": "Pattern Detective",
                "description": "Detect patterns, trends, and anomalies"
            },
            {
                "type": "causal",
                "name": "Causal Reasoner",
                "description": "Analyze relationships and causality"
            },
            {
                "type": "storyteller",
                "name": "Data Storyteller",
                "description": "Generate narrative insights"
            },
            {
                "type": "validator",
                "name": "Quality Validator",
                "description": "Validate data quality and integrity"
            },
            {
                "type": "recommender",
                "name": "Analysis Recommender",
                "description": "Suggest next analysis steps"
            }
        ]
    }
