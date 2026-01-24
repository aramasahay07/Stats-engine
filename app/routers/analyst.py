"""
AI Analyst Router.

Provides the /datasets/{dataset_id}/analyst endpoint for AI-assisted statistical analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.supabase_jwt import get_current_user
from app.config import settings, AUTH_DISABLED
from app.db import registry
from app.services.stats_service import run_stats
from app.agents import (
    AIAnalystAgent,
    AnalystRequest,
    AnalystResponse,
    DatasetInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_dataset_with_profile(dataset_id: str, user_id: str) -> Dict[str, Any]:
    """
    Fetch dataset metadata including profile_json for the analyst.

    Similar to validate_dataset_ready but also retrieves profile data.
    """
    row = await registry.fetchrow(
        """
        SELECT
            dataset_id,
            user_id,
            file_name,
            n_rows,
            n_cols,
            parquet_ref,
            state,
            schema_json,
            profile_json,
            error_message
        FROM datasets
        WHERE dataset_id = $1::uuid
        """,
        dataset_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row_user_id = row.get("user_id") if hasattr(row, "get") else row["user_id"]
    if row_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    state = (row.get("state") if hasattr(row, "get") else row["state"]) or "ready"

    if state == "reprocessing":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DATASET_REPROCESSING",
                "message": "Dataset is being reprocessed. Please wait.",
            },
        )

    if state == "processing":
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset is still processing."},
        )

    if state == "failed":
        err = row.get("error_message") if hasattr(row, "get") else row["error_message"]
        raise HTTPException(
            status_code=422,
            detail={"code": "DATASET_FAILED", "message": err or "Dataset processing failed"},
        )

    parquet_ref = row.get("parquet_ref") if hasattr(row, "get") else row["parquet_ref"]
    if not parquet_ref:
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset parquet is not ready yet."},
        )

    return dict(row)


def _extract_schema_from_profile(profile_json: Optional[Dict]) -> List[Dict[str, Any]]:
    """Extract schema with column info from profile."""
    if not profile_json:
        return []

    schema = profile_json.get("schema", [])
    if isinstance(schema, list):
        return schema

    return []


async def _get_data_sample(
    user_id: str,
    dataset_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get sample data for visualizations."""
    try:
        # Use existing profile sample if available
        row = await registry.fetchrow(
            """
            SELECT profile_json
            FROM datasets
            WHERE dataset_id = $1::uuid AND user_id = $2
            """,
            dataset_id,
            user_id,
        )

        if row:
            profile = row.get("profile_json") if hasattr(row, "get") else row["profile_json"]
            if profile and "sample_rows" in profile:
                sample = profile["sample_rows"]
                if isinstance(sample, list):
                    return sample[:limit]

        return []
    except Exception as e:
        logger.warning("Failed to fetch data sample for dataset %s: %s", dataset_id, str(e))
        return []


@router.post("/{dataset_id}/analyst", response_model=AnalystResponse)
async def analyze_dataset(
    dataset_id: str,
    req: AnalystRequest,
    user_id: Optional[str] = Query(None),
    user: Optional[dict] = Depends(get_current_user) if not AUTH_DISABLED else None,
):
    """
    AI-assisted statistical analysis endpoint.

    This endpoint:
    1. Analyzes the dataset profile and question
    2. Selects the appropriate statistical test
    3. Executes the analysis via the stats engine
    4. Generates explanations and visualizations
    5. Returns a comprehensive response

    Auth:
    - Prefers JWT authentication via Authorization header
    - Falls back to user_id query parameter for legacy compatibility
    """
    # Determine user_id from JWT or query param
    effective_user_id: Optional[str] = None

    if user and isinstance(user, dict):
        effective_user_id = user.get("user_id")

    if not effective_user_id and user_id:
        effective_user_id = user_id

    if not effective_user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide JWT token or user_id parameter.",
        )

    try:
        # Fetch dataset with profile
        dataset_row = await _get_dataset_with_profile(dataset_id, effective_user_id)

        # Build dataset info
        profile_json = dataset_row.get("profile_json") or {}
        schema_json = dataset_row.get("schema_json") or []

        # Prefer profile schema if available, fall back to schema_json
        schema = _extract_schema_from_profile(profile_json)
        if not schema and schema_json:
            if isinstance(schema_json, list):
                schema = schema_json
            elif isinstance(schema_json, dict) and "columns" in schema_json:
                schema = schema_json["columns"]

        dataset_info = DatasetInfo(
            dataset_id=str(dataset_row.get("dataset_id")),
            user_id=effective_user_id,
            file_name=dataset_row.get("file_name", ""),
            n_rows=dataset_row.get("n_rows") or 0,
            n_cols=dataset_row.get("n_cols") or 0,
            schema=schema,
            profile=profile_json,
            parquet_ref=dataset_row.get("parquet_ref"),
            parquet_sha=profile_json.get("parquet_sha"),
        )

        # Get sample data for visualizations
        data_sample = await _get_data_sample(effective_user_id, dataset_id)

        # Create analyst agent
        agent = AIAnalystAgent(
            openai_api_key=settings.openai_api_key,
        )

        # Define run_stats wrapper
        async def run_stats_wrapper(
            user_id: str,
            dataset_id: str,
            analysis: str,
            params: Dict[str, Any],
        ):
            return await run_stats(
                user_id=user_id,
                dataset_id=dataset_id,
                analysis=analysis,
                params=params,
            )

        # Execute analysis
        response = await agent.analyze(
            request=req,
            dataset_info=dataset_info,
            run_stats_func=run_stats_wrapper,
            data_sample=data_sample,
        )

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Unexpected error in analyst endpoint for dataset %s: %s", dataset_id, str(e))
        raise HTTPException(
            status_code=500,
            detail={"code": "ANALYSIS_ERROR", "message": "An unexpected error occurred during analysis"}
        )


@router.get("/{dataset_id}/analyst/available-tests")
async def get_available_tests(
    dataset_id: str,
    user_id: Optional[str] = Query(None),
    user: Optional[dict] = Depends(get_current_user) if not AUTH_DISABLED else None,
):
    """
    Get available statistical tests for a dataset based on its schema.

    Returns suggestions for appropriate tests given the dataset's columns.
    """
    # Determine user_id
    effective_user_id: Optional[str] = None
    if user and isinstance(user, dict):
        effective_user_id = user.get("user_id")
    if not effective_user_id and user_id:
        effective_user_id = user_id
    if not effective_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Fetch dataset
        dataset_row = await _get_dataset_with_profile(dataset_id, effective_user_id)
        profile_json = dataset_row.get("profile_json") or {}
        schema = _extract_schema_from_profile(profile_json)

        # Categorize columns
        numeric_cols = [c["name"] for c in schema if c.get("role") == "numeric"]
        categorical_cols = [c["name"] for c in schema if c.get("role") == "categorical"]
        datetime_cols = [c["name"] for c in schema if c.get("role") == "datetime"]

        suggestions = []

        # Two-group comparison
        if numeric_cols and categorical_cols:
            suggestions.append({
                "test": "two-sample-t-test",
                "name": "Two-Sample T-Test",
                "description": "Compare means between two groups",
                "requires": {"numeric": 1, "categorical": 1},
                "suggested_columns": {
                    "measure_column": numeric_cols[0] if numeric_cols else None,
                    "group_column": categorical_cols[0] if categorical_cols else None,
                },
            })

        # Multi-group comparison
        if numeric_cols and categorical_cols:
            suggestions.append({
                "test": "anova-one-way",
                "name": "One-Way ANOVA",
                "description": "Compare means across multiple groups",
                "requires": {"numeric": 1, "categorical": 1},
                "suggested_columns": {
                    "measure_column": numeric_cols[0] if numeric_cols else None,
                    "group_column": categorical_cols[0] if categorical_cols else None,
                },
            })

        # Correlation
        if len(numeric_cols) >= 2:
            suggestions.append({
                "test": "pearson-correlation",
                "name": "Pearson Correlation",
                "description": "Measure linear relationship between two numeric variables",
                "requires": {"numeric": 2},
                "suggested_columns": {
                    "x": numeric_cols[0],
                    "y": numeric_cols[1],
                },
            })

        # Regression
        if len(numeric_cols) >= 2:
            suggestions.append({
                "test": "simple-linear-regression",
                "name": "Simple Linear Regression",
                "description": "Predict one numeric variable from another",
                "requires": {"numeric": 2},
                "suggested_columns": {
                    "x": numeric_cols[0],
                    "y": numeric_cols[1],
                },
            })

        # Chi-square
        if len(categorical_cols) >= 2:
            suggestions.append({
                "test": "chi-square-test",
                "name": "Chi-Square Test",
                "description": "Test independence between two categorical variables",
                "requires": {"categorical": 2},
                "suggested_columns": {
                    "x": categorical_cols[0],
                    "y": categorical_cols[1],
                },
            })

        # Time series
        if datetime_cols and numeric_cols:
            suggestions.append({
                "test": "moving-average",
                "name": "Moving Average",
                "description": "Analyze trends over time",
                "requires": {"datetime": 1, "numeric": 1},
                "suggested_columns": {
                    "time_column": datetime_cols[0],
                    "measure_column": numeric_cols[0],
                },
            })

        # Descriptive stats (always available for numeric)
        if numeric_cols:
            suggestions.append({
                "test": "mean",
                "name": "Descriptive Statistics",
                "description": "Calculate summary statistics",
                "requires": {"numeric": 1},
                "suggested_columns": {
                    "column": numeric_cols[0],
                },
            })

        return {
            "dataset_id": dataset_id,
            "column_summary": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols,
            },
            "suggested_tests": suggestions,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in available-tests endpoint for dataset %s: %s", dataset_id, str(e))
        raise HTTPException(
            status_code=500,
            detail={"code": "ANALYSIS_ERROR", "message": "An unexpected error occurred"}
        )
