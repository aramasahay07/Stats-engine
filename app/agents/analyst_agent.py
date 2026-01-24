from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from app.db import registry


def _coerce_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            return default
    return default


def _lower_tokens(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [t.strip() for t in text.lower().replace("/", " ").replace("-", " ").split() if t.strip()]


def _profile_numeric_columns(profile_json: Dict[str, Any]) -> List[str]:
    numeric_summary = profile_json.get("numeric_summary") if isinstance(profile_json, dict) else {}
    if isinstance(numeric_summary, dict):
        return [str(k) for k in numeric_summary.keys()]
    return []


def _schema_numeric_columns(schema_json: Iterable[dict]) -> List[str]:
    numeric = []
    for col in schema_json or []:
        if not isinstance(col, dict):
            continue
        dtype = str(col.get("dtype") or col.get("type") or "").lower()
        name = col.get("name")
        if name and any(token in dtype for token in ("int", "float", "double", "decimal", "numeric", "real")):
            numeric.append(str(name))
    return numeric


def _infer_time_columns(schema_json: Iterable[dict]) -> List[str]:
    time_cols = []
    for col in schema_json or []:
        if not isinstance(col, dict):
            continue
        name = col.get("name")
        dtype = str(col.get("dtype") or col.get("type") or "").lower()
        if name and any(token in dtype for token in ("date", "time", "timestamp")):
            time_cols.append(str(name))
    return time_cols


def _select_test_rules(
    question: str,
    schema_json: List[dict],
    profile_json: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tokens = _lower_tokens(question)
    numeric_cols = _schema_numeric_columns(schema_json)
    if not numeric_cols and profile_json:
        numeric_cols = _profile_numeric_columns(profile_json)
    if context:
        preferred = context.get("preferred_tests") or context.get("preferred_test")
        if isinstance(preferred, str):
            preferred = [preferred]
    else:
        preferred = None

    if preferred:
        return {"analysis": preferred[0], "params": {}, "missing": []}

    if any(t in tokens for t in ("correlation", "relationship", "association")):
        params: Dict[str, Any] = {}
        missing = []
        if len(numeric_cols) >= 2:
            params["columns"] = numeric_cols[:2]
        else:
            missing = ["two numeric columns"]
        return {"analysis": "correlation", "params": params, "missing": missing}

    if any(t in tokens for t in ("compare", "difference", "vs", "versus")):
        params: Dict[str, Any] = {}
        missing = []
        if len(numeric_cols) >= 2:
            params["x"] = numeric_cols[0]
            params["y"] = numeric_cols[1]
        else:
            missing = ["two numeric columns"]
        return {"analysis": "ttest_2samp", "params": params, "missing": missing}

    if any(t in tokens for t in ("trend", "time", "over", "season")):
        time_cols = _infer_time_columns(schema_json)
        params: Dict[str, Any] = {}
        missing = []
        if numeric_cols:
            params["value_col"] = numeric_cols[0]
        else:
            missing.append("numeric value column")
        if time_cols:
            params["time_col"] = time_cols[0]
        else:
            missing.append("time column")
        return {"analysis": "trend_analysis", "params": params, "missing": missing}

    if any(t in tokens for t in ("predict", "regression", "model")):
        params: Dict[str, Any] = {}
        missing = []
        if len(numeric_cols) >= 2:
            params["y"] = numeric_cols[0]
            params["X"] = numeric_cols[1:3]
        else:
            missing = ["target and feature columns"]
        return {"analysis": "regression_ols", "params": params, "missing": missing}

    if any(t in tokens for t in ("distribution", "normal", "normality")):
        params: Dict[str, Any] = {}
        missing = []
        if numeric_cols:
            params["column"] = numeric_cols[0]
        else:
            missing = ["numeric column"]
        return {"analysis": "normality_test", "params": params, "missing": missing}

    # Default: descriptive summary
    params = {}
    missing = []
    if numeric_cols:
        params["columns"] = numeric_cols[:3]
    else:
        missing = ["numeric columns"]
    return {"analysis": "descriptives", "params": params, "missing": missing}


async def select_analysis(
    *,
    dataset_id: str,
    user_id: str,
    question: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Determine the analysis to run for a given dataset and question.

    Returns a dict with:
      - analysis: name of the test
      - params: arguments for stats_service
      - needs_info: optional list of missing requirements
    """
    row_any = await registry.fetchrow(
        """
        SELECT dataset_id, user_id, schema_json, profile_json
        FROM datasets
        WHERE dataset_id = $1::uuid
        """,
        dataset_id,
    )

    if not row_any:
        raise ValueError("Dataset not found")

    row_user_id = row_any.get("user_id") if hasattr(row_any, "get") else row_any["user_id"]
    if row_user_id != user_id:
        raise PermissionError("Access denied")

    schema_json = _coerce_json(row_any.get("schema_json") if hasattr(row_any, "get") else row_any["schema_json"], [])
    profile_json = _coerce_json(row_any.get("profile_json") if hasattr(row_any, "get") else row_any["profile_json"], {})

    if not schema_json:
        schema_json = profile_json.get("schema") if isinstance(profile_json, dict) else []

    selection = _select_test_rules(question, list(schema_json), profile_json, context)
    missing = selection.get("missing") or []
    if missing:
        return {
            "analysis": selection["analysis"],
            "params": selection.get("params", {}),
            "needs_info": missing,
        }

    return {"analysis": selection["analysis"], "params": selection.get("params", {})}
