from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from app.auth.supabase_jwt import get_current_user
from app.models.pipelines import PipelineCreateRequest, PipelineResponse
from app.engine.pipeline import pipeline_hash
from app.db import registry
from app.transformers.registry import transformer_registry

router = APIRouter()

# ---------------------------------------------------------------------
# Power BI-style "capabilities discovery" endpoints
# ---------------------------------------------------------------------
# These endpoints are READ-ONLY. They do not touch the database.
#
# Why they exist:
# - Frontend (Lovable) can render a menu of available transforms
# - LLM planner can stop hallucinating op names and args
#
# GET /v2/pipelines/ops
# GET /v2/pipelines/ops/{op}
# ---------------------------------------------------------------------

# Minimal metadata for common ops. If an op is registered but not listed
# here, it will still appear (just with limited description/schema).
OPS_META: Dict[str, Dict[str, Any]] = {
    # Core shaping
    "select_columns": {
        "category": "shape",
        "description": "Select a subset of columns.",
        "args_schema": {"columns": ["colA", "colB"]},
        "example": {"op": "select_columns", "args": {"columns": ["age", "los_days"]}},
    },
    "drop_columns": {
        "category": "shape",
        "description": "Drop columns by providing a keep-list (remaining columns).",
        "args_schema": {"keep": ["colA", "colB"]},
        "example": {"op": "drop_columns", "args": {"keep": ["patient_id", "age"]}},
    },
    "rename_columns": {
        "category": "shape",
        "description": "Rename columns using a mapping old->new. Requires a columns list in args.",
        "args_schema": {"mapping": {"old": "new"}, "columns": ["..."]},
        "example": {
            "op": "rename_columns",
            "args": {"mapping": {"Lengthofstay Days": "los_days"}, "columns": ["Lengthofstay Days", "patient_id"]},
        },
    },
    # Filtering
    "filter_rows": {
        "category": "filter",
        "description": "Filter rows using a raw SQL predicate (legacy). Prefer filter_rows_safe when possible.",
        "args_schema": {"where": "col > 3 AND unit = 'ICU'"},
        "example": {"op": "filter_rows", "args": {"where": "los_days > 3"}},
        "safe": False,
    },
    "filter_rows_safe": {
        "category": "filter",
        "description": "Filter rows using a SAFE expression tree (AST).",
        "args_schema": {"expr": {"op": ">", "left": {"col": "los_days"}, "right": {"val": 3}}},
        "example": {
            "op": "filter_rows_safe",
            "args": {"expr": {"op": ">", "left": {"col": "los_days"}, "right": {"val": 3}}},
        },
        "safe": True,
    },
    # Computed columns
    "add_computed_column": {
        "category": "compute",
        "description": "Add a computed column using raw SQL expression (legacy). Prefer add_computed_safe when possible.",
        "args_schema": {"name": "new_col", "expr": "a * 24"},
        "example": {"op": "add_computed_column", "args": {"name": "los_hours", "expr": "los_days * 24"}},
        "safe": False,
    },
    "add_computed_safe": {
        "category": "compute",
        "description": "Add a computed column using a SAFE expression tree (AST).",
        "args_schema": {"name": "new_col", "expr": {"op": "*", "left": {"col": "a"}, "right": {"val": 24}}},
        "example": {"op": "add_computed_safe", "args": {"name": "los_hours", "expr": {"op": "*", "left": {"col": "los_days"}, "right": {"val": 24}}}},
        "safe": True,
    },
    # Missing values
    "drop_nulls": {
        "category": "clean",
        "description": "Drop rows where any of the specified columns are NULL.",
        "args_schema": {"columns": ["colA", "colB"]},
        "example": {"op": "drop_nulls", "args": {"columns": ["los_days"]}},
    },
    "fill_nulls": {
        "category": "clean",
        "description": "Fill NULLs for specified columns (Power BI 'Replace Values'/fill pattern).",
        "args_schema": {"fills": [{"column": "colA", "value": 0}], "columns": ["..."]},
    },
    # Aggregation
    "group_aggregate": {
        "category": "aggregate",
        "description": "Group by one or more columns and compute aggregates (avg/sum/count/etc.).",
        "args_schema": {
            "group_by": ["group_col"],
            "aggs": [{"fn": "avg", "column": "measure_col", "as": "avg_measure"}],
        },
        "example": {
            "op": "group_aggregate",
            "args": {"group_by": ["unit"], "aggs": [{"fn": "avg", "column": "los_days", "as": "avg_los"}]},
        },
    },
    # Datetime
    "date_from_text": {
        "category": "datetime",
        "description": "Parse a text column into a timestamp/date using a strptime-like format.",
        "args_schema": {"column": "timestamp_text", "format": "%m/%d/%y %H:%M", "as": "timestamp_col"},
        "example": {"op": "date_from_text", "args": {"column": "admit_time", "format": "%m/%d/%y %H:%M", "as": "admit_ts"}},
    },
    "date_part": {
        "category": "datetime",
        "description": "Extract a datetime part (year, month, day, hour, weekday, etc.) into a new column.",
        "args_schema": {"column": "timestamp_col", "part": "month|hour|weekday|year", "as": "new_col"},
        "example": {"op": "date_part", "args": {"column": "admit_ts", "part": "month", "as": "month_num"}},
    },
    "format_datetime": {
        "category": "datetime",
        "description": "Format a timestamp into text using strftime (e.g., %B for month name, %I %p for hour bucket).",
        "args_schema": {"column": "timestamp_col", "format": "%B", "as": "month_name"},
        "example": {"op": "format_datetime", "args": {"column": "admit_ts", "format": "%B", "as": "month"}},
    },
}


def _list_registered_ops() -> List[str]:
    # Support both older registry (private _ops) and newer registry (available_ops()).
    if hasattr(transformer_registry, "available_ops"):
        return list(transformer_registry.available_ops())  # type: ignore[attr-defined]
    return sorted(list(getattr(transformer_registry, "_ops", {}).keys()))


@router.get("/ops")
async def list_ops() -> Dict[str, Any]:
    ops = _list_registered_ops()
    items: List[Dict[str, Any]] = []

    for op in ops:
        meta = OPS_META.get(op, {})
        items.append(
            {
                "op": op,
                "category": meta.get("category", "other"),
                "description": meta.get("description", ""),
                "safe": bool(meta.get("safe", op.endswith("_safe"))),
                "args_schema": meta.get("args_schema"),
                "example": meta.get("example"),
            }
        )

    return {"count": len(items), "ops": items}


@router.get("/ops/{op}")
async def get_op(op: str) -> Dict[str, Any]:
    # Ensure it's registered
    try:
        transformer_registry.get(op)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    meta = OPS_META.get(op, {})
    return {
        "op": op,
        "category": meta.get("category", "other"),
        "description": meta.get("description", ""),
        "safe": bool(meta.get("safe", op.endswith("_safe"))),
        "args_schema": meta.get("args_schema"),
        "example": meta.get("example"),
    }


# ---------------------------------------------------------------------
# Existing pipeline create endpoint (unchanged)
# ---------------------------------------------------------------------
@router.post("/{dataset_id}", response_model=PipelineResponse)
async def create_pipeline(dataset_id: str, req: PipelineCreateRequest, user=Depends(get_current_user)):
    user_id = user["user_id"]
    ph = pipeline_hash(req.steps)
    pipeline_id = str(uuid.uuid4())
    try:
        await registry.execute(
            """INSERT INTO pipelines (id, dataset_id, user_id, name, steps_json, pipeline_hash)
               VALUES ($1,$2,$3,$4,$5,$6)""",
            pipeline_id, dataset_id, user_id, req.name, [s.model_dump() for s in req.steps], ph
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PipelineResponse(pipeline_id=pipeline_id, dataset_id=dataset_id, name=req.name, steps=req.steps, pipeline_hash=ph)
