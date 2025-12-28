from __future__ import annotations
import hashlib
import json
from typing import List
from app.models.pipelines import PipelineStep
from app.transformers.registry import transformer_registry
from app.engine.duckdb_engine import DuckDBEngine

def pipeline_hash(steps: List[PipelineStep]) -> str:
    payload = json.dumps([s.model_dump() for s in steps], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def compile_pipeline_sql(base_view: str, steps: List[PipelineStep]) -> str:
    """Compile steps into a single SELECT SQL statement.

    We follow a PowerQuery-like pattern: each step wraps the previous step.
    """
    current = f"SELECT * FROM {base_view}"
    for step in steps:
        t = transformer_registry.get(step.op)
        current = t.apply_sql(current, step.args)
    return current

def ensure_pipeline_view(con, dataset_id: str, base_view: str, steps: List[PipelineStep]) -> str:
    ph = pipeline_hash(steps)
    view = DuckDBEngine.pipeline_view_name(dataset_id, ph)
    sql = compile_pipeline_sql(base_view, steps)
    con.execute(f"CREATE OR REPLACE VIEW {view} AS {sql}")
    return view
