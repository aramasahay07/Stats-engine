from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='mae-func',
    topic_id='topic-id',
    topic_slug='predictive-ml',
    slug='mae',
    title='Mean Absolute Error',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['mae'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Mean Absolute Error - fully functional implementation."""
    from sklearn.metrics import mean_absolute_error
    
    y_true_col = params.get('y_true_column')
    y_pred_col = params.get('y_pred_column')
    
    query = f"SELECT {y_true_col}, {y_pred_col} FROM dataset WHERE {y_true_col} IS NOT NULL AND {y_pred_col} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    y_true = [r[0] for r in data]
    y_pred = [r[1] for r in data]
    
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'mae': float(mae),
        'mean_absolute_error': float(mae),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
