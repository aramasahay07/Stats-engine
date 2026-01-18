from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='rmse-func',
    topic_id='topic-id',
    topic_slug='predictive-ml',
    slug='rmse',
    title='Root Mean Squared Error',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['rmse'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Root Mean Squared Error - fully functional implementation."""
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    y_true_col = params.get('y_true_column')
    y_pred_col = params.get('y_pred_column')
    
    query = f"SELECT {y_true_col}, {y_pred_col} FROM dataset WHERE {y_true_col} IS NOT NULL AND {y_pred_col} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    y_true = [r[0] for r in data]
    y_pred = [r[1] for r in data]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'rmse': float(rmse),
        'root_mean_squared_error': float(rmse),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
