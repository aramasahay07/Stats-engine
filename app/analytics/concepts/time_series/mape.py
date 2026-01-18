from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='mape-final',
    topic_id='topic-final',
    topic_slug='time-series',
    slug='mape',
    title='Mape',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['mape'],
    tags=['time-series'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    
    y_true_col = params.get('y_true_column')
    y_pred_col = params.get('y_pred_column')
    
    query = f"SELECT {y_true_col}, {y_pred_col} FROM dataset WHERE {y_true_col} IS NOT NULL AND {y_pred_col} IS NOT NULL AND {y_true_col} != 0"
    data = ctx.con.execute(query).fetchall()
    
    y_true = [r[0] for r in data]
    y_pred = [r[1] for r in data]
    
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    
    return {
        'mape': float(mape),
        'mean_absolute_percentage_error': float(mape),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
