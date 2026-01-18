from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='log-loss-final',
    topic_id='topic-final',
    topic_slug='predictive-ml',
    slug='log-loss',
    title='Log Loss',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['log_loss'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from sklearn.metrics import log_loss
    import numpy as np
    
    y_true_col = params.get('y_true_column')
    y_pred_proba_col = params.get('y_pred_proba_column')
    
    query = f"SELECT {y_true_col}, {y_pred_proba_col} FROM dataset WHERE {y_true_col} IS NOT NULL AND {y_pred_proba_col} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    y_true = [r[0] for r in data]
    y_pred_proba = [r[1] for r in data]
    
    ll = log_loss(y_true, y_pred_proba)
    
    return {
        'log_loss': float(ll),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
