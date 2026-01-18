from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='brier-score-final',
    topic_id='topic-final',
    topic_slug='predictive-ml',
    slug='brier-score',
    title='Brier Score',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['brier_score'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from sklearn.metrics import brier_score_loss
    
    y_true_col = params.get('y_true_column')
    y_pred_proba_col = params.get('y_pred_proba_column')
    
    query = f"SELECT {y_true_col}, {y_pred_proba_col} FROM dataset WHERE {y_true_col} IS NOT NULL AND {y_pred_proba_col} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    y_true = [r[0] for r in data]
    y_pred_proba = [r[1] for r in data]
    
    bs = brier_score_loss(y_true, y_pred_proba)
    
    return {
        'brier_score': float(bs),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
