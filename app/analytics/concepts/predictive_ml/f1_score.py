from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='f1-score-func',
    topic_id='topic-id',
    topic_slug='predictive-ml',
    slug='f1-score',
    title='F1 Score',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['f1_score'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """F1 Score - fully functional implementation."""
    from sklearn.metrics import f1_score
    
    y_true_col = params.get('y_true_column')
    y_pred_col = params.get('y_pred_column')
    
    query = f"SELECT {y_true_col}, {y_pred_col} FROM dataset WHERE {y_true_col} IS NOT NULL AND {y_pred_col} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    y_true = [r[0] for r in data]
    y_pred = [r[1] for r in data]
    
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'f1_score': float(f1),
        'f1': float(f1),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
