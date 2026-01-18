from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='roc-curve-func',
    topic_id='topic-id',
    topic_slug='predictive-ml',
    slug='roc-curve',
    title='ROC Curve',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['roc_curve'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """ROC Curve - fully functional implementation."""
    from sklearn.metrics import roc_curve, roc_auc_score
    
    y_true_col = params.get('y_true_column')
    y_score_col = params.get('y_score_column')
    
    query = f"SELECT {y_true_col}, {y_score_col} FROM dataset WHERE {y_true_col} IS NOT NULL AND {y_score_col} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    y_true = [r[0] for r in data]
    y_score = [r[1] for r in data]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    return {
        'fpr': fpr.tolist()[:100],
        'tpr': tpr.tolist()[:100],
        'thresholds': thresholds.tolist()[:100],
        'auc': float(auc),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
