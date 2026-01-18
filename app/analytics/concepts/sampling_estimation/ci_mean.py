from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ci-mean-func',
    topic_id='topic-id',
    topic_slug='sampling-estimation',
    slug='ci-mean',
    title='Confidence Interval for Mean',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['ci_mean'],
    tags=['sampling-estimation'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Confidence Interval for Mean - fully functional implementation."""
    from scipy import stats
    import numpy as np
    
    column = params.get('column')
    confidence = params.get('confidence_level', 0.95)
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, mean, se)
    
    return {
        'mean': float(mean),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'confidence_level': float(confidence),
        'margin_of_error': float(ci[1] - mean),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
