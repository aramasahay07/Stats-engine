from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='exponential-smoothing-func',
    topic_id='topic-id',
    topic_slug='time-series',
    slug='exponential-smoothing',
    title='Exponential Smoothing',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['exponential_smoothing'],
    tags=['time-series'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Exponential Smoothing - fully functional implementation."""
    import numpy as np
    
    column = params.get('column')
    alpha = params.get('alpha', 0.3)
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL ORDER BY rowid"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[-1])
    
    return {
        'smoothed_values': result,
        'alpha': float(alpha),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
