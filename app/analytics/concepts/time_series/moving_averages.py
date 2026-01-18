from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='moving-averages-func',
    topic_id='topic-id',
    topic_slug='time-series',
    slug='moving-averages',
    title='Moving Averages',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['moving_averages'],
    tags=['time-series'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Moving Averages - fully functional implementation."""
    import numpy as np
    
    column = params.get('column')
    window = params.get('window', 7)
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL ORDER BY rowid"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    data = np.array(data)
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    
    return {
        'moving_average': ma.tolist(),
        'window': window,
        'n_original': len(data),
        'n_ma': len(ma),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
