from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='histogram-func',
    topic_id='topic-id',
    topic_slug='visualization-eda',
    slug='histogram',
    title='Histogram',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['histogram'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Histogram - fully functional implementation."""
    import numpy as np
    
    column = params.get('column')
    bins = params.get('bins', 30)
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    counts, bin_edges = np.histogram(data, bins=bins)
    
    return {
        'counts': counts.tolist(),
        'bin_edges': bin_edges.tolist(),
        'bins': bins,
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
