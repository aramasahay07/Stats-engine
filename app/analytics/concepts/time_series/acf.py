from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='acf-final',
    topic_id='topic-final',
    topic_slug='time-series',
    slug='acf',
    title='Acf',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['acf'],
    tags=['time-series'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from statsmodels.tsa.stattools import acf
    import numpy as np
    
    column = params.get('column')
    nlags = params.get('nlags', 40)
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL ORDER BY rowid"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    acf_vals = acf(data, nlags=nlags)
    
    return {
        'acf': acf_vals.tolist(),
        'nlags': nlags,
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
