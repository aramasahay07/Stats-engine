from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='decomposition-final',
    topic_id='topic-final',
    topic_slug='time-series',
    slug='decomposition',
    title='Decomposition',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['decomposition'],
    tags=['time-series'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    import numpy as np
    
    column = params.get('column')
    period = params.get('period', 12)
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL ORDER BY rowid"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    result = seasonal_decompose(data, model='additive', period=period)
    
    return {
        'trend': result.trend.tolist(),
        'seasonal': result.seasonal.tolist(),
        'residual': result.resid.tolist(),
        'period': period,
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
