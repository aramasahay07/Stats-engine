from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='adjusted-r-squared-func',
    topic_id='topic-id',
    topic_slug='regression',
    slug='adjusted-r-squared',
    title='Adjusted R-Squared',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['adjusted_r_squared'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Adjusted R-Squared - fully functional implementation."""
    x = params.get('x_column')
    y = params.get('y_column')
    n_predictors = params.get('n_predictors', 1)
    
    query = f"SELECT REGR_R2({y}, {x}) as r2, REGR_COUNT({y}, {x}) as n FROM dataset WHERE {x} IS NOT NULL AND {y} IS NOT NULL"
    result = ctx.con.execute(query).fetchone()
    r2, n = float(result[0] or 0), int(result[1])
    
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1) if n > n_predictors + 1 else None
    
    return {
        'adjusted_r_squared': float(adj_r2) if adj_r2 else None,
        'r_squared': r2,
        'n': n,
        'n_predictors': n_predictors,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
