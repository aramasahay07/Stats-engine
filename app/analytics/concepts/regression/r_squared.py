from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='r-squared-001',
    topic_id='47670940-6e51-4e25-aa11-9f78987e5194',
    topic_slug='regression',
    slug='r-squared',
    title='R-Squared',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['r_squared', 'r2'],
    tags=['regression', 'fit'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate R² (coefficient of determination)."""
    x = params.get('x_column', params.get('independent'))
    y = params.get('y_column', params.get('dependent'))
    
    if not x or not y:
        raise ValueError('x_column and y_column required')
    
    query = f"""
        SELECT 
            REGR_R2({y}, {x}) as r2,
            REGR_COUNT({y}, {x}) as n
        FROM dataset
        WHERE {x} IS NOT NULL AND {y} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    r2, n = float(result[0] or 0), int(result[1])
    
    if n < 3:
        return {'error': 'Insufficient data', 'n': n}
    
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
    
    fit = 'excellent' if r2 >= 0.9 else 'good' if r2 >= 0.7 else 'moderate' if r2 >= 0.5 else 'weak' if r2 >= 0.3 else 'poor'
    
    return {
        'r_squared': r2,
        'r2': r2,
        'adj_r_squared': float(adj_r2),
        'variance_explained_pct': r2 * 100,
        'n': n,
        'fit_quality': fit,
        'x_column': x,
        'y_column': y,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
