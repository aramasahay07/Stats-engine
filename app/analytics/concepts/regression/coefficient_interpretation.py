from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='coefficient-interpretation-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='coefficient-interpretation',
    title='Coefficient Interpretation',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['coefficient_interpretation'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    x = params.get('x_column')
    y = params.get('y_column')
    
    query = f"SELECT REGR_SLOPE({y}, {x}) as slope, REGR_INTERCEPT({y}, {x}) as intercept FROM dataset WHERE {x} IS NOT NULL AND {y} IS NOT NULL"
    result = ctx.con.execute(query).fetchone()
    
    return {
        'slope': float(result[0]),
        'slope_interpretation': f'For each unit increase in {x}, {y} increases by {result[0]:.4f}',
        'intercept': float(result[1]),
        'intercept_interpretation': f'When {x} is 0, predicted {y} is {result[1]:.4f}',
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
