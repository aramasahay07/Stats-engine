from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='range-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='range',
    title='Range',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['range', 'min_max_range'],
    tags=['descriptive', 'dispersion'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate range with min, max, and midrange."""
    column = params.get('column', params.get('measure_column'))
    
    if not column:
        raise ValueError('column parameter required')
    
    query = f"""
        SELECT 
            MIN({column}) as min,
            MAX({column}) as max,
            MAX({column}) - MIN({column}) as range,
            COUNT({column}) as n,
            AVG({column}) as mean
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    min_val = float(result[0]) if result[0] is not None else None
    max_val = float(result[1]) if result[1] is not None else None
    range_val = float(result[2]) if result[2] is not None else None
    n = int(result[3])
    mean = float(result[4]) if result[4] is not None else None
    
    if min_val is None or max_val is None or n < 1:
        return {'error': 'No valid data', 'n': n}
    
    # Midrange
    midrange = (min_val + max_val) / 2
    
    return {
        'range': range_val,
        'min_max_range': range_val,
        'min': min_val,
        'minimum': min_val,
        'max': max_val,
        'maximum': max_val,
        'midrange': midrange,
        'mean': mean,
        'n': n,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
