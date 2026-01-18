from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='percentile-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='percentiles',
    title='Percentiles',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['percentiles', 'percentile'],
    tags=['descriptive', 'position'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate percentiles (default: 5, 10, 25, 50, 75, 90, 95, 99)."""
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    percentiles = params.get('percentiles', [5, 10, 25, 50, 75, 90, 95, 99])
    
    if not column:
        raise ValueError('column parameter required')
    
    # Ensure percentiles is a list
    if not isinstance(percentiles, list):
        percentiles = [percentiles]
    
    # Build query for all percentiles
    percentile_queries = [
        f"PERCENTILE_CONT({p/100}) WITHIN GROUP (ORDER BY {column}) as p{p}"
        for p in percentiles
    ]
    
    query = f"""
        SELECT 
            {', '.join(percentile_queries)},
            COUNT({column}) as n,
            MIN({column}) as min,
            MAX({column}) as max,
            AVG({column}) as mean,
            STDDEV_SAMP({column}) as std
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    n = int(result[len(percentiles)])
    
    if n < 1:
        return {'error': 'No valid data', 'n': n}
    
    # Extract percentile values
    percentile_values = {}
    for i, p in enumerate(percentiles):
        val = float(result[i]) if result[i] is not None else None
        percentile_values[f'p{p}'] = val
        percentile_values[f'percentile_{p}'] = val
    
    # Additional statistics
    min_val = float(result[len(percentiles) + 1]) if result[len(percentiles) + 1] else None
    max_val = float(result[len(percentiles) + 2]) if result[len(percentiles) + 2] else None
    mean = float(result[len(percentiles) + 3]) if result[len(percentiles) + 3] else None
    std = float(result[len(percentiles) + 4]) if result[len(percentiles) + 4] else None
    
    return {
        **percentile_values,
        'percentiles_requested': percentiles,
        'n': n,
        'min': min_val,
        'max': max_val,
        'mean': mean,
        'std': std,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
