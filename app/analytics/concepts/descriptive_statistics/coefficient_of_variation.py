from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='ef0a896d-3a00-4b3e-b300-03b70b8abe62',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='coefficient-of-variation',
    title='Coefficient of Variation (CV)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the coefficient of variation (CV = std/mean * 100)."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"""
        SELECT 
            STDDEV_SAMP({column}) as std,
            AVG({column}) as mean,
            COUNT({column}) as valid_count
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    std, mean, count = result
    
    if mean == 0 or mean is None:
        return {'cv': None, 'error': 'Mean is zero or null', 'std': std, 'mean': mean}
    
    cv = (std / abs(mean)) * 100
    
    return {
        'cv': float(cv),
        'coefficient_of_variation': float(cv),
        'std': float(std),
        'mean': float(mean),
        'valid_count': int(count),
        'measure': column
    }
