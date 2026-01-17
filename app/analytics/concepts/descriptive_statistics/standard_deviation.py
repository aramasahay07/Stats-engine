from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='6dbde885-e2e2-4ef9-a45f-99544058bcb7',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='standard-deviation',
    title='Standard Deviation',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['std_dev', 'standard_deviation', 'sd'],
    tags=['spread', 'variation'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the standard deviation of a numeric column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    population = params.get('population', False)
    std_func = 'STDDEV_POP' if population else 'STDDEV_SAMP'
    
    query = f"""
        SELECT 
            {std_func}({column}) as std,
            COUNT({column}) as valid_count
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    return {
        'std': float(result[0]) if result[0] is not None else None,
        'stddev': float(result[0]) if result[0] is not None else None,
        'valid_count': int(result[1]),
        'population': population,
        'measure': column
    }
