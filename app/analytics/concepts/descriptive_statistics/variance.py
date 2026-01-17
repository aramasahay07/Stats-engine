from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='29f6bdc6-58d8-4b66-85cc-632725ce40a7',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='variance',
    title='Variance',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['variance', 'var'],
    tags=['spread'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the variance of a numeric column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    population = params.get('population', False)
    var_func = 'VAR_POP' if population else 'VAR_SAMP'
    
    query = f"""
        SELECT 
            {var_func}({column}) as variance,
            COUNT({column}) as valid_count
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    return {
        'variance': float(result[0]) if result[0] is not None else None,
        'var': float(result[0]) if result[0] is not None else None,
        'valid_count': int(result[1]),
        'population': population,
        'measure': column
    }
