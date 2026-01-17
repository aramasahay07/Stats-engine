from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='fc4fa906-146b-430f-8494-44fa68315411',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='range',
    title='Range',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['range'],
    tags=['spread'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the range (max - min) of a numeric column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"""
        SELECT 
            MIN({column}) as min_val,
            MAX({column}) as max_val,
            MAX({column}) - MIN({column}) as range_val,
            COUNT({column}) as valid_count
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    return {
        'min': float(result[0]) if result[0] is not None else None,
        'max': float(result[1]) if result[1] is not None else None,
        'range': float(result[2]) if result[2] is not None else None,
        'valid_count': int(result[3]),
        'measure': column
    }
