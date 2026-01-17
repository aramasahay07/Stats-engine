from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='3884e6e8-12fb-4fa0-b18c-65afb8321eff',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='mean',
    title='Mean',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['mean', 'avg'],
    tags=['center'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the mean (average) of a numeric column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"""
        SELECT 
            AVG({column}) as mean,
            COUNT(*) as count,
            COUNT({column}) as valid_count
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    return {
        'mean': float(result[0]) if result[0] is not None else None,
        'avg': float(result[0]) if result[0] is not None else None,
        'count': int(result[1]),
        'valid_count': int(result[2]),
        'measure': column
    }
