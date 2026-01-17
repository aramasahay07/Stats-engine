from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='2f31ed69-bbbd-42d8-b10e-ba087b04f592',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='mode',
    title='Mode',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['mode'],
    tags=['center'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the mode (most frequent value) of a column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"""
        WITH freq AS (
            SELECT {column} as val, COUNT(*) as freq
            FROM dataset
            WHERE {column} IS NOT NULL
            GROUP BY {column}
            ORDER BY freq DESC
            LIMIT 1
        )
        SELECT val as mode, freq, 
               (SELECT COUNT(*) FROM dataset WHERE {column} IS NOT NULL) as total
        FROM freq
    """
    
    result = ctx.con.execute(query).fetchone()
    
    if result:
        return {
            'mode': result[0],
            'frequency': int(result[1]),
            'total_count': int(result[2]),
            'measure': column
        }
    return {'mode': None, 'frequency': 0, 'measure': column}
