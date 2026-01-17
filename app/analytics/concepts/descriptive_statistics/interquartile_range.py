from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='fd323cb3-9370-47be-bd83-6ae22227af10',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='interquartile-range',
    title='Interquartile Range (IQR)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the interquartile range (Q3 - Q1)."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"""
        SELECT 
            QUANTILE({column}, 0.25) as q1,
            QUANTILE({column}, 0.75) as q3,
            QUANTILE({column}, 0.75) - QUANTILE({column}, 0.25) as iqr,
            COUNT({column}) as valid_count
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    return {
        'q1': float(result[0]) if result[0] is not None else None,
        'q3': float(result[1]) if result[1] is not None else None,
        'iqr': float(result[2]) if result[2] is not None else None,
        'valid_count': int(result[3]),
        'measure': column
    }
