from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='98c75907-7d76-43f2-abfa-dda9c4016c42',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='mad',
    title='Median Absolute Deviation (MAD)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the median absolute deviation."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = ctx.con.execute(query).fetchnumpy()[column]
    
    if len(data) == 0:
        return {'mad': None, 'error': 'No valid data'}
    
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    return {
        'mad': float(mad),
        'median_absolute_deviation': float(mad),
        'median': float(median),
        'valid_count': len(data),
        'measure': column
    }
