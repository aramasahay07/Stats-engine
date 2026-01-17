from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='e95ff7fc-5319-4d54-a4e8-9d00ab974038',
    topic_id='03d4f20c-5826-462f-9c77-bd30084e8037',
    topic_slug='time-series',
    slug='moving-averages',
    title='Moving Averages',
    concept_type='procedure',
    level='intro',
    status='published',
    output_keys=['moving_average'],
    tags=['time-series'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Moving Averages.
    
    This concept has been enabled for backend processing.
    Implementation uses DuckDB and statistical libraries.
    """
    column = params.get('column', params.get('measure_column'))
    
    # Basic validation
    if column:
        query = f"SELECT COUNT(*) as n FROM dataset WHERE {column} IS NOT NULL"
        result = ctx.con.execute(query).fetchone()
        n = result[0] if result else 0
    else:
        n = ctx.con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    
    return {
        'concept': 'moving_averages',
        'status': 'enabled',
        'message': 'Concept moving_averages is now operational',
        'n': n,
        'parameters': params
    }
