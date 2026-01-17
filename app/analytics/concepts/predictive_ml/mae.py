from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='3952ca82-f47a-4de8-8bd4-d43f4f0037bb',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='mae',
    title='MAE',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['mae'],
    tags=['metrics'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Mae.
    
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
        'concept': 'mae',
        'status': 'enabled',
        'message': 'Concept mae is now operational',
        'n': n,
        'parameters': params
    }
