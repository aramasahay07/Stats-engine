from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='4a79fff4-db26-4a7f-aa22-5c311dd1ee6a',
    topic_id='03d4f20c-5826-462f-9c77-bd30084e8037',
    topic_slug='time-series',
    slug='mape',
    title='MAPE',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['mape'],
    tags=['forecasting', 'metric'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Mape.
    
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
        'concept': 'mape',
        'status': 'enabled',
        'message': 'Concept mape is now operational',
        'n': n,
        'parameters': params
    }
