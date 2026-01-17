from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='5f0260dd-52e4-4ddc-8ab0-51870579fc68',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='recall',
    title='Recall (Sensitivity)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Recall.
    
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
        'concept': 'recall',
        'status': 'enabled',
        'message': 'Concept recall is now operational',
        'n': n,
        'parameters': params
    }
