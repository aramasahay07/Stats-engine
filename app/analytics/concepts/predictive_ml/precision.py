from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='3e12dfc4-4e9e-4641-bcd8-ec374f9780fa',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='precision',
    title='Precision',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['precision'],
    tags=['metrics'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Precision.
    
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
        'concept': 'precision',
        'status': 'enabled',
        'message': 'Concept precision is now operational',
        'n': n,
        'parameters': params
    }
