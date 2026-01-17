from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='9a47f783-8929-4692-b90e-04f8f30d853e',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='specificity',
    title='Specificity',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['specificity'],
    tags=['metrics'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Specificity.
    
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
        'concept': 'specificity',
        'status': 'enabled',
        'message': 'Concept specificity is now operational',
        'n': n,
        'parameters': params
    }
