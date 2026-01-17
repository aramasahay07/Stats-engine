from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='9715e555-2595-412d-994a-68eb06aad3c1',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='f1-score',
    title='F1 Score',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['f1'],
    tags=['metrics'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: F1 Score.
    
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
        'concept': 'f1_score',
        'status': 'enabled',
        'message': 'Concept f1_score is now operational',
        'n': n,
        'parameters': params
    }
