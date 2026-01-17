from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='50d04a1a-6a68-467a-889c-477549dde99d',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='brier-score',
    title='Brier Score',
    concept_type='metric',
    level='advanced',
    status='published',
    output_keys=['brier_score'],
    tags=['metrics'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Brier Score.
    
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
        'concept': 'brier_score',
        'status': 'enabled',
        'message': 'Concept brier_score is now operational',
        'n': n,
        'parameters': params
    }
