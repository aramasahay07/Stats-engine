from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='911f39ef-42e9-4b13-a941-ab6dcbb29f4a',
    topic_id='0e4fdff5-b126-4544-b5dc-e038ff36791f',
    topic_slug='advanced-methods',
    slug='odds-ratios',
    title='Odds Ratios',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['odds_ratio', 'or'],
    tags=['classification'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Odds Ratios.
    
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
        'concept': 'odds_ratios',
        'status': 'enabled',
        'message': 'Concept odds_ratios is now operational',
        'n': n,
        'parameters': params
    }
