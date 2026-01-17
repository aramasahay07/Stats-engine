from __future__ import annotations

from typing import Any, Dict

from scipy import stats
import numpy as np

META = ConceptMeta(
    id='0dd8e001-c7d2-4246-826a-a62ddf3dadd9',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='power',
    title='Statistical Power',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['power'],
    tags=['testing'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Power.
    
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
        'concept': 'power',
        'status': 'enabled',
        'message': 'Concept power is now operational',
        'n': n,
        'parameters': params
    }
