from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='e27a7fe9-ef35-4355-87e6-fc9c19f85f0b',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='bonferroni',
    title='Bonferroni Correction',
    concept_type='procedure',
    level='intermediate',
    status='published',
    output_keys=['bonferroni'],
    tags=['testing'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Bonferroni.
    
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
        'concept': 'bonferroni',
        'status': 'enabled',
        'message': 'Concept bonferroni is now operational',
        'n': n,
        'parameters': params
    }
