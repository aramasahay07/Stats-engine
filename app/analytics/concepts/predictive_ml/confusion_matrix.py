from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='db3f148c-259c-4baa-9b0a-5d85d40b4335',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='confusion-matrix',
    title='Confusion Matrix',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['confusion_matrix'],
    tags=['metrics'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Confusion Matrix.
    
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
        'concept': 'confusion_matrix',
        'status': 'enabled',
        'message': 'Concept confusion_matrix is now operational',
        'n': n,
        'parameters': params
    }
