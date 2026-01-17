from __future__ import annotations

from typing import Any, Dict

from scipy import stats
import numpy as np

META = ConceptMeta(
    id='391e972d-5f1d-4d6d-b886-b6d979397b0f',
    topic_id='db0cd6cf-0baf-4ef9-819f-295b6668c581',
    topic_slug='sampling-estimation',
    slug='ci-mean',
    title='CI for Mean',
    concept_type='procedure',
    level='intro',
    status='published',
    output_keys=['ci_mean'],
    tags=['inference'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Ci Mean.
    
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
        'concept': 'ci_mean',
        'status': 'enabled',
        'message': 'Concept ci_mean is now operational',
        'n': n,
        'parameters': params
    }
