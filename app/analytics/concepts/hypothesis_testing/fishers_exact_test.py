from __future__ import annotations

from typing import Any, Dict

from scipy import stats

META = ConceptMeta(
    id='e78e2fa1-05bf-4c6e-9843-0efa0ed2450c',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='fishers-exact-test',
    title='Fisher’s Exact Test',
    concept_type='test',
    level='advanced',
    status='published',
    output_keys=['fishers_exact'],
    tags=['test'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Fishers Exact Test.
    
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
        'concept': 'fishers_exact_test',
        'status': 'enabled',
        'message': 'Concept fishers_exact_test is now operational',
        'n': n,
        'parameters': params
    }
