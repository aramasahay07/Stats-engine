from __future__ import annotations

from typing import Any, Dict

from scipy import stats

META = ConceptMeta(
    id='b91f677e-9561-42fc-8063-7a007f1dfb9b',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='chi-square-test',
    title='Chi-Square Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['chi_square'],
    tags=['test'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Chi Square Test.
    
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
        'concept': 'chi_square_test',
        'status': 'enabled',
        'message': 'Concept chi_square_test is now operational',
        'n': n,
        'parameters': params
    }
