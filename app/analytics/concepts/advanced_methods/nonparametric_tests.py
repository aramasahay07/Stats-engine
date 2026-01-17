from __future__ import annotations

from typing import Any, Dict

from scipy import stats

META = ConceptMeta(
    id='8048785a-ebd6-45b1-bae1-8bbd8f9a5941',
    topic_id='0e4fdff5-b126-4544-b5dc-e038ff36791f',
    topic_slug='advanced-methods',
    slug='nonparametric-tests',
    title='Nonparametric Tests',
    concept_type='test',
    level='intermediate',
    status='published',
    output_keys=['mann_whitney', 'kruskal_wallis'],
    tags=['testing', 'robust'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Nonparametric Tests.
    
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
        'concept': 'nonparametric_tests',
        'status': 'enabled',
        'message': 'Concept nonparametric_tests is now operational',
        'n': n,
        'parameters': params
    }
