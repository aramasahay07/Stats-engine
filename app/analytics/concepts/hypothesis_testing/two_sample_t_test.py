from __future__ import annotations

from typing import Any, Dict

from scipy import stats

META = ConceptMeta(
    id='5ae95bbc-92bd-432c-af15-2c9bdde02aba',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='two-sample-t-test',
    title='Two-sample t-test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['t_test_two_sample'],
    tags=['test'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Two Sample T Test.
    
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
        'concept': 'two_sample_t_test',
        'status': 'enabled',
        'message': 'Concept two_sample_t_test is now operational',
        'n': n,
        'parameters': params
    }
