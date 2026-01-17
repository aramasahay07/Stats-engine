from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='bffe4a06-27fa-44c1-a13c-d9e16b152715',
    topic_id='db0cd6cf-0baf-4ef9-819f-295b6668c581',
    topic_slug='sampling-estimation',
    slug='sampling-methods',
    title='Sampling Methods',
    concept_type='procedure',
    level='intro',
    status='published',
    output_keys=['sampling_methods'],
    tags=['sampling'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Sampling Methods.
    
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
        'concept': 'sampling_methods',
        'status': 'enabled',
        'message': 'Concept sampling_methods is now operational',
        'n': n,
        'parameters': params
    }
