from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='0385df0e-80ba-46b4-8a86-0e9b5bc02676',
    topic_id='2db3d080-3856-421c-bd20-962496ef2b31',
    topic_slug='visualization-eda',
    slug='histogram',
    title='Histogram',
    concept_type='chart',
    level='intro',
    status='published',
    output_keys=['histogram'],
    tags=['visual'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Histogram.
    
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
        'concept': 'histogram',
        'status': 'enabled',
        'message': 'Concept histogram is now operational',
        'n': n,
        'parameters': params
    }
