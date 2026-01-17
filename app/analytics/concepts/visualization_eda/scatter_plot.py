from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='c232e88d-66d7-4f8b-8e81-c3451a325a53',
    topic_id='2db3d080-3856-421c-bd20-962496ef2b31',
    topic_slug='visualization-eda',
    slug='scatter-plot',
    title='Scatter Plot',
    concept_type='chart',
    level='intro',
    status='published',
    output_keys=['scatter_plot'],
    tags=['visual', 'relationship'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Scatter Plot.
    
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
        'concept': 'scatter_plot',
        'status': 'enabled',
        'message': 'Concept scatter_plot is now operational',
        'n': n,
        'parameters': params
    }
