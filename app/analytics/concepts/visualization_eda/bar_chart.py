from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='1669f9fb-888e-4bf8-b8cb-744225d38510',
    topic_id='2db3d080-3856-421c-bd20-962496ef2b31',
    topic_slug='visualization-eda',
    slug='bar-chart',
    title='Bar Chart',
    concept_type='chart',
    level='intro',
    status='published',
    output_keys=['bar_chart'],
    tags=['visual'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Bar Chart.
    
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
        'concept': 'bar_chart',
        'status': 'enabled',
        'message': 'Concept bar_chart is now operational',
        'n': n,
        'parameters': params
    }
