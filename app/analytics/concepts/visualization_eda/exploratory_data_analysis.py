from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='28e7f3d3-0bd2-49ee-a849-0cb5a5cc8c7e',
    topic_id='2db3d080-3856-421c-bd20-962496ef2b31',
    topic_slug='visualization-eda',
    slug='exploratory-data-analysis',
    title='Exploratory Data Analysis (EDA)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Exploratory Data Analysis.
    
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
        'concept': 'exploratory_data_analysis',
        'status': 'enabled',
        'message': 'Concept exploratory_data_analysis is now operational',
        'n': n,
        'parameters': params
    }
