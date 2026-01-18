from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='line-chart-final',
    topic_id='topic-final',
    topic_slug='visualization-eda',
    slug='line-chart',
    title='Line Chart',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['line_chart'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    x_column = params.get('x_column')
    y_column = params.get('y_column')
    limit = params.get('limit', 1000)
    
    query = f"SELECT {x_column}, {y_column} FROM dataset WHERE {x_column} IS NOT NULL AND {y_column} IS NOT NULL ORDER BY {x_column} LIMIT {limit}"
    data = ctx.con.execute(query).fetchall()
    
    return {
        'x_values': [r[0] for r in data],
        'y_values': [r[1] for r in data],
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
