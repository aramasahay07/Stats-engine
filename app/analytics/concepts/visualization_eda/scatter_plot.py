from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='scatter-plot-func',
    topic_id='topic-id',
    topic_slug='visualization-eda',
    slug='scatter-plot',
    title='Scatter Plot',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['scatter_plot'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Scatter Plot - fully functional implementation."""
    column_x = params.get('x_column')
    column_y = params.get('y_column')
    limit = params.get('limit', 1000)
    
    query = f"SELECT {column_x}, {column_y} FROM dataset WHERE {column_x} IS NOT NULL AND {column_y} IS NOT NULL LIMIT {limit}"
    data = ctx.con.execute(query).fetchall()
    
    return {
        'x_values': [r[0] for r in data],
        'y_values': [r[1] for r in data],
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
