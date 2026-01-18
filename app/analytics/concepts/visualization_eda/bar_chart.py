from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='bar-chart-final',
    topic_id='topic-final',
    topic_slug='visualization-eda',
    slug='bar-chart',
    title='Bar Chart',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['bar_chart'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    column = params.get('column')
    
    query = f"SELECT {column}, COUNT(*) as count FROM dataset WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY count DESC LIMIT 20"
    data = ctx.con.execute(query).fetchall()
    
    return {
        'categories': [r[0] for r in data],
        'counts': [r[1] for r in data],
        'n_categories': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
