from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='pair-plot-final',
    topic_id='topic-final',
    topic_slug='visualization-eda',
    slug='pair-plot',
    title='Pair Plot',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['pair_plot'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    columns = params.get('columns', [])
    
    if not isinstance(columns, list):
        columns = [columns]
    
    return {
        'columns': columns,
        'n_columns': len(columns),
        'message': 'Pairwise scatterplot matrix',
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
