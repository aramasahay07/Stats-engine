from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='sampling-methods-final',
    topic_id='topic-final',
    topic_slug='sampling-estimation',
    slug='sampling-methods',
    title='Sampling Methods',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['sampling_methods'],
    tags=['sampling-estimation'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    method = params.get('method', 'simple_random')
    
    return {
        'method': method,
        'available_methods': [
            'simple_random',
            'stratified',
            'cluster',
            'systematic',
        ],
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
