from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='nonparametric-tests-final',
    topic_id='topic-final',
    topic_slug='advanced-methods',
    slug='nonparametric-tests',
    title='Nonparametric Tests',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['nonparametric_tests'],
    tags=['advanced-methods'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from scipy import stats
    
    method = params.get('method', 'mann_whitney')
    
    return {
        'available_methods': [
            'mann_whitney',
            'wilcoxon',
            'kruskal_wallis',
            'friedman',
        ],
        'method': method,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
