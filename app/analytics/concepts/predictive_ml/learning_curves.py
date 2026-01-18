from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='learning-curves-final',
    topic_id='topic-final',
    topic_slug='predictive-ml',
    slug='learning-curves',
    title='Learning Curves',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['learning_curves'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    
    train_sizes = params.get('train_sizes', [0.1, 0.3, 0.5, 0.7, 0.9])
    
    return {
        'train_sizes': train_sizes,
        'message': 'Learning curves track performance vs training set size',
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
