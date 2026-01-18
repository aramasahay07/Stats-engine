from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ci-proportion-func',
    topic_id='topic-id',
    topic_slug='sampling-estimation',
    slug='ci-proportion',
    title='Confidence Interval for Proportion',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['ci_proportion'],
    tags=['sampling-estimation'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Confidence Interval for Proportion - fully functional implementation."""
    from scipy import stats
    import numpy as np
    
    successes = params.get('successes')
    n = params.get('n')
    confidence = params.get('confidence_level', 0.95)
    
    if successes is None or n is None:
        raise ValueError('successes and n required')
    
    p = successes / n
    se = np.sqrt(p * (1 - p) / n)
    z = stats.norm.ppf((1 + confidence) / 2)
    
    return {
        'proportion': float(p),
        'ci_lower': float(max(0, p - z * se)),
        'ci_upper': float(min(1, p + z * se)),
        'confidence_level': float(confidence),
        'n': int(n),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
