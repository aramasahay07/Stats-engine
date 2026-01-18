from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='sample-size-final',
    topic_id='topic-final',
    topic_slug='sampling-estimation',
    slug='sample-size',
    title='Sample Size',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['sample_size'],
    tags=['sampling-estimation'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from scipy import stats
    import numpy as np
    
    effect_size = params.get('effect_size')
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    
    if effect_size is None:
        raise ValueError('effect_size required')
    
    # Sample size calculation for t-test
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = (2 * (z_alpha + z_beta)**2) / (effect_size**2)
    
    return {
        'required_sample_size': int(np.ceil(n)),
        'per_group': int(np.ceil(n/2)),
        'effect_size': float(effect_size),
        'alpha': float(alpha),
        'power': float(power),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
