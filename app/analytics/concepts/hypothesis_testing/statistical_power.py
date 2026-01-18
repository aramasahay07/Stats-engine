from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='power-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='statistical-power',
    title='Statistical Power',
    concept_type='metric',
    level='advanced',
    status='published',
    output_keys=['statistical_power', 'power'],
    tags=['hypothesis_test', 'power'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistical power for t-test."""
    from statsmodels.stats.power import TTestIndPower
    
    effect_size = params.get('effect_size', params.get('cohens_d'))
    n_per_group = params.get('n_per_group', params.get('n'))
    alpha = params.get('alpha', 0.05)
    
    if effect_size is None:
        raise ValueError('effect_size required')
    if n_per_group is None:
        raise ValueError('n_per_group required')
    
    # Calculate power
    analysis = TTestIndPower()
    power = analysis.solve_power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha, ratio=1.0)
    
    # Interpret
    if power >= 0.8:
        adequacy = 'adequate'
    elif power >= 0.6:
        adequacy = 'moderate'
    else:
        adequacy = 'low'
    
    return {
        'power': float(power),
        'statistical_power': float(power),
        'effect_size': float(effect_size),
        'n_per_group': int(n_per_group),
        'alpha': float(alpha),
        'adequacy': adequacy,
        'recommendation': 'Sufficient power' if power >= 0.8 else f'Increase sample size to achieve 80% power',
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
