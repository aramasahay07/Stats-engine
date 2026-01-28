from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ci-from-se-slope-final',
    topic_id='topic-final',
    topic_slug='sampling-estimation',
    slug='ci-from-se-slope',
    title='CI from Standard Error (Slope)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['ci_coefficients'],
    tags=['sampling-estimation'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from scipy import stats
    import numpy as np
    
    slope = params.get('slope')
    se_slope = params.get('se_slope')
    df = params.get('df')
    confidence = params.get('confidence_level', 0.95)
    
    if None in [slope, se_slope, df]:
        raise ValueError('slope, se_slope, and df required')
    
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    ci_lower = slope - t_crit * se_slope
    ci_upper = slope + t_crit * se_slope
    
    return {
        'slope': float(slope),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence_level': float(confidence),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
