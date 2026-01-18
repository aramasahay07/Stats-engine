from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='odds-ratio-final',
    topic_id='topic-final',
    topic_slug='advanced-methods',
    slug='odds-ratio',
    title='Odds Ratio',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['odds_ratio'],
    tags=['advanced-methods'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    a = params.get('a')  # exposed, diseased
    b = params.get('b')  # exposed, not diseased
    c = params.get('c')  # not exposed, diseased
    d = params.get('d')  # not exposed, not diseased
    
    if None in [a, b, c, d]:
        raise ValueError('All four cell counts (a, b, c, d) required')
    
    odds_ratio = (a * d) / (b * c) if b * c > 0 else None
    
    # Confidence interval
    import numpy as np
    if odds_ratio:
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
    else:
        ci_lower, ci_upper = None, None
    
    return {
        'odds_ratio': float(odds_ratio) if odds_ratio else None,
        'ci_lower': float(ci_lower) if ci_lower else None,
        'ci_upper': float(ci_upper) if ci_upper else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
