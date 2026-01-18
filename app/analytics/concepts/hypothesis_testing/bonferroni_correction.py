from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='bonferroni-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='bonferroni-correction',
    title='Bonferroni Correction',
    concept_type='adjustment',
    level='intermediate',
    status='published',
    output_keys=['bonferroni_correction', 'adjusted_alpha'],
    tags=['hypothesis_test', 'multiple_testing'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Bonferroni correction for multiple comparisons."""
    p_values = params.get('p_values', [])
    alpha = params.get('alpha', 0.05)
    
    if not p_values:
        raise ValueError('p_values list required')
    
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    
    # Check which p-values are significant after correction
    significant = [p < adjusted_alpha for p in p_values]
    n_significant = sum(significant)
    
    return {
        'adjusted_alpha': float(adjusted_alpha),
        'original_alpha': float(alpha),
        'n_tests': n_tests,
        'n_significant': n_significant,
        'significant_tests': significant,
        'p_values': [float(p) for p in p_values],
        'correction_factor': n_tests,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
