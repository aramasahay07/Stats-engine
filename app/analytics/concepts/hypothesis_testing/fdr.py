from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='fdr-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='fdr',
    title='False Discovery Rate (FDR)',
    concept_type='adjustment',
    level='advanced',
    status='published',
    output_keys=['fdr', 'q_values'],
    tags=['hypothesis_test', 'multiple_testing'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction."""
    from statsmodels.stats.multitest import multipletests
    
    p_values = params.get('p_values', [])
    alpha = params.get('alpha', 0.05)
    
    if not p_values:
        raise ValueError('p_values list required')
    
    # Apply FDR correction
    reject, p_adjusted, alphacSidak, alphacBonf = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    n_significant = sum(reject)
    
    return {
        'q_values': [float(p) for p in p_adjusted],
        'reject_null': reject.tolist(),
        'n_tests': len(p_values),
        'n_significant': int(n_significant),
        'alpha': float(alpha),
        'method': 'Benjamini-Hochberg',
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
