from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='permutation-tests-final',
    topic_id='topic-final',
    topic_slug='advanced-methods',
    slug='permutation-tests',
    title='Permutation Tests',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['permutation_tests'],
    tags=['advanced-methods'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    from scipy import stats
    
    group1_data = params.get('group1_data', [])
    group2_data = params.get('group2_data', [])
    n_permutations = params.get('n_permutations', 1000)
    
    if not group1_data or not group2_data:
        return {'error': 'group1_data and group2_data required'}
    
    observed_diff = np.mean(group1_data) - np.mean(group2_data)
    
    all_data = group1_data + group2_data
    n1 = len(group1_data)
    
    perm_diffs = []
    for _ in range(n_permutations):
        permuted = np.random.permutation(all_data)
        perm_diffs.append(np.mean(permuted[:n1]) - np.mean(permuted[n1:]))
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return {
        'observed_difference': float(observed_diff),
        'p_value': float(p_value),
        'n_permutations': n_permutations,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
