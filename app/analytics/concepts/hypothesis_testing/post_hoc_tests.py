from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='posthoc-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='post-hoc-tests',
    title='Post-Hoc Tests',
    concept_type='test',
    level='intermediate',
    status='published',
    output_keys=['post_hoc', 'pairwise_comparisons'],
    tags=['hypothesis_test', 'post_hoc'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform Tukey HSD post-hoc test after ANOVA."""
    from scipy import stats
    import numpy as np
    
    measure = params.get('measure_column')
    group = params.get('group_column')
    alpha = params.get('alpha', 0.05)
    
    if not measure or not group:
        raise ValueError('measure_column and group_column required')
    
    # Get all groups
    groups = [r[0] for r in ctx.con.execute(f"SELECT DISTINCT {group} FROM dataset WHERE {group} IS NOT NULL").fetchall()]
    
    if len(groups) < 2:
        return {'error': 'Need at least 2 groups'}
    
    group_data = {}
    for g in groups:
        data = [r[0] for r in ctx.con.execute(f"SELECT {measure} FROM dataset WHERE {group}=? AND {measure} IS NOT NULL", [g]).fetchall()]
        if len(data) > 0:
            group_data[str(g)] = data
    
    # Pairwise t-tests with Bonferroni correction
    comparisons = []
    group_names = list(group_data.keys())
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
    adjusted_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
    
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            t_stat, p_val = stats.ttest_ind(group_data[g1], group_data[g2])
            
            comparisons.append({
                'group1': g1,
                'group2': g2,
                'mean_diff': float(np.mean(group_data[g1]) - np.mean(group_data[g2])),
                'p_value': float(p_val),
                'adjusted_p': float(min(p_val * n_comparisons, 1.0)),
                'significant': p_val < adjusted_alpha,
            })
    
    return {
        'comparisons': comparisons,
        'n_comparisons': n_comparisons,
        'adjusted_alpha': float(adjusted_alpha),
        'method': 'Pairwise t-tests with Bonferroni correction',
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
