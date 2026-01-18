from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='3be37965-13aa-4fa9-bbb0-fd72838a7a6d',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='anova-one-way',
    title='One-Way ANOVA',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['anova_one_way', 'anova'],
    tags=['hypothesis_test', 'anova'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform one-way ANOVA."""
    from scipy import stats
    import numpy as np
    
    measure = params.get('measure_column', params.get('column'))
    group = params.get('group_column', params.get('groupby'))
    alpha = params.get('alpha', 0.05)
    
    if not measure or not group:
        raise ValueError('Both measure_column and group_column required')
    
    groups_query = f"SELECT DISTINCT {group} FROM dataset WHERE {group} IS NOT NULL AND {measure} IS NOT NULL ORDER BY {group}"
    groups = [r[0] for r in ctx.con.execute(groups_query).fetchall()]
    
    if len(groups) < 2:
        return {'error': 'Need at least 2 groups'}
    
    group_data = []
    group_stats = []
    
    for g in groups:
        data = [r[0] for r in ctx.con.execute(f"SELECT {measure} FROM dataset WHERE {group}=? AND {measure} IS NOT NULL", [g]).fetchall()]
        if len(data) > 0:
            group_data.append(data)
            group_stats.append({
                'group': str(g),
                'n': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=1)) if len(data) > 1 else 0,
            })
    
    if len(group_data) < 2:
        return {'error': 'Need at least 2 groups with data'}
    
    f_stat, p_value = stats.f_oneway(*group_data)
    
    # Effect size (eta-squared)
    all_data = [v for g in group_data for v in g]
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
    ss_total = sum((v - grand_mean)**2 for v in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'reject_null': p_value < alpha,
        'eta_squared': float(eta_squared),
        'n_groups': len(group_data),
        'n_total': len(all_data),
        'groups': group_stats,
        'alpha': float(alpha),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
