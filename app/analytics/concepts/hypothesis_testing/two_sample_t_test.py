from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='c3d4e5f6-a7b8-9c0d-1e2f-3a4b5c6d7e8f',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='two-sample-t-test',
    title='Two-Sample T-Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['two_sample_t_test', 't_test_independent'],
    tags=['hypothesis_test', 't_test'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform independent two-sample t-test."""
    from scipy import stats
    import numpy as np
    
    measure_column = params.get('measure_column', params.get('column'))
    group_column = params.get('group_column', params.get('groupby'))
    equal_var = params.get('equal_var', True)
    alpha = params.get('alpha', 0.05)
    
    if not measure_column or not group_column:
        raise ValueError('Both measure_column and group_column are required')
    
    groups_query = f"""
        SELECT DISTINCT {group_column}
        FROM dataset
        WHERE {group_column} IS NOT NULL AND {measure_column} IS NOT NULL
        ORDER BY {group_column}
        LIMIT 2
    """
    
    groups = ctx.con.execute(groups_query).fetchall()
    
    if len(groups) < 2:
        return {'error': 'Need exactly 2 groups', 'groups_found': len(groups)}
    
    g1_value, g2_value = groups[0][0], groups[1][0]
    
    data1 = [r[0] for r in ctx.con.execute(f"SELECT {measure_column} FROM dataset WHERE {group_column}=? AND {measure_column} IS NOT NULL", [g1_value]).fetchall()]
    data2 = [r[0] for r in ctx.con.execute(f"SELECT {measure_column} FROM dataset WHERE {group_column}=? AND {measure_column} IS NOT NULL", [g2_value]).fetchall()]
    
    if len(data1) < 2 or len(data2) < 2:
        return {'error': 'Each group needs >= 2 observations'}
    
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    
    # Pooled std and Cohen's d
    if equal_var:
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std
    else:
        cohens_d = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'reject_null': p_value < alpha,
        'group1': str(g1_value),
        'mean_group1': float(mean1),
        'std_group1': float(std1),
        'n_group1': int(n1),
        'group2': str(g2_value),
        'mean_group2': float(mean2),
        'std_group2': float(std2),
        'n_group2': int(n2),
        'mean_difference': float(mean1 - mean2),
        'cohens_d': float(cohens_d),
        'equal_var': equal_var,
        'alpha': float(alpha),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
