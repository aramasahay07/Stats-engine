from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='paired-t-test',
    title='Paired T-Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['paired_t_test', 't_test_paired'],
    tags=['hypothesis_test', 't_test'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform paired t-test for matched samples."""
    from scipy import stats
    import numpy as np
    
    column1 = params.get('column1', params.get('before'))
    column2 = params.get('column2', params.get('after'))
    alpha = params.get('alpha', 0.05)
    
    if not column1 or not column2:
        raise ValueError('Both column1 and column2 are required')
    
    query = f"SELECT {column1}, {column2} FROM dataset WHERE {column1} IS NOT NULL AND {column2} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    if len(data) < 2:
        return {'error': 'Need at least 2 pairs', 'n': len(data)}
    
    values1 = np.array([row[0] for row in data])
    values2 = np.array([row[1] for row in data])
    
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    differences = values2 - values1
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(len(differences))
    
    # Cohen's d
    cohens_d = mean_diff / std_diff if std_diff > 0 else None
    
    # CI
    df = len(differences) - 1
    t_crit = stats.t.ppf((1 + (1 - alpha)) / 2, df)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'cohens_d': float(cohens_d) if cohens_d else None,
        'n_pairs': len(data),
        'df': int(df),
        'alpha': float(alpha),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
