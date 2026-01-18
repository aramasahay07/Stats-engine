from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='b2c3d4e5-f6a7-8b9c-0d1e-2f3a4b5c6d7e',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='one-sample-t-test',
    title='One-Sample T-Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['one_sample_t_test', 't_test'],
    tags=['hypothesis_test', 't_test'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform one-sample t-test."""
    from scipy import stats
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    hypothesized_mean = params.get('hypothesized_mean', params.get('mu', 0))
    alpha = params.get('alpha', 0.05)
    
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = [row[0] for row in ctx.con.execute(query).fetchall()]
    
    if len(data) < 2:
        return {'error': 'Need at least 2 data points', 'n': len(data)}
    
    t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(len(data))
    
    # Confidence interval
    df = len(data) - 1
    t_crit = stats.t.ppf((1 + (1 - alpha)) / 2, df)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    
    # Cohen's d
    cohens_d = (mean - hypothesized_mean) / std
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'reject_null': p_value < alpha,
        'sample_mean': float(mean),
        'hypothesized_mean': float(hypothesized_mean),
        'difference': float(mean - hypothesized_mean),
        'sample_std': float(std),
        'standard_error': float(se),
        'confidence_level': 1 - alpha,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'cohens_d': float(cohens_d),
        'n': len(data),
        'df': int(df),
        'alpha': float(alpha),
        'column': column,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
