from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='f6a7b8c9-d0e1-2f3a-4b5c-6d7e8f9a0b1c',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='p-value',
    title='P-Value',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['p_value'],
    tags=['hypothesis_test', 'inference'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate p-value from test statistic."""
    from scipy import stats
    
    test_stat = params.get('test_statistic')
    df = params.get('df', params.get('degrees_of_freedom'))
    distribution = params.get('distribution', 't')
    tail = params.get('tail', 'two')
    
    if test_stat is None:
        raise ValueError('test_statistic required')
    
    if distribution == 't':
        if tail == 'two':
            p = 2 * (1 - stats.t.cdf(abs(test_stat), df))
        elif tail == 'right':
            p = 1 - stats.t.cdf(test_stat, df)
        else:
            p = stats.t.cdf(test_stat, df)
    elif distribution == 'z':
        if tail == 'two':
            p = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        elif tail == 'right':
            p = 1 - stats.norm.cdf(test_stat)
        else:
            p = stats.norm.cdf(test_stat)
    elif distribution == 'f':
        p = 1 - stats.f.cdf(test_stat, df, params.get('df2', df))
    elif distribution == 'chi2':
        p = 1 - stats.chi2.cdf(test_stat, df)
    else:
        raise ValueError(f'Unknown distribution: {distribution}')
    
    interp = 'highly significant' if p < 0.001 else 'very significant' if p < 0.01 else 'significant' if p < 0.05 else 'not significant'
    
    return {
        'p_value': float(p),
        'test_statistic': float(test_stat),
        'distribution': distribution,
        'df': int(df) if df else None,
        'tail': tail,
        'interpretation': interp,
        'significant_at_05': p < 0.05,
        'significant_at_01': p < 0.01,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
