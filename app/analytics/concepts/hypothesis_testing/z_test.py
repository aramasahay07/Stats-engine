from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='z-test-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='z-test',
    title='Z-Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['z_test', 'z_statistic'],
    tags=['hypothesis_test'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform one-sample z-test (known population std)."""
    from scipy import stats
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    population_mean = params.get('population_mean', params.get('mu', 0))
    population_std = params.get('population_std', params.get('sigma'))
    alpha = params.get('alpha', 0.05)
    
    if not column:
        raise ValueError('column required')
    if population_std is None:
        raise ValueError('population_std (sigma) required for z-test')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = [row[0] for row in ctx.con.execute(query).fetchall()]
    
    if len(data) < 2:
        return {'error': 'Insufficient data', 'n': len(data)}
    
    sample_mean = np.mean(data)
    n = len(data)
    
    # Z-statistic: z = (x̄ - μ) / (σ/√n)
    z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(n))
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Confidence interval
    z_crit = stats.norm.ppf((1 + (1 - alpha)) / 2)
    margin = z_crit * (population_std / np.sqrt(n))
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
    
    return {
        'z_statistic': float(z_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'sample_mean': float(sample_mean),
        'population_mean': float(population_mean),
        'population_std': float(population_std),
        'n': n,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'alpha': float(alpha),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
