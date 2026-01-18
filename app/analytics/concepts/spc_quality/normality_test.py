from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='normality-001',
    topic_id='quality-tools-topic',
    topic_slug='spc-quality',
    slug='normality-test',
    title='Normality Test (Anderson-Darling)',
    concept_type='diagnostic',
    level='intermediate',
    status='published',
    output_keys=['normality_test', 'anderson_darling'],
    tags=['spc', 'diagnostic', 'assumptions'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Test for normality using Anderson-Darling test."""
    from scipy import stats
    import numpy as np
    
    measure_column = params.get('measure_column')
    alpha = params.get('alpha', 0.05)
    
    if not measure_column:
        raise ValueError('measure_column required')
    
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 3:
        return {'error': 'Need at least 3 observations'}
    
    # Anderson-Darling test
    ad_result = stats.anderson(values, dist='norm')
    ad_statistic = ad_result.statistic
    
    # Critical values at different significance levels
    critical_values = dict(zip([0.15, 0.10, 0.05, 0.025, 0.01], ad_result.critical_values))
    critical_value_05 = critical_values.get(0.05)
    
    # Shapiro-Wilk test (for comparison)
    if len(values) <= 5000:
        sw_statistic, sw_p_value = stats.shapiro(values)
    else:
        sw_statistic, sw_p_value = None, None
    
    # Ryan-Joiner (similar to Shapiro-Wilk correlation test)
    from scipy.stats import pearsonr
    theoretical_quantiles = stats.norm.ppf((np.arange(1, len(values) + 1) - 0.5) / len(values))
    rj_correlation, _ = pearsonr(np.sort(values), theoretical_quantiles)
    
    # Skewness and kurtosis
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)
    
    # Decision
    normal_ad = ad_statistic < critical_value_05 if critical_value_05 else None
    normal_sw = sw_p_value > alpha if sw_p_value else None
    
    return {
        'anderson_darling_statistic': float(ad_statistic),
        'anderson_darling_critical_values': critical_values,
        'anderson_darling_normal': normal_ad,
        'shapiro_wilk_statistic': float(sw_statistic) if sw_statistic else None,
        'shapiro_wilk_p_value': float(sw_p_value) if sw_p_value else None,
        'shapiro_wilk_normal': normal_sw,
        'ryan_joiner_correlation': float(rj_correlation),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'n': len(values),
        'conclusion': 'Data appears normally distributed' if normal_ad and (normal_sw is None or normal_sw) else 'Data may not be normally distributed',
        'alpha': alpha,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
