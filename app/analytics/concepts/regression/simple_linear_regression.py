from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='f0b92526-cfab-4e5e-82c6-7b4ced628b24',
    topic_id='47670940-6e51-4e25-aa11-9f78987e5194',
    topic_slug='regression',
    slug='simple-linear-regression',
    title='Simple Linear Regression',
    concept_type='model',
    level='intro',
    status='published',
    output_keys=['linear_regression_simple', 'ols'],
    tags=['regression'],
    quality_score=85,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform simple linear regression: Y = β₀ + β₁X + ε"""
    import math
    from scipy import stats
    
    x_column = params.get('x_column', params.get('independent'))
    y_column = params.get('y_column', params.get('dependent'))
    
    if not x_column or not y_column:
        raise ValueError('Both x_column and y_column required')
    
    query = f"""
        SELECT 
            REGR_SLOPE({y_column}, {x_column}) as slope,
            REGR_INTERCEPT({y_column}, {x_column}) as intercept,
            REGR_R2({y_column}, {x_column}) as r_squared,
            REGR_COUNT({y_column}, {x_column}) as n,
            CORR({y_column}, {x_column}) as correlation,
            STDDEV_SAMP({y_column}) as std_y
        FROM dataset
        WHERE {x_column} IS NOT NULL AND {y_column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    slope, intercept, r2, n, corr, std_y = result
    
    if slope is None or n < 3:
        return {'error': 'Insufficient data', 'n': int(n) if n else 0}
    
    # Adjusted R²
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else None
    
    # F-statistic
    if r2 and r2 < 1 and std_y:
        msr = r2 * std_y**2 * (n - 1)
        mse = (1 - r2) * std_y**2 * (n - 1) / (n - 2)
        f_stat = msr / mse if mse > 0 else None
        p_value = 1 - stats.f.cdf(f_stat, 1, n - 2) if f_stat else None
    else:
        f_stat, p_value = None, None
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r2),
        'adj_r_squared': float(adj_r2) if adj_r2 else None,
        'correlation': float(corr) if corr else None,
        'n': int(n),
        'f_statistic': float(f_stat) if f_stat else None,
        'p_value': float(p_value) if p_value else None,
        'equation': f'y = {intercept:.4f} + {slope:.4f}*x',
        'x_variable': x_column,
        'y_variable': y_column,
        'model_significant': p_value < 0.05 if p_value else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
