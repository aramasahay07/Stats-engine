from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='hypothesis-test-coefficients-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='hypothesis-test-coefficients',
    title='Hypothesis Test Coefficients',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['hypothesis_test_coefficients'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    x = params.get('x_column')
    y = params.get('y_column')
    
    from scipy import stats
    import numpy as np
    
    query = f"SELECT {x}, {y} FROM dataset WHERE {x} IS NOT NULL AND {y} IS NOT NULL"
    data = np.array(ctx.con.execute(query).fetchall())
    
    from sklearn.linear_model import LinearRegression
    X, y_data = data[:, 0].reshape(-1, 1), data[:, 1]
    
    model = LinearRegression().fit(X, y_data)
    predictions = model.predict(X)
    residuals = y_data - predictions
    mse = np.mean(residuals**2)
    
    x_mean = np.mean(X)
    se_slope = np.sqrt(mse / np.sum((X - x_mean)**2))
    
    t_stat = model.coef_[0] / se_slope
    df = len(data) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return {
        'slope': float(model.coef_[0]),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'df': df,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
