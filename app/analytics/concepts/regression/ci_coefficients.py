from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ci-coefficients-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='ci-coefficients',
    title='Ci Coefficients',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['ci_coefficients'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    x = params.get('x_column')
    y = params.get('y_column')
    confidence = params.get('confidence_level', 0.95)
    
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
    
    # Standard error of slope
    x_mean = np.mean(X)
    se_slope = np.sqrt(mse / np.sum((X - x_mean)**2))
    
    df = len(data) - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    
    ci_slope = (model.coef_[0] - t_crit * se_slope, model.coef_[0] + t_crit * se_slope)
    
    return {
        'slope': float(model.coef_[0]),
        'ci_slope_lower': float(ci_slope[0]),
        'ci_slope_upper': float(ci_slope[1]),
        'confidence_level': confidence,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
