from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='heteroscedasticity-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='heteroscedasticity',
    title='Heteroscedasticity',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['heteroscedasticity'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from scipy import stats
    import numpy as np
    
    x = params.get('x_column')
    y = params.get('y_column')
    
    query = f"SELECT {x}, {y} FROM dataset WHERE {x} IS NOT NULL AND {y} IS NOT NULL"
    data = np.array(ctx.con.execute(query).fetchall())
    
    from sklearn.linear_model import LinearRegression
    X, y_data = data[:, 0].reshape(-1, 1), data[:, 1]
    
    model = LinearRegression().fit(X, y_data)
    residuals = y_data - model.predict(X)
    
    # Breusch-Pagan test (simplified)
    squared_resid = residuals ** 2
    corr, p_value = stats.pearsonr(X.flatten(), squared_resid)
    
    return {
        'correlation_resid_x': float(corr),
        'p_value': float(p_value),
        'heteroscedasticity_detected': p_value < 0.05,
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
