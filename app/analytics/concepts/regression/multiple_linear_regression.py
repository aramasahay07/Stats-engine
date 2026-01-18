from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='multiple-regression-001',
    topic_id='47670940-6e51-4e25-aa11-9f78987e5194',
    topic_slug='regression',
    slug='multiple-linear-regression',
    title='Multiple Linear Regression',
    concept_type='model',
    level='intermediate',
    status='published',
    output_keys=['linear_regression_multiple', 'mlr'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform multiple linear regression."""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    y_column = params.get('y_column', params.get('dependent'))
    x_columns = params.get('x_columns', params.get('predictors', []))
    
    if not y_column or not x_columns:
        raise ValueError('y_column and x_columns required')
    
    if not isinstance(x_columns, list):
        x_columns = [x_columns]
    
    all_cols = x_columns + [y_column]
    query = f"""
        SELECT {', '.join(all_cols)}
        FROM dataset
        WHERE {' AND '.join([f"{c} IS NOT NULL" for c in all_cols])}
    """
    
    data = np.array(ctx.con.execute(query).fetchall())
    
    if len(data) < len(x_columns) + 2:
        return {'error': 'Insufficient data', 'n': len(data)}
    
    X = data[:, :-1]
    y = data[:, -1]
    
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    
    n, k = len(data), len(x_columns)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    
    coeffs = {x_columns[i]: float(model.coef_[i]) for i in range(len(x_columns))}
    
    return {
        'intercept': float(model.intercept_),
        'coefficients': coeffs,
        'r_squared': float(r2),
        'adj_r_squared': float(adj_r2),
        'n': n,
        'n_predictors': k,
        'y_variable': y_column,
        'x_variables': x_columns,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
