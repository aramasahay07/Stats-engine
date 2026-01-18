from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='polynomial-regression-func',
    topic_id='topic-id',
    topic_slug='regression',
    slug='polynomial-regression',
    title='Polynomial Regression',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['polynomial_regression'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Polynomial Regression - fully functional implementation."""
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    x = params.get('x_column')
    y = params.get('y_column')
    degree = params.get('degree', 2)
    
    query = f"SELECT {x}, {y} FROM dataset WHERE {x} IS NOT NULL AND {y} IS NOT NULL"
    data = np.array(ctx.con.execute(query).fetchall())
    
    X = data[:, 0].reshape(-1, 1)
    y_data = data[:, 1]
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression().fit(X_poly, y_data)
    r2 = model.score(X_poly, y_data)
    
    return {
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'r_squared': float(r2),
        'degree': degree,
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
