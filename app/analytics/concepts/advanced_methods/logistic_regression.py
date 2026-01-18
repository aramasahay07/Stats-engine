from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='logistic-regression-func',
    topic_id='topic-id',
    topic_slug='advanced-methods',
    slug='logistic-regression',
    title='Logistic Regression',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['logistic_regression'],
    tags=['advanced-methods'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Logistic Regression - fully functional implementation."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    
    x_cols = params.get('x_columns', [])
    y_col = params.get('y_column')
    
    if not isinstance(x_cols, list):
        x_cols = [x_cols]
    
    cols = x_cols + [y_col]
    query = f"SELECT {', '.join(cols)} FROM dataset WHERE {' AND '.join([f'{c} IS NOT NULL' for c in cols])}"
    data = np.array(ctx.con.execute(query).fetchall())
    
    X, y = data[:, :-1], data[:, -1]
    model = LogisticRegression().fit(X, y)
    
    return {
        'coefficients': {x_cols[i]: float(model.coef_[0][i]) for i in range(len(x_cols))},
        'intercept': float(model.intercept_[0]),
        'n': len(data),
        'classes': model.classes_.tolist(),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
