from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ridge-func',
    topic_id='topic-id',
    topic_slug='regression',
    slug='ridge',
    title='Ridge Regression',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['ridge'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Ridge Regression - fully functional implementation."""
    import numpy as np
    from sklearn.linear_model import Ridge
    
    x_cols = params.get('x_columns', [])
    y_col = params.get('y_column')
    alpha = params.get('alpha', 1.0)
    
    if not isinstance(x_cols, list):
        x_cols = [x_cols]
    
    cols = x_cols + [y_col]
    query = f"SELECT {', '.join(cols)} FROM dataset WHERE {' AND '.join([f'{c} IS NOT NULL' for c in cols])}"
    data = np.array(ctx.con.execute(query).fetchall())
    
    X, y = data[:, :-1], data[:, -1]
    model = Ridge(alpha=alpha).fit(X, y)
    
    return {
        'coefficients': {x_cols[i]: float(model.coef_[i]) for i in range(len(x_cols))},
        'intercept': float(model.intercept_),
        'alpha': alpha,
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
