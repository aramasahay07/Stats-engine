from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='interaction-terms-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='interaction-terms',
    title='Interaction Terms',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['interaction_terms'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    x1 = params.get('x1_column')
    x2 = params.get('x2_column')
    y = params.get('y_column')
    
    query = f"SELECT {x1}, {x2}, {y} FROM dataset WHERE {x1} IS NOT NULL AND {x2} IS NOT NULL AND {y} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    import numpy as np
    data = np.array(data)
    
    # Create interaction term
    interaction = data[:, 0] * data[:, 1]
    
    from sklearn.linear_model import LinearRegression
    X = np.column_stack([data[:, 0], data[:, 1], interaction])
    y_data = data[:, 2]
    
    model = LinearRegression().fit(X, y_data)
    
    return {
        'coefficients': {
            x1: float(model.coef_[0]),
            x2: float(model.coef_[1]),
            f'{x1}*{x2}': float(model.coef_[2]),
        },
        'intercept': float(model.intercept_),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
