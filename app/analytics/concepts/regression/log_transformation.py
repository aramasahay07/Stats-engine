from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='log-transformation-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='log-transformation',
    title='Log Transformation',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['log_transformation'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    
    x = params.get('x_column')
    y = params.get('y_column')
    
    query = f"SELECT {x}, {y} FROM dataset WHERE {x} IS NOT NULL AND {y} IS NOT NULL AND {x} > 0 AND {y} > 0"
    data = np.array(ctx.con.execute(query).fetchall())
    
    # Log transform
    log_x = np.log(data[:, 0])
    log_y = np.log(data[:, 1])
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(log_x.reshape(-1, 1), log_y)
    
    return {
        'log_slope': float(model.coef_[0]),
        'log_intercept': float(model.intercept_),
        'elasticity': float(model.coef_[0]),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
