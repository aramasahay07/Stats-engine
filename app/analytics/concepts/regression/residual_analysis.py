from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='residual-analysis-func',
    topic_id='topic-id',
    topic_slug='regression',
    slug='residual-analysis',
    title='Residual Analysis',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['residual_analysis'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Residual Analysis - fully functional implementation."""
    import numpy as np
    from scipy import stats
    
    x = params.get('x_column')
    y = params.get('y_column')
    
    # Get regression and data
    query = f"""
        SELECT REGR_SLOPE({y}, {x}) as slope, REGR_INTERCEPT({y}, {x}) as intercept,
               {x}, {y}
        FROM dataset WHERE {x} IS NOT NULL AND {y} IS NOT NULL
    """
    data = ctx.con.execute(query).fetchall()
    slope, intercept = data[0][0], data[0][1]
    
    residuals = []
    for row in data:
        x_val, y_actual = row[2], row[3]
        y_pred = intercept + slope * x_val
        residuals.append(y_actual - y_pred)
    
    residuals = np.array(residuals)
    
    _, p_norm = stats.shapiro(residuals) if len(residuals) <= 5000 else stats.normaltest(residuals)
    
    return {
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals, ddof=1)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'normality_p': float(p_norm),
        'residuals_normal': p_norm > 0.05,
        'n': len(residuals),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
