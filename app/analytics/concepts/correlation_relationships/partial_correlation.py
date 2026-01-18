from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='be158621-e56c-4bhi-hdi8-bb09e49fe035',
    topic_id='67b7a540-6033-429e-bf49-507aac685ec8',
    topic_slug='correlation-relationships',
    slug='partial-correlation',
    title='Partial Correlation',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['partial_correlation', 'partial_r'],
    tags=['relationship', 'control'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate partial correlation controlling for confounding variables."""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    x_column = params.get('x_column')
    y_column = params.get('y_column')
    control_columns = params.get('control_columns', [])
    
    if not x_column or not y_column:
        raise ValueError('Both x_column and y_column are required')
    
    if not isinstance(control_columns, list):
        control_columns = [control_columns] if control_columns else []
    
    all_cols = [x_column, y_column] + control_columns
    query = f"""
        SELECT {', '.join(all_cols)}
        FROM dataset
        WHERE {' AND '.join([f"{c} IS NOT NULL" for c in all_cols])}
    """
    
    data = np.array(ctx.con.execute(query).fetchall())
    
    if len(data) < len(all_cols) + 2:
        return {'error': 'Insufficient data', 'n': len(data)}
    
    X = data[:, 0]
    Y = data[:, 1]
    
    if len(control_columns) > 0:
        Z = data[:, 2:]
        
        # Regress X on Z, Y on Z
        model_x = LinearRegression().fit(Z, X)
        model_y = LinearRegression().fit(Z, Y)
        
        # Get residuals
        res_x = X - model_x.predict(Z)
        res_y = Y - model_y.predict(Z)
        
        # Correlate residuals
        partial_r = np.corrcoef(res_x, res_y)[0, 1]
    else:
        # No controls = regular correlation
        partial_r = np.corrcoef(X, Y)[0, 1]
    
    abs_r = abs(partial_r)
    strength = 'strong' if abs_r >= 0.7 else 'moderate' if abs_r >= 0.4 else 'weak'
    direction = 'positive' if partial_r > 0 else 'negative' if partial_r < 0 else 'none'
    
    return {
        'partial_correlation': float(partial_r),
        'partial_r': float(partial_r),
        'n': len(data),
        'strength': strength,
        'direction': direction,
        'x_column': x_column,
        'y_column': y_column,
        'control_columns': control_columns,
        'n_controls': len(control_columns),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
