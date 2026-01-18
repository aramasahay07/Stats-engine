from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='cf269732-f67d-5cij-iej9-cc10f50gf146',
    topic_id='67b7a540-6033-429e-bf49-507aac685ec8',
    topic_slug='correlation-relationships',
    slug='multicollinearity',
    title='Multicollinearity (VIF)',
    concept_type='diagnostic',
    level='intermediate',
    status='published',
    output_keys=['vif', 'multicollinearity'],
    tags=['regression', 'diagnostic'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Detect multicollinearity using Variance Inflation Factor (VIF)."""
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    columns = params.get('columns', params.get('predictors', []))
    
    if not isinstance(columns, list):
        columns = [columns] if columns else []
    
    if len(columns) < 2:
        raise ValueError('At least 2 columns are required')
    
    query = f"""
        SELECT {', '.join(columns)}
        FROM dataset
        WHERE {' AND '.join([f"{c} IS NOT NULL" for c in columns])}
    """
    
    data = np.array(ctx.con.execute(query).fetchall())
    
    if len(data) < len(columns) + 1:
        return {'error': 'Insufficient data', 'n': len(data)}
    
    vif_scores = {}
    
    for i, col in enumerate(columns):
        # Use other columns as predictors
        X = np.delete(data, i, axis=1)
        y = data[:, i]
        
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        
        # VIF = 1 / (1 - R²)
        vif = 1 / (1 - r_squared) if r_squared < 0.9999 else 999.0
        
        severity = 'severe' if vif > 10 else 'moderate' if vif > 5 else 'low'
        
        vif_scores[col] = {
            'vif': float(vif),
            'r_squared': float(r_squared),
            'severity': severity
        }
    
    max_vif = max(v['vif'] for v in vif_scores.values())
    
    return {
        'vif_scores': vif_scores,
        'max_vif': float(max_vif),
        'n': len(data),
        'multicollinearity_detected': max_vif > 5,
        'overall_severity': 'severe' if max_vif > 10 else 'moderate' if max_vif > 5 else 'low',
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
