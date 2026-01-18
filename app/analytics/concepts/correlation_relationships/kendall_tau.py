from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='9c936409-c34a-29gf-fbg6-9987c27dc813',
    topic_id='67b7a540-6033-429e-bf49-507aac685ec8',
    topic_slug='correlation-relationships',
    slug='kendall-tau',
    title="Kendall's Tau",
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['kendall_tau', 'tau'],
    tags=['relationship', 'nonparametric'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Kendall's Tau correlation coefficient."""
    from scipy import stats
    import numpy as np
    
    x_column = params.get('x_column', params.get('column1'))
    y_column = params.get('y_column', params.get('column2'))
    
    if not x_column or not y_column:
        raise ValueError('Both x_column and y_column are required')
    
    query = f"""
        SELECT {x_column}, {y_column}
        FROM dataset
        WHERE {x_column} IS NOT NULL AND {y_column} IS NOT NULL
    """
    
    data = ctx.con.execute(query).fetchall()
    
    if len(data) < 2:
        return {'error': 'Need at least 2 data points', 'n': len(data)}
    
    x_vals = np.array([row[0] for row in data])
    y_vals = np.array([row[1] for row in data])
    
    tau, p_value = stats.kendalltau(x_vals, y_vals)
    
    abs_tau = abs(tau)
    strength = 'strong' if abs_tau >= 0.7 else 'moderate' if abs_tau >= 0.4 else 'weak'
    direction = 'positive' if tau > 0 else 'negative' if tau < 0 else 'none'
    
    return {
        'kendall_tau': float(tau),
        'tau': float(tau),
        'p_value': float(p_value),
        'n': len(data),
        'significant': p_value < 0.05,
        'strength': strength,
        'direction': direction,
        'x_column': x_column,
        'y_column': y_column,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
