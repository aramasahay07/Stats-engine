from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='7b825298-b239-18fe-eaf5-8876b16cb702',
    topic_id='67b7a540-6033-429e-bf49-507aac685ec8',
    topic_slug='correlation-relationships',
    slug='spearman-correlation',
    title='Spearman Correlation',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['spearman_rho', 'spearman_correlation'],
    tags=['relationship', 'nonparametric'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Spearman rank correlation coefficient."""
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
    
    if len(data) < 3:
        return {'error': 'Need at least 3 data points', 'n': len(data)}
    
    x_vals = np.array([row[0] for row in data])
    y_vals = np.array([row[1] for row in data])
    
    rho, p_value = stats.spearmanr(x_vals, y_vals)
    
    abs_rho = abs(rho)
    strength = 'strong' if abs_rho >= 0.7 else 'moderate' if abs_rho >= 0.4 else 'weak'
    direction = 'positive' if rho > 0 else 'negative' if rho < 0 else 'none'
    
    return {
        'spearman_rho': float(rho),
        'rho': float(rho),
        'rank_correlation': float(rho),
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
