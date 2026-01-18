from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict
import math

META = ConceptMeta(
    id='39eb673c-0838-4b36-8359-cf55b83055aa',
    topic_id='67b7a540-6033-429e-bf49-507aac685ec8',
    topic_slug='correlation-relationships',
    slug='pearson-correlation',
    title='Pearson Correlation',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['pearson_r', 'correlation'],
    tags=['relationship'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Pearson correlation coefficient between two variables."""
    from scipy import stats
    
    x_column = params.get('x_column', params.get('column1'))
    y_column = params.get('y_column', params.get('column2'))
    
    if not x_column or not y_column:
        raise ValueError('Both x_column and y_column are required')
    
    query = f"""
        SELECT 
            CORR({x_column}, {y_column}) as correlation,
            COUNT(*) as n,
            AVG({x_column}) as mean_x,
            AVG({y_column}) as mean_y,
            STDDEV_SAMP({x_column}) as std_x,
            STDDEV_SAMP({y_column}) as std_y
        FROM dataset
        WHERE {x_column} IS NOT NULL 
          AND {y_column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    r = float(result[0]) if result[0] is not None else None
    n = int(result[1])
    
    if r is None or n < 3:
        return {'error': 'Insufficient data for correlation', 'n': n}
    
    # Calculate t-statistic: t = r * sqrt(n-2) / sqrt(1-r^2)
    if abs(r) < 1:
        t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    else:
        t_stat = None
        p_value = None
    
    # Interpret strength
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = 'strong'
    elif abs_r >= 0.4:
        strength = 'moderate'
    elif abs_r >= 0.2:
        strength = 'weak'
    else:
        strength = 'very weak'
    
    direction = 'positive' if r > 0 else 'negative' if r < 0 else 'none'
    
    return {
        'correlation': r,
        'pearson_r': r,
        'r': r,
        'r_squared': r**2,
        'n': n,
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_of_freedom': n - 2,
        'strength': strength,
        'direction': direction,
        'significant': p_value < 0.05 if p_value else None,
        'x_column': x_column,
        'y_column': y_column,
        'mean_x': float(result[2]) if result[2] else None,
        'mean_y': float(result[3]) if result[3] else None,
        'std_x': float(result[4]) if result[4] else None,
        'std_y': float(result[5]) if result[5] else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
