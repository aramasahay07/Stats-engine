from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ad047510-d45b-3agh-gch7-aa98d38ed924',
    topic_id='67b7a540-6033-429e-bf49-507aac685ec8',
    topic_slug='correlation-relationships',
    slug='covariance',
    title='Covariance',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['covariance', 'cov'],
    tags=['relationship', 'variation'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate covariance between two variables."""
    x_column = params.get('x_column', params.get('column1'))
    y_column = params.get('y_column', params.get('column2'))
    population = params.get('population', False)
    
    if not x_column or not y_column:
        raise ValueError('Both x_column and y_column are required')
    
    cov_func = 'COVAR_POP' if population else 'COVAR_SAMP'
    
    query = f"""
        SELECT 
            {cov_func}({x_column}, {y_column}) as covariance,
            CORR({x_column}, {y_column}) as correlation,
            COUNT(*) as n,
            AVG({x_column}) as mean_x,
            AVG({y_column}) as mean_y,
            STDDEV_SAMP({x_column}) as std_x,
            STDDEV_SAMP({y_column}) as std_y
        FROM dataset
        WHERE {x_column} IS NOT NULL AND {y_column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    cov = float(result[0]) if result[0] is not None else None
    
    direction = 'positive' if cov and cov > 0 else 'negative' if cov and cov < 0 else 'none'
    
    return {
        'covariance': cov,
        'cov': cov,
        'correlation': float(result[1]) if result[1] else None,
        'n': int(result[2]),
        'direction': direction,
        'population': population,
        'x_column': x_column,
        'y_column': y_column,
        'mean_x': float(result[3]) if result[3] else None,
        'mean_y': float(result[4]) if result[4] else None,
        'std_x': float(result[5]) if result[5] else None,
        'std_y': float(result[6]) if result[6] else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
