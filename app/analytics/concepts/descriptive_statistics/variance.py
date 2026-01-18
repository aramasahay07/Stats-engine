from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='var-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='variance',
    title='Variance',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['variance', 'var'],
    tags=['descriptive', 'dispersion', 'variability'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate variance (sample and population)."""
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    population = params.get('population', False)
    
    if not column:
        raise ValueError('column parameter required')
    
    var_func = 'VAR_POP' if population else 'VAR_SAMP'
    std_func = 'STDDEV_POP' if population else 'STDDEV_SAMP'
    
    query = f"""
        SELECT 
            {var_func}({column}) as variance,
            {std_func}({column}) as std,
            AVG({column}) as mean,
            COUNT({column}) as n,
            SUM(POWER({column} - (SELECT AVG({column}) FROM dataset WHERE {column} IS NOT NULL), 2)) as sum_squares
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    variance = float(result[0]) if result[0] is not None else None
    std = float(result[1]) if result[1] is not None else None
    mean = float(result[2]) if result[2] is not None else None
    n = int(result[3])
    sum_squares = float(result[4]) if result[4] is not None else None
    
    if variance is None or n < (1 if population else 2):
        return {'error': 'Insufficient data', 'n': n}
    
    # Degrees of freedom
    df = n if population else n - 1
    
    return {
        'variance': variance,
        'var': variance,
        'standard_deviation': std,
        'std': std,
        'mean': mean,
        'n': n,
        'sum_of_squares': sum_squares,
        'degrees_of_freedom': df,
        'population': population,
        'ms': variance,  # Mean square = variance
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
