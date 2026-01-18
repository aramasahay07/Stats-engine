from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='std-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='standard-deviation',
    title='Standard Deviation',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['std', 'stdev', 'standard_deviation'],
    tags=['descriptive', 'dispersion', 'variability'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate standard deviation (sample and population)."""
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    population = params.get('population', False)
    
    if not column:
        raise ValueError('column parameter required')
    
    std_func = 'STDDEV_POP' if population else 'STDDEV_SAMP'
    var_func = 'VAR_POP' if population else 'VAR_SAMP'
    
    query = f"""
        SELECT 
            {std_func}({column}) as std,
            {var_func}({column}) as variance,
            AVG({column}) as mean,
            COUNT({column}) as n,
            MIN({column}) as min,
            MAX({column}) as max
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    std = float(result[0]) if result[0] is not None else None
    variance = float(result[1]) if result[1] is not None else None
    mean = float(result[2]) if result[2] is not None else None
    n = int(result[3])
    min_val = float(result[4]) if result[4] is not None else None
    max_val = float(result[5]) if result[5] is not None else None
    
    if std is None or n < (1 if population else 2):
        return {'error': 'Insufficient data', 'n': n}
    
    # Calculate coefficient of variation if mean != 0
    cv = (std / abs(mean) * 100) if mean and mean != 0 else None
    
    # Standard error (if sample)
    se = std / np.sqrt(n) if not population and n > 0 else None
    
    # Range and relative measures
    range_val = max_val - min_val if min_val and max_val else None
    
    result_dict = {
        'std': std,
        'stdev': std,
        'standard_deviation': std,
        'variance': variance,
        'mean': mean,
        'n': n,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'population': population,
        'degrees_of_freedom': n if population else n - 1,
    }
    
    if cv is not None:
        result_dict['coefficient_of_variation'] = cv
        result_dict['cv_percent'] = cv
    
    if se is not None:
        result_dict['standard_error'] = se
        result_dict['se_mean'] = se
    
    return result_dict

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
