from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='cv-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='coefficient-of-variation',
    title='Coefficient of Variation (CV)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['cv', 'coefficient_of_variation'],
    tags=['descriptive', 'dispersion', 'relative_variability'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Coefficient of Variation (relative standard deviation)."""
    column = params.get('column', params.get('measure_column'))
    percent = params.get('as_percent', True)
    
    if not column:
        raise ValueError('column parameter required')
    
    query = f"""
        SELECT 
            AVG({column}) as mean,
            STDDEV_SAMP({column}) as std,
            COUNT({column}) as n,
            MIN({column}) as min,
            MAX({column}) as max
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    mean = float(result[0]) if result[0] is not None else None
    std = float(result[1]) if result[1] is not None else None
    n = int(result[2])
    min_val = float(result[3]) if result[3] is not None else None
    max_val = float(result[4]) if result[4] is not None else None
    
    if mean is None or std is None or n < 2:
        return {'error': 'Insufficient data', 'n': n}
    
    if mean == 0:
        return {
            'error': 'Cannot calculate CV when mean is zero',
            'mean': mean,
            'std': std,
            'n': n,
        }
    
    # Calculate CV
    cv = (std / abs(mean)) * (100 if percent else 1)
    
    # Interpret CV
    if cv < 10:
        variability = 'low'
        interpretation = 'low relative variability'
    elif cv < 20:
        variability = 'moderate'
        interpretation = 'moderate relative variability'
    elif cv < 30:
        variability = 'high'
        interpretation = 'high relative variability'
    else:
        variability = 'very_high'
        interpretation = 'very high relative variability'
    
    return {
        'cv': cv,
        'coefficient_of_variation': cv,
        'cv_percent': cv if percent else cv * 100,
        'mean': mean,
        'std': std,
        'n': n,
        'min': min_val,
        'max': max_val,
        'variability': variability,
        'interpretation': interpretation,
        'as_percent': percent,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
