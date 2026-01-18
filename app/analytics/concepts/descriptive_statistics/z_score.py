from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='zscore-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='z-score',
    title='Z-Score (Standardized Score)',
    concept_type='transformation',
    level='intro',
    status='published',
    output_keys=['z_score', 'standardized_score'],
    tags=['descriptive', 'standardization', 'normalization'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate z-scores with statistics."""
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    value = params.get('value')  # Specific value to calculate z-score for
    
    if not column:
        raise ValueError('column parameter required')
    
    query = f"""
        SELECT 
            AVG({column}) as mean,
            STDDEV_SAMP({column}) as std,
            MIN({column}) as min,
            MAX({column}) as max,
            COUNT({column}) as n
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    mean = float(result[0]) if result[0] is not None else None
    std = float(result[1]) if result[1] is not None else None
    min_val = float(result[2]) if result[2] is not None else None
    max_val = float(result[3]) if result[3] is not None else None
    n = int(result[4])
    
    if mean is None or std is None or std == 0 or n < 2:
        return {'error': 'Insufficient data or zero standard deviation', 'n': n}
    
    result_dict = {
        'mean': mean,
        'std': std,
        'n': n,
        'min_value': min_val,
        'max_value': max_val,
    }
    
    # Calculate z-score for specific value if provided
    if value is not None:
        z = (value - mean) / std
        
        # Percentile (approximate using normal distribution)
        from scipy import stats
        percentile = stats.norm.cdf(z) * 100
        
        result_dict['value'] = float(value)
        result_dict['z_score'] = float(z)
        result_dict['percentile'] = float(percentile)
        result_dict['unusual'] = abs(z) > 2
        result_dict['very_unusual'] = abs(z) > 3
        
        if abs(z) > 3:
            interpretation = 'very unusual (>3 std from mean)'
        elif abs(z) > 2:
            interpretation = 'unusual (>2 std from mean)'
        elif abs(z) > 1:
            interpretation = 'somewhat unusual (>1 std from mean)'
        else:
            interpretation = 'typical (within 1 std of mean)'
        
        result_dict['interpretation'] = interpretation
    else:
        # Calculate z-scores for all values and provide summary
        query_z = f"""
            SELECT 
                ({column} - {mean}) / {std} as z_score
            FROM dataset
            WHERE {column} IS NOT NULL
        """
        
        z_scores = [float(r[0]) for r in ctx.con.execute(query_z).fetchall()]
        z_array = np.array(z_scores)
        
        result_dict['z_scores_calculated'] = len(z_scores)
        result_dict['min_z_score'] = float(np.min(z_array))
        result_dict['max_z_score'] = float(np.max(z_array))
        result_dict['n_unusual'] = int(np.sum(np.abs(z_array) > 2))
        result_dict['n_very_unusual'] = int(np.sum(np.abs(z_array) > 3))
        result_dict['percent_unusual'] = float(np.sum(np.abs(z_array) > 2) / len(z_scores) * 100)
    
    return result_dict

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
