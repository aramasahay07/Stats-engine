from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='skew-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='skewness',
    title='Skewness',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['skewness', 'skew'],
    tags=['descriptive', 'shape', 'distribution'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate skewness with interpretation."""
    from scipy import stats
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    
    if not column:
        raise ValueError('column parameter required')
    
    # DuckDB skewness
    query = f"""
        SELECT 
            SKEWNESS({column}) as skewness,
            COUNT({column}) as n,
            AVG({column}) as mean,
            MEDIAN({column}) as median,
            MODE() WITHIN GROUP (ORDER BY {column}) as mode,
            STDDEV_SAMP({column}) as std
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    skew = float(result[0]) if result[0] is not None else None
    n = int(result[1])
    mean = float(result[2]) if result[2] is not None else None
    median = float(result[3]) if result[3] is not None else None
    mode_val = result[4]
    std = float(result[5]) if result[5] is not None else None
    
    if skew is None or n < 3:
        return {'error': 'Insufficient data for skewness', 'n': n}
    
    # Interpret skewness
    if abs(skew) < 0.5:
        interpretation = 'approximately symmetric'
        symmetry = 'symmetric'
    elif skew > 0:
        if skew < 1.0:
            interpretation = 'moderately right-skewed (positive skew)'
        else:
            interpretation = 'highly right-skewed (positive skew)'
        symmetry = 'right_skewed'
    else:
        if skew > -1.0:
            interpretation = 'moderately left-skewed (negative skew)'
        else:
            interpretation = 'highly left-skewed (negative skew)'
        symmetry = 'left_skewed'
    
    # Standard error of skewness
    se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 2 else None
    
    # Z-score for skewness
    z_skew = skew / se_skew if se_skew and se_skew > 0 else None
    
    return {
        'skewness': skew,
        'skew': skew,
        'interpretation': interpretation,
        'symmetry': symmetry,
        'n': n,
        'mean': mean,
        'median': median,
        'mode': mode_val,
        'std': std,
        'se_skewness': se_skew,
        'z_skewness': z_skew,
        'significant_skew': abs(z_skew) > 1.96 if z_skew else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
