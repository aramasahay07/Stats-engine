from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='kurt-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='kurtosis',
    title='Kurtosis',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['kurtosis', 'kurt'],
    tags=['descriptive', 'shape', 'distribution'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate kurtosis with interpretation."""
    from scipy import stats
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    excess = params.get('excess', True)  # Use excess kurtosis (subtract 3)
    
    if not column:
        raise ValueError('column parameter required')
    
    # DuckDB kurtosis (returns excess kurtosis by default)
    query = f"""
        SELECT 
            KURTOSIS({column}) as kurtosis,
            COUNT({column}) as n,
            AVG({column}) as mean,
            STDDEV_SAMP({column}) as std
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    kurt = float(result[0]) if result[0] is not None else None
    n = int(result[1])
    mean = float(result[2]) if result[2] is not None else None
    std = float(result[3]) if result[3] is not None else None
    
    if kurt is None or n < 4:
        return {'error': 'Insufficient data for kurtosis', 'n': n}
    
    # DuckDB returns excess kurtosis, so adjust if needed
    excess_kurt = kurt
    raw_kurt = kurt + 3 if excess else kurt
    
    # Interpret kurtosis
    if abs(excess_kurt) < 0.5:
        interpretation = 'approximately mesokurtic (normal tails)'
        tail_type = 'mesokurtic'
    elif excess_kurt > 0:
        if excess_kurt < 1.0:
            interpretation = 'slightly leptokurtic (heavier tails than normal)'
        else:
            interpretation = 'highly leptokurtic (very heavy tails)'
        tail_type = 'leptokurtic'
    else:
        if excess_kurt > -1.0:
            interpretation = 'slightly platykurtic (lighter tails than normal)'
        else:
            interpretation = 'highly platykurtic (very light tails)'
        tail_type = 'platykurtic'
    
    # Standard error of kurtosis
    se_kurt = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) if n > 3 else None
    
    # Z-score for kurtosis
    z_kurt = excess_kurt / se_kurt if se_kurt and se_kurt > 0 else None
    
    return {
        'kurtosis': excess_kurt if excess else raw_kurt,
        'excess_kurtosis': excess_kurt,
        'raw_kurtosis': raw_kurt,
        'interpretation': interpretation,
        'tail_type': tail_type,
        'n': n,
        'mean': mean,
        'std': std,
        'se_kurtosis': se_kurt,
        'z_kurtosis': z_kurt,
        'significant_kurtosis': abs(z_kurt) > 1.96 if z_kurt else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
