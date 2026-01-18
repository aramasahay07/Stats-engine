from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='median-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='median',
    title='Median (50th Percentile)',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['median', 'q2', 'percentile_50'],
    tags=['descriptive', 'central_tendency', 'robust'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate median with quartiles and confidence interval."""
    import numpy as np
    from scipy import stats
    
    column = params.get('column', params.get('measure_column'))
    
    if not column:
        raise ValueError('column parameter required')
    
    query = f"""
        SELECT 
            MEDIAN({column}) as median,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as q1,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as q3,
            COUNT({column}) as n,
            MIN({column}) as min,
            MAX({column}) as max
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    median = float(result[0]) if result[0] is not None else None
    q1 = float(result[1]) if result[1] is not None else None
    q3 = float(result[2]) if result[2] is not None else None
    n = int(result[3])
    min_val = float(result[4]) if result[4] is not None else None
    max_val = float(result[5]) if result[5] is not None else None
    
    if median is None or n < 1:
        return {'error': 'No valid data', 'n': n}
    
    # IQR
    iqr = q3 - q1 if q1 and q3 else None
    
    # Confidence interval for median (approximate)
    # CI ~ median ± 1.57 * IQR / sqrt(n)
    if iqr and n > 2:
        se_median = 1.57 * iqr / np.sqrt(n)
        ci_lower = median - 1.96 * se_median
        ci_upper = median + 1.96 * se_median
    else:
        ci_lower = ci_upper = se_median = None
    
    return {
        'median': median,
        'q2': median,
        'percentile_50': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'n': n,
        'min': min_val,
        'max': max_val,
        'se_median': se_median,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'range': max_val - min_val if min_val and max_val else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
