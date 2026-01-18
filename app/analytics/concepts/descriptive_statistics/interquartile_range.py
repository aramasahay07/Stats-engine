from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='iqr-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='interquartile-range',
    title='Interquartile Range (IQR)',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['iqr', 'interquartile_range'],
    tags=['descriptive', 'dispersion', 'robust'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate IQR with quartiles and outlier boundaries."""
    column = params.get('column', params.get('measure_column'))
    
    if not column:
        raise ValueError('column parameter required')
    
    query = f"""
        SELECT 
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as q1,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {column}) as q2,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as q3,
            COUNT({column}) as n,
            MIN({column}) as min,
            MAX({column}) as max
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    q1 = float(result[0]) if result[0] is not None else None
    q2 = float(result[1]) if result[1] is not None else None
    q3 = float(result[2]) if result[2] is not None else None
    n = int(result[3])
    min_val = float(result[4]) if result[4] is not None else None
    max_val = float(result[5]) if result[5] is not None else None
    
    if q1 is None or q3 is None or n < 1:
        return {'error': 'No valid data', 'n': n}
    
    # IQR
    iqr = q3 - q1
    
    # Outlier boundaries (Tukey's fences)
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # Extreme outlier boundaries
    lower_extreme = q1 - 3.0 * iqr
    upper_extreme = q3 + 3.0 * iqr
    
    # Count outliers
    outlier_query = f"""
        SELECT 
            SUM(CASE WHEN {column} < {lower_fence} OR {column} > {upper_fence} THEN 1 ELSE 0 END) as n_outliers,
            SUM(CASE WHEN {column} < {lower_extreme} OR {column} > {upper_extreme} THEN 1 ELSE 0 END) as n_extreme
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    outlier_result = ctx.con.execute(outlier_query).fetchone()
    n_outliers = int(outlier_result[0]) if outlier_result[0] else 0
    n_extreme = int(outlier_result[1]) if outlier_result[1] else 0
    
    return {
        'iqr': iqr,
        'interquartile_range': iqr,
        'q1': q1,
        'q2': q2,
        'median': q2,
        'q3': q3,
        'n': n,
        'min': min_val,
        'max': max_val,
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'lower_extreme_fence': lower_extreme,
        'upper_extreme_fence': upper_extreme,
        'n_outliers': n_outliers,
        'n_extreme_outliers': n_extreme,
        'outlier_percent': (n_outliers / n * 100) if n > 0 else 0,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
