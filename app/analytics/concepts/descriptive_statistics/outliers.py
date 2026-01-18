from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='outliers-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='outliers',
    title='Outlier Detection',
    concept_type='diagnostic',
    level='intermediate',
    status='published',
    output_keys=['outliers', 'outlier_detection'],
    tags=['descriptive', 'diagnostics', 'quality'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Detect outliers using multiple methods."""
    import numpy as np
    from scipy import stats
    
    column = params.get('column', params.get('measure_column'))
    method = params.get('method', 'iqr')  # 'iqr', 'z_score', 'modified_z', 'both'
    
    if not column:
        raise ValueError('column parameter required')
    
    # Get basic statistics
    query = f"""
        SELECT 
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as q1,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {column}) as q2,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as q3,
            AVG({column}) as mean,
            STDDEV_SAMP({column}) as std,
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
    mean = float(result[3]) if result[3] is not None else None
    std = float(result[4]) if result[4] is not None else None
    n = int(result[5])
    min_val = float(result[6]) if result[6] is not None else None
    max_val = float(result[7]) if result[7] is not None else None
    
    if n < 3:
        return {'error': 'Insufficient data', 'n': n}
    
    outliers_result = {
        'n': n,
        'mean': mean,
        'std': std,
        'q1': q1,
        'q2': q2,
        'q3': q3,
        'min': min_val,
        'max': max_val,
        'method': method,
    }
    
    # IQR Method
    if method in ['iqr', 'both'] and q1 and q3:
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        lower_extreme = q1 - 3.0 * iqr
        upper_extreme = q3 + 3.0 * iqr
        
        outlier_query_iqr = f"""
            SELECT 
                SUM(CASE WHEN {column} < {lower_fence} OR {column} > {upper_fence} THEN 1 ELSE 0 END) as n_outliers,
                SUM(CASE WHEN {column} < {lower_extreme} OR {column} > {upper_extreme} THEN 1 ELSE 0 END) as n_extreme,
                SUM(CASE WHEN {column} < {lower_fence} THEN 1 ELSE 0 END) as n_low,
                SUM(CASE WHEN {column} > {upper_fence} THEN 1 ELSE 0 END) as n_high
            FROM dataset
            WHERE {column} IS NOT NULL
        """
        
        iqr_result = ctx.con.execute(outlier_query_iqr).fetchone()
        
        outliers_result['iqr_method'] = {
            'iqr': iqr,
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
            'lower_extreme_fence': lower_extreme,
            'upper_extreme_fence': upper_extreme,
            'n_outliers': int(iqr_result[0]) if iqr_result[0] else 0,
            'n_extreme_outliers': int(iqr_result[1]) if iqr_result[1] else 0,
            'n_low_outliers': int(iqr_result[2]) if iqr_result[2] else 0,
            'n_high_outliers': int(iqr_result[3]) if iqr_result[3] else 0,
            'outlier_percent': (int(iqr_result[0]) / n * 100) if iqr_result[0] and n > 0 else 0,
        }
    
    # Z-Score Method
    if method in ['z_score', 'both'] and mean and std and std > 0:
        outlier_query_z = f"""
            SELECT 
                SUM(CASE WHEN ABS(({column} - {mean}) / {std}) > 3 THEN 1 ELSE 0 END) as n_extreme,
                SUM(CASE WHEN ABS(({column} - {mean}) / {std}) > 2 THEN 1 ELSE 0 END) as n_outliers
            FROM dataset
            WHERE {column} IS NOT NULL
        """
        
        z_result = ctx.con.execute(outlier_query_z).fetchone()
        
        outliers_result['z_score_method'] = {
            'mean': mean,
            'std': std,
            'threshold_2sigma': 2,
            'threshold_3sigma': 3,
            'n_beyond_2sigma': int(z_result[1]) if z_result[1] else 0,
            'n_beyond_3sigma': int(z_result[0]) if z_result[0] else 0,
            'outlier_percent_2sigma': (int(z_result[1]) / n * 100) if z_result[1] and n > 0 else 0,
            'outlier_percent_3sigma': (int(z_result[0]) / n * 100) if z_result[0] and n > 0 else 0,
        }
    
    # Overall summary
    if method == 'both':
        iqr_outliers = outliers_result.get('iqr_method', {}).get('n_outliers', 0)
        z_outliers = outliers_result.get('z_score_method', {}).get('n_beyond_2sigma', 0)
        
        outliers_result['summary'] = {
            'iqr_outliers': iqr_outliers,
            'z_score_outliers': z_outliers,
            'agreement': iqr_outliers == z_outliers,
        }
    
    return outliers_result

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
