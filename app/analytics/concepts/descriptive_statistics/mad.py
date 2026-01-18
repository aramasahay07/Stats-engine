from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='mad-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='mad',
    title='Mean Absolute Deviation (MAD)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['mad', 'mean_absolute_deviation'],
    tags=['descriptive', 'dispersion', 'robust'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Mean Absolute Deviation."""
    import numpy as np
    
    column = params.get('column', params.get('measure_column'))
    center = params.get('center', 'mean')  # 'mean' or 'median'
    
    if not column:
        raise ValueError('column parameter required')
    
    # Get center value
    if center == 'median':
        center_query = f"SELECT MEDIAN({column}) as center FROM dataset WHERE {column} IS NOT NULL"
    else:
        center_query = f"SELECT AVG({column}) as center FROM dataset WHERE {column} IS NOT NULL"
    
    center_val = ctx.con.execute(center_query).fetchone()[0]
    
    if center_val is None:
        return {'error': 'No valid data'}
    
    # Calculate MAD
    mad_query = f"""
        SELECT 
            AVG(ABS({column} - {center_val})) as mad,
            COUNT({column}) as n,
            AVG({column}) as mean,
            MEDIAN({column}) as median,
            STDDEV_SAMP({column}) as std
        FROM dataset
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(mad_query).fetchone()
    
    mad = float(result[0]) if result[0] is not None else None
    n = int(result[1])
    mean = float(result[2]) if result[2] is not None else None
    median = float(result[3]) if result[3] is not None else None
    std = float(result[4]) if result[4] is not None else None
    
    if mad is None or n < 1:
        return {'error': 'No valid data', 'n': n}
    
    # MAD to std ratio (for normally distributed data, MAD ≈ 0.8 * std)
    mad_std_ratio = mad / std if std and std > 0 else None
    
    return {
        'mad': mad,
        'mean_absolute_deviation': mad,
        'center': center,
        'center_value': float(center_val),
        'n': n,
        'mean': mean,
        'median': median,
        'std': std,
        'mad_to_std_ratio': mad_std_ratio,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
