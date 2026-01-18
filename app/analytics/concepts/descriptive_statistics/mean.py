from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='mean-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='mean',
    title='Mean (Arithmetic Average)',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['mean', 'average', 'arithmetic_mean'],
    tags=['descriptive', 'central_tendency'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate mean with trimmed mean options and confidence interval."""
    import numpy as np
    from scipy import stats
    
    column = params.get('column', params.get('measure_column'))
    trim_percent = params.get('trim_percent', 0)  # 0-49, for trimmed mean
    confidence = params.get('confidence_level', 0.95)
    
    if not column:
        raise ValueError('column parameter required')
    
    query = f"""
        SELECT 
            AVG({column}) as mean,
            COUNT({column}) as n,
            STDDEV_SAMP({column}) as std,
            MIN({column}) as min,
            MAX({column}) as max,
            SUM({column}) as sum
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    mean = float(result[0]) if result[0] is not None else None
    n = int(result[1])
    std = float(result[2]) if result[2] is not None else None
    min_val = float(result[3]) if result[3] is not None else None
    max_val = float(result[4]) if result[4] is not None else None
    total = float(result[5]) if result[5] is not None else None
    
    if mean is None or n < 1:
        return {'error': 'No valid data', 'n': n}
    
    # Standard error and confidence interval
    se = std / np.sqrt(n) if std and n > 1 else None
    
    if se and n > 1:
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = ci_upper = None
    
    result_dict = {
        'mean': mean,
        'arithmetic_mean': mean,
        'average': mean,
        'n': n,
        'sum': total,
        'std': std,
        'se_mean': se,
        'min': min_val,
        'max': max_val,
    }
    
    # Confidence interval
    if ci_lower is not None:
        result_dict['confidence_level'] = confidence
        result_dict['ci_lower'] = ci_lower
        result_dict['ci_upper'] = ci_upper
        result_dict['margin_of_error'] = ci_upper - mean
    
    # Trimmed mean if requested
    if trim_percent > 0 and trim_percent < 50:
        query_data = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
        data = np.array([r[0] for r in ctx.con.execute(query_data).fetchall()])
        
        trim_mean = stats.trim_mean(data, trim_percent / 100)
        result_dict['trimmed_mean'] = float(trim_mean)
        result_dict['trim_percent'] = trim_percent
    
    return result_dict

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
