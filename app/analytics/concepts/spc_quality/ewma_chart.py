from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ewma-001',
    topic_id='spc-topic',
    topic_slug='spc-quality',
    slug='ewma-chart',
    title='EWMA Control Chart (Exponentially Weighted Moving Average)',
    concept_type='control_chart',
    level='advanced',
    status='published',
    output_keys=['ewma_chart'],
    tags=['spc', 'control_chart', 'variables', 'advanced'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate EWMA control chart (sensitive to small shifts)."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    time_column = params.get('time_column')
    lambda_param = params.get('lambda', 0.2)  # Weight parameter (0-1)
    target = params.get('target')  # Target value
    sigma = params.get('sigma')  # Known process sigma
    
    if not measure_column:
        raise ValueError('measure_column required')
    
    # Get values in time order
    if time_column:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY {time_column}"
    else:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY rowid"
    
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 3:
        return {'error': 'Need at least 3 observations'}
    
    # Estimate target and sigma if not provided
    if target is None:
        target = np.mean(values)
    if sigma is None:
        # Estimate from moving ranges
        mr = np.abs(np.diff(values))
        sigma = np.mean(mr) / 1.128
    
    # Calculate EWMA values
    ewma_values = [target]  # Start at target
    for i, x in enumerate(values):
        z = lambda_param * x + (1 - lambda_param) * ewma_values[-1]
        ewma_values.append(z)
    
    ewma_values = np.array(ewma_values[1:])  # Remove initial target
    
    # Control limits (vary with time)
    L = 3  # Control limit width (typically 3)
    ucl_values = []
    lcl_values = []
    
    for i in range(len(values)):
        factor = np.sqrt((lambda_param / (2 - lambda_param)) * (1 - (1 - lambda_param)**(2*(i+1))))
        ucl = target + L * sigma * factor
        lcl = target - L * sigma * factor
        ucl_values.append(ucl)
        lcl_values.append(lcl)
    
    ucl_values = np.array(ucl_values)
    lcl_values = np.array(lcl_values)
    
    # Detect violations
    violations = []
    for i, (ewma, ucl, lcl) in enumerate(zip(ewma_values, ucl_values, lcl_values)):
        if ewma > ucl or ewma < lcl:
            violations.append({
                'observation': i+1,
                'ewma_value': float(ewma),
                'ucl': float(ucl),
                'lcl': float(lcl),
                'original_value': float(values[i])
            })
    
    return {
        'chart_type': 'ewma',
        'target': float(target),
        'lambda': float(lambda_param),
        'sigma': float(sigma),
        'ewma_values': ewma_values.tolist(),
        'original_values': values.tolist(),
        'ucl': ucl_values.tolist(),
        'lcl': lcl_values.tolist(),
        'violations': violations,
        'n_violations': len(violations),
        'in_control': len(violations) == 0,
        'n_observations': len(values),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
