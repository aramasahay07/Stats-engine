from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='cusum-001',
    topic_id='spc-topic',
    topic_slug='spc-quality',
    slug='cusum-chart',
    title='CUSUM Control Chart (Cumulative Sum)',
    concept_type='control_chart',
    level='advanced',
    status='published',
    output_keys=['cusum_chart'],
    tags=['spc', 'control_chart', 'variables', 'advanced'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate CUSUM control chart (sensitive to small sustained shifts)."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    time_column = params.get('time_column')
    target = params.get('target')
    sigma = params.get('sigma')
    k = params.get('k', 0.5)  # Reference value (typically 0.5*sigma)
    h = params.get('h', 5)  # Decision interval (typically 4-5*sigma)
    
    if not measure_column:
        raise ValueError('measure_column required')
    
    # Get values
    if time_column:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY {time_column}"
    else:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY rowid"
    
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 3:
        return {'error': 'Need at least 3 observations'}
    
    # Estimate parameters if not provided
    if target is None:
        target = np.mean(values)
    if sigma is None:
        mr = np.abs(np.diff(values))
        sigma = np.mean(mr) / 1.128
    
    # Calculate CUSUM values
    C_plus = [0]  # Upper CUSUM
    C_minus = [0]  # Lower CUSUM
    
    k_value = k * sigma
    h_value = h * sigma
    
    for x in values:
        cp = max(0, C_plus[-1] + (x - target) - k_value)
        cm = max(0, C_minus[-1] + (target - x) - k_value)
        C_plus.append(cp)
        C_minus.append(cm)
    
    C_plus = np.array(C_plus[1:])
    C_minus = np.array(C_minus[1:])
    
    # Detect violations
    violations = []
    for i in range(len(values)):
        if C_plus[i] > h_value:
            violations.append({
                'observation': i+1,
                'type': 'upward_shift',
                'c_plus': float(C_plus[i]),
                'threshold': float(h_value),
                'value': float(values[i])
            })
        if C_minus[i] > h_value:
            violations.append({
                'observation': i+1,
                'type': 'downward_shift',
                'c_minus': float(C_minus[i]),
                'threshold': float(h_value),
                'value': float(values[i])
            })
    
    return {
        'chart_type': 'cusum',
        'target': float(target),
        'sigma': float(sigma),
        'k': float(k),
        'h': float(h),
        'k_value': float(k_value),
        'h_value': float(h_value),
        'c_plus': C_plus.tolist(),
        'c_minus': C_minus.tolist(),
        'original_values': values.tolist(),
        'violations': violations,
        'n_violations': len(violations),
        'in_control': len(violations) == 0,
        'n_observations': len(values),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
