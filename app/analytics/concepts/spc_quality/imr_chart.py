from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='imr-001',
    topic_id='spc-topic',
    topic_slug='spc-quality',
    slug='imr-chart',
    title='I-MR Control Chart (Individuals & Moving Range)',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['imr_chart', 'individuals_chart'],
    tags=['spc', 'control_chart', 'variables', 'individuals'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate I-MR control chart for individual measurements."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    time_column = params.get('time_column')
    moving_range_span = params.get('mr_span', 2)
    
    if not measure_column:
        raise ValueError('measure_column required')
    
    # Get individual values in time order
    if time_column:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY {time_column}"
    else:
        query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL ORDER BY rowid"
    
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 3:
        return {'error': 'Need at least 3 observations'}
    
    # Calculate moving ranges
    moving_ranges = np.abs(np.diff(values))
    
    # Individuals chart
    xbar = np.mean(values)
    mR = np.mean(moving_ranges)
    
    # Constants for moving range of 2
    d2 = 1.128
    D3 = 0
    D4 = 3.267
    
    # Control limits for Individuals
    sigma_est = mR / d2
    UCL_I = xbar + 3 * sigma_est
    LCL_I = xbar - 3 * sigma_est
    
    # Control limits for Moving Range
    UCL_MR = D4 * mR
    LCL_MR = D3 * mR
    
    # Detect violations - Individuals
    i_violations = []
    for i, val in enumerate(values):
        if val > UCL_I or val < LCL_I:
            i_violations.append({
                'observation': i+1,
                'value': float(val),
                'limit': 'UCL' if val > UCL_I else 'LCL'
            })
    
    # Detect violations - Moving Range
    mr_violations = []
    for i, mr in enumerate(moving_ranges):
        if mr > UCL_MR or mr < LCL_MR:
            mr_violations.append({
                'observation': i+2,  # MR between obs i+1 and i+2
                'value': float(mr),
                'limit': 'UCL' if mr > UCL_MR else 'LCL'
            })
    
    return {
        'chart_type': 'imr',
        'individuals_chart': {
            'center_line': float(xbar),
            'ucl': float(UCL_I),
            'lcl': float(LCL_I),
            'values': values.tolist(),
            'violations': i_violations,
            'n_violations': len(i_violations),
            'in_control': len(i_violations) == 0,
        },
        'moving_range_chart': {
            'center_line': float(mR),
            'ucl': float(UCL_MR),
            'lcl': float(LCL_MR),
            'values': moving_ranges.tolist(),
            'violations': mr_violations,
            'n_violations': len(mr_violations),
            'in_control': len(mr_violations) == 0,
        },
        'n_observations': len(values),
        'sigma_estimate': float(sigma_est),
        'process_mean': float(xbar),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
