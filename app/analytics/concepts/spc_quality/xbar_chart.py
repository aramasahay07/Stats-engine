from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='xbar-001',
    topic_id='spc-quality-topic',
    topic_slug='spc-quality',
    slug='xbar-chart',
    title='X-bar Control Chart',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['xbar_chart', 'xbar_control_chart'],
    tags=['spc', 'control_chart', 'variables', 'quality'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """X-bar control chart for process mean monitoring."""
    import numpy as np
    from scipy import stats
    
    measurement_column = params.get('measurement_column', params.get('column'))
    subgroup_column = params.get('subgroup_column')
    subgroup_size = params.get('subgroup_size')
    sigma_method = params.get('sigma', 'rbar')  # 'rbar', 'sbar', or known value
    
    if not measurement_column:
        raise ValueError('measurement_column required')
    
    # Get subgroup statistics
    if subgroup_column:
        query = f"""
            SELECT 
                {subgroup_column} as subgroup,
                AVG({measurement_column}) as xbar,
                COUNT({measurement_column}) as n,
                STDDEV_SAMP({measurement_column}) as s,
                MAX({measurement_column}) - MIN({measurement_column}) as range
            FROM dataset
            WHERE {measurement_column} IS NOT NULL
            GROUP BY {subgroup_column}
            ORDER BY {subgroup_column}
        """
    else:
        # Use sequential subgroups
        if not subgroup_size:
            raise ValueError('Either subgroup_column or subgroup_size required')
        
        query = f"""
            SELECT 
                FLOOR((ROW_NUMBER() OVER (ORDER BY rowid) - 1) / {subgroup_size}) as subgroup,
                AVG({measurement_column}) as xbar,
                COUNT({measurement_column}) as n,
                STDDEV_SAMP({measurement_column}) as s,
                MAX({measurement_column}) - MIN({measurement_column}) as range
            FROM dataset
            WHERE {measurement_column} IS NOT NULL
            GROUP BY subgroup
            ORDER BY subgroup
        """
    
    results = ctx.con.execute(query).fetchall()
    
    if len(results) < 2:
        return {'error': 'Need at least 2 subgroups', 'n_subgroups': len(results)}
    
    subgroups = [r[0] for r in results]
    xbars = np.array([float(r[1]) for r in results])
    n_values = [int(r[2]) for r in results]
    s_values = np.array([float(r[3]) if r[3] else 0 for r in results])
    ranges = np.array([float(r[4]) for r in results])
    
    # Average subgroup size
    n = int(np.mean(n_values))
    
    # Calculate center line (grand mean)
    xbar_bar = np.mean(xbars)
    
    # Calculate control limits based on sigma method
    # Constants for control charts (from statistical tables)
    A2_values = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 
                 8: 0.373, 9: 0.337, 10: 0.308, 15: 0.223, 20: 0.180, 25: 0.153}
    A3_values = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182,
                 8: 1.099, 9: 1.032, 10: 0.975, 15: 0.789, 20: 0.680, 25: 0.606}
    
    if sigma_method == 'rbar':
        # Use average range
        rbar = np.mean(ranges)
        A2 = A2_values.get(n, 3 / np.sqrt(n))  # Approximation if n not in table
        ucl = xbar_bar + A2 * rbar
        lcl = xbar_bar - A2 * rbar
        sigma_estimate = rbar / (1.128 if n == 2 else 1.693 if n == 3 else 2.059 if n == 4 else 2.326 if n == 5 else 2.534)
    elif sigma_method == 'sbar':
        # Use average standard deviation
        sbar = np.mean(s_values)
        A3 = A3_values.get(n, 3 / (np.sqrt(n) * 0.9213))  # c4 constant
        ucl = xbar_bar + A3 * sbar
        lcl = xbar_bar - A3 * sbar
        c4 = 0.7979 if n == 2 else 0.8862 if n == 3 else 0.9213 if n == 4 else 0.9400 if n == 5 else 0.9515
        sigma_estimate = sbar / c4
    else:
        # Known sigma
        sigma = float(sigma_method)
        ucl = xbar_bar + 3 * sigma / np.sqrt(n)
        lcl = xbar_bar - 3 * sigma / np.sqrt(n)
        sigma_estimate = sigma
    
    # Detect out-of-control points
    out_of_control = []
    for i, xbar in enumerate(xbars):
        if xbar > ucl or xbar < lcl:
            out_of_control.append({
                'subgroup': subgroups[i],
                'xbar': float(xbar),
                'ucl': float(ucl),
                'lcl': float(lcl),
                'violation': 'above_ucl' if xbar > ucl else 'below_lcl'
            })
    
    # Western Electric Rules (Zone tests)
    zone_violations = detect_zone_violations(xbars, xbar_bar, ucl, lcl)
    
    return {
        'chart_type': 'xbar',
        'center_line': float(xbar_bar),
        'ucl': float(ucl),
        'lcl': float(lcl),
        'n_subgroups': len(subgroups),
        'subgroup_size': n,
        'sigma_estimate': float(sigma_estimate),
        'sigma_method': sigma_method,
        'subgroup_means': xbars.tolist(),
        'subgroups': [str(s) for s in subgroups],
        'n_out_of_control': len(out_of_control),
        'out_of_control_points': out_of_control,
        'in_control': len(out_of_control) == 0,
        'zone_violations': zone_violations,
    }

def detect_zone_violations(data, center, ucl, lcl):
    """Detect Western Electric zone rule violations."""
    violations = []
    
    # Zone boundaries (1, 2, 3 sigma)
    sigma = (ucl - center) / 3
    zone_a_upper = center + 2 * sigma
    zone_a_lower = center - 2 * sigma
    zone_b_upper = center + sigma
    zone_b_lower = center - sigma
    
    # Rule 2: 2 out of 3 consecutive points in zone A or beyond
    for i in range(len(data) - 2):
        points_3 = data[i:i+3]
        in_zone_a = sum((p > zone_a_upper or p < zone_a_lower) for p in points_3)
        if in_zone_a >= 2:
            violations.append({'rule': 2, 'position': i+2, 'description': '2 of 3 points in Zone A'})
    
    # Rule 3: 4 out of 5 consecutive points in zone B or beyond
    for i in range(len(data) - 4):
        points_5 = data[i:i+5]
        in_zone_b = sum((p > zone_b_upper or p < zone_b_lower) for p in points_5)
        if in_zone_b >= 4:
            violations.append({'rule': 3, 'position': i+4, 'description': '4 of 5 points in Zone B'})
    
    # Rule 4: 8 consecutive points on one side of center
    for i in range(len(data) - 7):
        points_8 = data[i:i+8]
        if all(p > center for p in points_8) or all(p < center for p in points_8):
            violations.append({'rule': 4, 'position': i+7, 'description': '8 consecutive on one side'})
    
    return violations

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
