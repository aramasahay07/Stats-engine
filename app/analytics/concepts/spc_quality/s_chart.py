from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='s-chart-001',
    topic_id='spc-quality-topic',
    topic_slug='spc-quality',
    slug='s-chart',
    title='S Chart (Standard Deviation Control Chart)',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['s_chart', 'std_chart'],
    tags=['spc', 'control_chart', 'variability', 'quality'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """S chart for monitoring process variability using standard deviation."""
    import numpy as np
    
    measurement_column = params.get('measurement_column', params.get('column'))
    subgroup_column = params.get('subgroup_column')
    subgroup_size = params.get('subgroup_size')
    
    if not measurement_column:
        raise ValueError('measurement_column required')
    
    # Get subgroup standard deviations
    if subgroup_column:
        query = f"""
            SELECT 
                {subgroup_column} as subgroup,
                STDDEV_SAMP({measurement_column}) as s,
                COUNT({measurement_column}) as n
            FROM dataset
            WHERE {measurement_column} IS NOT NULL
            GROUP BY {subgroup_column}
            ORDER BY {subgroup_column}
        """
    else:
        if not subgroup_size:
            raise ValueError('Either subgroup_column or subgroup_size required')
        
        query = f"""
            SELECT 
                FLOOR((ROW_NUMBER() OVER (ORDER BY rowid) - 1) / {subgroup_size}) as subgroup,
                STDDEV_SAMP({measurement_column}) as s,
                COUNT({measurement_column}) as n
            FROM dataset
            WHERE {measurement_column} IS NOT NULL
            GROUP BY subgroup
            ORDER BY subgroup
        """
    
    results = ctx.con.execute(query).fetchall()
    
    if len(results) < 2:
        return {'error': 'Need at least 2 subgroups', 'n_subgroups': len(results)}
    
    subgroups = [r[0] for r in results]
    s_values = np.array([float(r[1]) if r[1] else 0 for r in results])
    n_values = [int(r[2]) for r in results]
    
    n = int(np.mean(n_values))
    
    # Calculate center line (average standard deviation)
    sbar = np.mean(s_values)
    
    # Control limit constants (c4, B3, B4)
    c4_values = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594, 
                 8: 0.9650, 9: 0.9693, 10: 0.9727, 15: 0.9823, 20: 0.9869, 25: 0.9896}
    B3_values = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.029, 7: 0.113, 8: 0.179, 9: 0.232, 10: 0.276}
    B4_values = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}
    
    c4 = c4_values.get(n, np.sqrt(2/(n-1)) * (1 - 1/(4*(n-1))))
    B3 = B3_values.get(n, max(0, 1 - 3*np.sqrt(1-c4**2)/c4))
    B4 = B4_values.get(n, 1 + 3*np.sqrt(1-c4**2)/c4)
    
    ucl = B4 * sbar
    lcl = B3 * sbar
    
    # Detect out-of-control points
    out_of_control = []
    for i, s in enumerate(s_values):
        if s > ucl or s < lcl:
            out_of_control.append({
                'subgroup': subgroups[i],
                'std': float(s),
                'ucl': float(ucl),
                'lcl': float(lcl),
                'violation': 'above_ucl' if s > ucl else 'below_lcl'
            })
    
    return {
        'chart_type': 's',
        'center_line': float(sbar),
        'ucl': float(ucl),
        'lcl': float(lcl),
        'n_subgroups': len(subgroups),
        'subgroup_size': n,
        'std_values': s_values.tolist(),
        'subgroups': [str(s) for s in subgroups],
        'n_out_of_control': len(out_of_control),
        'out_of_control_points': out_of_control,
        'in_control': len(out_of_control) == 0,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
