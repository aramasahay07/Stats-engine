from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='r-chart-001',
    topic_id='spc-quality-topic',
    topic_slug='spc-quality',
    slug='r-chart',
    title='R Chart (Range Control Chart)',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['r_chart', 'range_chart'],
    tags=['spc', 'control_chart', 'variability', 'quality'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """R chart for monitoring process variability using range."""
    import numpy as np
    
    measurement_column = params.get('measurement_column', params.get('column'))
    subgroup_column = params.get('subgroup_column')
    subgroup_size = params.get('subgroup_size')
    
    if not measurement_column:
        raise ValueError('measurement_column required')
    
    # Get subgroup ranges
    if subgroup_column:
        query = f"""
            SELECT 
                {subgroup_column} as subgroup,
                MAX({measurement_column}) - MIN({measurement_column}) as range,
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
                MAX({measurement_column}) - MIN({measurement_column}) as range,
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
    ranges = np.array([float(r[1]) for r in results])
    n_values = [int(r[2]) for r in results]
    
    n = int(np.mean(n_values))
    
    # Calculate center line (average range)
    rbar = np.mean(ranges)
    
    # Control limit constants (from statistical tables)
    D3_values = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
    D4_values = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
    
    D3 = D3_values.get(n, max(0, 1 - 3/(1.128*np.sqrt(n))))
    D4 = D4_values.get(n, 1 + 3/(1.128*np.sqrt(n)))
    
    ucl = D4 * rbar
    lcl = D3 * rbar
    
    # Detect out-of-control points
    out_of_control = []
    for i, r in enumerate(ranges):
        if r > ucl or r < lcl:
            out_of_control.append({
                'subgroup': subgroups[i],
                'range': float(r),
                'ucl': float(ucl),
                'lcl': float(lcl),
                'violation': 'above_ucl' if r > ucl else 'below_lcl'
            })
    
    return {
        'chart_type': 'r',
        'center_line': float(rbar),
        'ucl': float(ucl),
        'lcl': float(lcl),
        'n_subgroups': len(subgroups),
        'subgroup_size': n,
        'ranges': ranges.tolist(),
        'subgroups': [str(s) for s in subgroups],
        'n_out_of_control': len(out_of_control),
        'out_of_control_points': out_of_control,
        'in_control': len(out_of_control) == 0,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
