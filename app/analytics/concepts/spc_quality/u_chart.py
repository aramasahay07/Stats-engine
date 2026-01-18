from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='u-chart-001',
    topic_id='spc-topic',
    topic_slug='spc-quality',
    slug='u-chart',
    title='U Control Chart (Defects Per Unit)',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['u_chart', 'defects_per_unit_chart'],
    tags=['spc', 'control_chart', 'attributes'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate U control chart for defects per unit (variable area)."""
    import numpy as np
    
    defects_column = params.get('defects_column')
    area_column = params.get('area_column')  # Number of units inspected
    subgroup_column = params.get('subgroup_column')
    
    if not defects_column or not area_column:
        raise ValueError('defects_column and area_column required')
    
    # Get data
    if subgroup_column:
        query = f"SELECT {subgroup_column}, {defects_column}, {area_column} FROM dataset WHERE {defects_column} IS NOT NULL AND {area_column} IS NOT NULL ORDER BY {subgroup_column}"
    else:
        query = f"SELECT {defects_column}, {area_column} FROM dataset WHERE {defects_column} IS NOT NULL AND {area_column} IS NOT NULL ORDER BY rowid"
    
    data = ctx.con.execute(query).fetchall()
    
    if len(data) < 2:
        return {'error': 'Need at least 2 observations'}
    
    if subgroup_column:
        subgroups = [r[0] for r in data]
        defects = np.array([r[1] for r in data])
        areas = np.array([r[2] for r in data])
    else:
        subgroups = list(range(1, len(data) + 1))
        defects = np.array([r[0] for r in data])
        areas = np.array([r[1] for r in data])
    
    # Calculate defects per unit
    u_values = defects / areas
    
    # Calculate average defects per unit
    ubar = np.sum(defects) / np.sum(areas)
    
    # Control limits (vary by area)
    ucl_values = []
    lcl_values = []
    
    for area in areas:
        sigma_u = np.sqrt(ubar / area)
        ucl = ubar + 3 * sigma_u
        lcl = max(0, ubar - 3 * sigma_u)
        ucl_values.append(ucl)
        lcl_values.append(lcl)
    
    ucl_values = np.array(ucl_values)
    lcl_values = np.array(lcl_values)
    
    # Detect violations
    violations = []
    for i, (u, ucl, lcl) in enumerate(zip(u_values, ucl_values, lcl_values)):
        if u > ucl or u < lcl:
            violations.append({
                'subgroup': subgroups[i],
                'defects_per_unit': float(u),
                'defects': int(defects[i]),
                'area': int(areas[i]),
                'ucl': float(ucl),
                'lcl': float(lcl),
                'limit': 'UCL' if u > ucl else 'LCL'
            })
    
    return {
        'chart_type': 'u',
        'center_line': float(ubar),
        'subgroups': [str(s) for s in subgroups],
        'defects_per_unit': u_values.tolist(),
        'defects': defects.tolist(),
        'areas': areas.tolist(),
        'ucl': ucl_values.tolist() if len(set(ucl_values)) > 1 else float(ucl_values[0]),
        'lcl': lcl_values.tolist() if len(set(lcl_values)) > 1 else float(lcl_values[0]),
        'violations': violations,
        'n_violations': len(violations),
        'in_control': len(violations) == 0,
        'n_subgroups': len(subgroups),
        'total_defects': int(np.sum(defects)),
        'total_area': int(np.sum(areas)),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
