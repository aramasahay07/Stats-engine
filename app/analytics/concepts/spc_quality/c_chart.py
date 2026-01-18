from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='c-chart-001',
    topic_id='spc-topic',
    topic_slug='spc-quality',
    slug='c-chart',
    title='C Control Chart (Count of Defects)',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['c_chart', 'defects_chart'],
    tags=['spc', 'control_chart', 'attributes'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate C control chart for number of defects per unit (constant area)."""
    import numpy as np
    
    defects_column = params.get('defects_column')
    subgroup_column = params.get('subgroup_column')
    
    if not defects_column:
        raise ValueError('defects_column required')
    
    # Get defect counts
    if subgroup_column:
        query = f"SELECT {subgroup_column}, {defects_column} FROM dataset WHERE {defects_column} IS NOT NULL ORDER BY {subgroup_column}"
    else:
        query = f"SELECT {defects_column} FROM dataset WHERE {defects_column} IS NOT NULL ORDER BY rowid"
    
    data = ctx.con.execute(query).fetchall()
    
    if len(data) < 2:
        return {'error': 'Need at least 2 observations'}
    
    if subgroup_column:
        subgroups = [r[0] for r in data]
        defects = np.array([r[1] for r in data])
    else:
        subgroups = list(range(1, len(data) + 1))
        defects = np.array([r[0] for r in data])
    
    # Calculate average defects per unit
    cbar = np.mean(defects)
    
    # Control limits (Poisson distribution)
    sigma_c = np.sqrt(cbar)
    UCL = cbar + 3 * sigma_c
    LCL = max(0, cbar - 3 * sigma_c)
    
    # Detect violations
    violations = []
    for i, c in enumerate(defects):
        if c > UCL or c < LCL:
            violations.append({
                'subgroup': subgroups[i],
                'defects': int(c),
                'limit': 'UCL' if c > UCL else 'LCL'
            })
    
    return {
        'chart_type': 'c',
        'center_line': float(cbar),
        'ucl': float(UCL),
        'lcl': float(LCL),
        'subgroups': [str(s) for s in subgroups],
        'values': defects.tolist(),
        'violations': violations,
        'n_violations': len(violations),
        'in_control': len(violations) == 0,
        'n_subgroups': len(subgroups),
        'total_defects': int(np.sum(defects)),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
