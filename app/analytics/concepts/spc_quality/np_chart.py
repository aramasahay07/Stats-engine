from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='np-chart-001',
    topic_id='spc-topic',
    topic_slug='spc-quality',
    slug='np-chart',
    title='NP Control Chart (Number Defective)',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['np_chart', 'number_defective_chart'],
    tags=['spc', 'control_chart', 'attributes'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate NP control chart for number of defective items (constant n)."""
    import numpy as np
    
    defectives_column = params.get('defectives_column')
    subgroup_column = params.get('subgroup_column')
    n = params.get('sample_size')
    
    if not defectives_column or not n:
        raise ValueError('defectives_column and sample_size required')
    
    # Get defective counts
    if subgroup_column:
        query = f"""
            SELECT {subgroup_column}, {defectives_column}
            FROM dataset
            WHERE {defectives_column} IS NOT NULL
            ORDER BY {subgroup_column}
        """
    else:
        query = f"SELECT {defectives_column} FROM dataset WHERE {defectives_column} IS NOT NULL ORDER BY rowid"
    
    data = ctx.con.execute(query).fetchall()
    
    if len(data) < 2:
        return {'error': 'Need at least 2 observations'}
    
    if subgroup_column:
        subgroups = [r[0] for r in data]
        defectives = np.array([r[1] for r in data])
    else:
        subgroups = list(range(1, len(data) + 1))
        defectives = np.array([r[0] for r in data])
    
    # Calculate average number defective
    npbar = np.mean(defectives)
    pbar = npbar / n
    
    # Control limits
    sigma_np = np.sqrt(n * pbar * (1 - pbar))
    UCL = npbar + 3 * sigma_np
    LCL = max(0, npbar - 3 * sigma_np)
    
    # Detect violations
    violations = []
    for i, np_val in enumerate(defectives):
        if np_val > UCL or np_val < LCL:
            violations.append({
                'subgroup': subgroups[i],
                'number_defective': int(np_val),
                'proportion': float(np_val / n),
                'limit': 'UCL' if np_val > UCL else 'LCL'
            })
    
    return {
        'chart_type': 'np',
        'center_line': float(npbar),
        'ucl': float(UCL),
        'lcl': float(LCL),
        'sample_size': n,
        'average_proportion': float(pbar),
        'subgroups': [str(s) for s in subgroups],
        'values': defectives.tolist(),
        'violations': violations,
        'n_violations': len(violations),
        'in_control': len(violations) == 0,
        'n_subgroups': len(subgroups),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
