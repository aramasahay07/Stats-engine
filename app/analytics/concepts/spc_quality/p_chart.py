from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='p-chart-001',
    topic_id='spc-topic',
    topic_slug='spc-quality',
    slug='p-chart',
    title='P Control Chart (Proportion Defective)',
    concept_type='control_chart',
    level='intermediate',
    status='published',
    output_keys=['p_chart', 'proportion_chart'],
    tags=['spc', 'control_chart', 'attributes'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate P control chart for proportion of defective items."""
    import numpy as np
    
    defectives_column = params.get('defectives_column')
    sample_size_column = params.get('sample_size_column')
    subgroup_column = params.get('subgroup_column')
    constant_n = params.get('sample_size')  # If all samples same size
    
    if not defectives_column:
        raise ValueError('defectives_column required')
    
    # Get data
    if subgroup_column and sample_size_column:
        query = f"""
            SELECT 
                {subgroup_column},
                SUM({defectives_column}) as defectives,
                SUM({sample_size_column}) as n
            FROM dataset
            WHERE {defectives_column} IS NOT NULL
            GROUP BY {subgroup_column}
            ORDER BY {subgroup_column}
        """
    elif constant_n:
        query = f"""
            SELECT 
                {subgroup_column},
                {defectives_column} as defectives
            FROM dataset
            WHERE {defectives_column} IS NOT NULL
            ORDER BY {subgroup_column}
        """
    else:
        raise ValueError('Need sample_size_column or constant sample_size')
    
    data = ctx.con.execute(query).fetchall()
    
    if len(data) < 2:
        return {'error': 'Need at least 2 subgroups'}
    
    subgroups = [r[0] for r in data]
    
    if constant_n:
        defectives = np.array([r[1] for r in data])
        n_values = np.full(len(data), constant_n)
    else:
        defectives = np.array([r[1] for r in data])
        n_values = np.array([r[2] for r in data])
    
    # Calculate proportions
    proportions = defectives / n_values
    
    # Calculate average proportion
    pbar = np.sum(defectives) / np.sum(n_values)
    
    # Control limits (can vary by subgroup if n varies)
    ucl_values = []
    lcl_values = []
    
    for n in n_values:
        sigma_p = np.sqrt(pbar * (1 - pbar) / n)
        ucl = min(1.0, pbar + 3 * sigma_p)
        lcl = max(0.0, pbar - 3 * sigma_p)
        ucl_values.append(ucl)
        lcl_values.append(lcl)
    
    ucl_values = np.array(ucl_values)
    lcl_values = np.array(lcl_values)
    
    # Detect violations
    violations = []
    for i, (p, ucl, lcl) in enumerate(zip(proportions, ucl_values, lcl_values)):
        if p > ucl or p < lcl:
            violations.append({
                'subgroup': subgroups[i],
                'proportion': float(p),
                'defectives': int(defectives[i]),
                'sample_size': int(n_values[i]),
                'ucl': float(ucl),
                'lcl': float(lcl),
                'limit': 'UCL' if p > ucl else 'LCL'
            })
    
    return {
        'chart_type': 'p',
        'center_line': float(pbar),
        'subgroups': [str(s) for s in subgroups],
        'proportions': proportions.tolist(),
        'defectives': defectives.tolist(),
        'sample_sizes': n_values.tolist(),
        'ucl': ucl_values.tolist() if len(set(ucl_values)) > 1 else float(ucl_values[0]),
        'lcl': lcl_values.tolist() if len(set(lcl_values)) > 1 else float(lcl_values[0]),
        'violations': violations,
        'n_violations': len(violations),
        'in_control': len(violations) == 0,
        'n_subgroups': len(subgroups),
        'total_inspected': int(np.sum(n_values)),
        'total_defective': int(np.sum(defectives)),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
