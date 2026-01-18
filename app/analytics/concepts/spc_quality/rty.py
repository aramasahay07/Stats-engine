from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='rty-001',
    topic_id='six-sigma-topic',
    topic_slug='spc-quality',
    slug='rty',
    title='RTY (Rolled Throughput Yield)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['rty', 'rolled_throughput_yield'],
    tags=['spc', 'six_sigma', 'yield'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate RTY - cumulative yield across multiple process steps."""
    import numpy as np
    
    # Can provide individual yields or defect data per step
    yields = params.get('yields')  # List of yield rates per step
    step_defects = params.get('step_defects')  # List of [defects, units] per step
    
    if yields:
        # Direct yield calculation
        if not isinstance(yields, list):
            yields = [yields]
        
        rty = np.prod(yields)
        
        # Calculate equivalent defects
        dpu_equivalent = -np.log(rty) if rty > 0 else float('inf')
        
        return {
            'rty': float(rty),
            'rolled_throughput_yield': float(rty),
            'n_steps': len(yields),
            'step_yields': [float(y) for y in yields],
            'rty_pct': float(rty * 100),
            'dpu_equivalent': float(dpu_equivalent) if dpu_equivalent != float('inf') else None,
        }
        
    elif step_defects:
        # Calculate from defect data
        step_yields = []
        for defects, units in step_defects:
            fpy = (units - defects) / units if units > 0 else 0  # First pass yield
            step_yields.append(fpy)
        
        rty = np.prod(step_yields)
        
        return {
            'rty': float(rty),
            'rolled_throughput_yield': float(rty),
            'n_steps': len(step_defects),
            'step_yields': [float(y) for y in step_yields],
            'rty_pct': float(rty * 100),
        }
    
    else:
        raise ValueError('Need either yields or step_defects parameter')

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
