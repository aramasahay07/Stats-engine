from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='cp-001',
    topic_id='capability-topic',
    topic_slug='spc-quality',
    slug='cp',
    title='Cp (Process Capability - Potential)',
    concept_type='capability',
    level='intermediate',
    status='published',
    output_keys=['cp', 'process_capability'],
    tags=['spc', 'capability', 'six_sigma'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Cp - potential capability (assumes centered process)."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    usl = params.get('usl')  # Upper Specification Limit
    lsl = params.get('lsl')  # Lower Specification Limit
    
    if not measure_column:
        raise ValueError('measure_column required')
    if usl is None or lsl is None:
        raise ValueError('Both USL and LSL required')
    
    # Get data
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 2:
        return {'error': 'Insufficient data', 'n': len(values)}
    
    # Estimate process standard deviation
    # Use within-subgroup variation (short-term)
    mr = np.abs(np.diff(values))
    sigma = np.mean(mr) / 1.128
    
    # Calculate Cp
    tolerance = usl - lsl
    cp = tolerance / (6 * sigma)
    
    # Interpret Cp
    if cp >= 2.0:
        interpretation = 'Excellent (Six Sigma level)'
        classification = 'world_class'
    elif cp >= 1.33:
        interpretation = 'Adequate (meets requirements)'
        classification = 'adequate'
    elif cp >= 1.0:
        interpretation = 'Marginal (barely capable)'
        classification = 'marginal'
    else:
        interpretation = 'Inadequate (not capable)'
        classification = 'inadequate'
    
    return {
        'cp': float(cp),
        'usl': float(usl),
        'lsl': float(lsl),
        'tolerance': float(tolerance),
        'sigma': float(sigma),
        'sigma_short_term': float(sigma),
        'n': len(values),
        'interpretation': interpretation,
        'classification': classification,
        'capable': cp >= 1.33,
        'minimum_acceptable': 1.33,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
