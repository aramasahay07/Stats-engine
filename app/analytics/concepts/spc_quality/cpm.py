from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='cpm-001',
    topic_id='capability-topic',
    topic_slug='spc-quality',
    slug='cpm',
    title='Cpm (Taguchi Capability Index)',
    concept_type='capability',
    level='advanced',
    status='published',
    output_keys=['cpm', 'taguchi_capability'],
    tags=['spc', 'capability', 'taguchi'],
    quality_score=85,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Cpm - Taguchi index (penalizes off-target performance)."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    usl = params.get('usl')
    lsl = params.get('lsl')
    target = params.get('target')
    
    if not measure_column:
        raise ValueError('measure_column required')
    if usl is None or lsl is None:
        raise ValueError('Both USL and LSL required')
    if target is None:
        target = (usl + lsl) / 2  # Assume midpoint if not specified
    
    # Get data
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 2:
        return {'error': 'Insufficient data'}
    
    # Calculate mean and sigma
    mean = np.mean(values)
    mr = np.abs(np.diff(values))
    sigma = np.mean(mr) / 1.128
    
    # Taguchi loss function - includes deviation from target
    tau_squared = sigma**2 + (mean - target)**2
    
    # Calculate Cpm
    tolerance = usl - lsl
    cpm = tolerance / (6 * np.sqrt(tau_squared))
    
    # Also calculate Cpk for comparison
    cpu = (usl - mean) / (3 * sigma)
    cpl = (mean - lsl) / (3 * sigma)
    cpk = min(cpu, cpl)
    
    # Interpret
    if cpm >= 1.33:
        interpretation = 'Process centered and capable'
    elif cpm >= 1.0:
        interpretation = 'Process marginally capable'
    else:
        interpretation = 'Process not capable or not centered'
    
    return {
        'cpm': float(cpm),
        'cpk': float(cpk),
        'target': float(target),
        'mean': float(mean),
        'distance_from_target': float(abs(mean - target)),
        'sigma': float(sigma),
        'tau': float(np.sqrt(tau_squared)),
        'usl': float(usl),
        'lsl': float(lsl),
        'interpretation': interpretation,
        'n': len(values),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
