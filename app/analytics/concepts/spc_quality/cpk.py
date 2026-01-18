from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='cpk-001',
    topic_id='capability-topic',
    topic_slug='spc-quality',
    slug='cpk',
    title='Cpk (Process Capability - Actual)',
    concept_type='capability',
    level='intermediate',
    status='published',
    output_keys=['cpk', 'process_capability_index'],
    tags=['spc', 'capability', 'six_sigma'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Cpk - actual capability (accounts for centering)."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    usl = params.get('usl')
    lsl = params.get('lsl')
    target = params.get('target')  # Optional target value
    
    if not measure_column:
        raise ValueError('measure_column required')
    if usl is None and lsl is None:
        raise ValueError('At least one specification limit (USL or LSL) required')
    
    # Get data
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 2:
        return {'error': 'Insufficient data'}
    
    # Calculate mean and sigma
    mean = np.mean(values)
    mr = np.abs(np.diff(values))
    sigma = np.mean(mr) / 1.128
    
    # Calculate Cpu and Cpl
    cpu = (usl - mean) / (3 * sigma) if usl is not None else None
    cpl = (mean - lsl) / (3 * sigma) if lsl is not None else None
    
    # Cpk is minimum of Cpu and Cpl
    if cpu is not None and cpl is not None:
        cpk = min(cpu, cpl)
        limiting_side = 'upper' if cpu < cpl else 'lower'
    elif cpu is not None:
        cpk = cpu
        limiting_side = 'upper'
    else:
        cpk = cpl
        limiting_side = 'lower'
    
    # Calculate Cp for comparison
    if usl is not None and lsl is not None:
        cp = (usl - lsl) / (6 * sigma)
        centering_loss = cp - cpk
    else:
        cp = None
        centering_loss = None
    
    # PPM (parts per million defective)
    if usl is not None and lsl is not None:
        from scipy import stats
        ppm_upper = (1 - stats.norm.cdf((usl - mean) / sigma)) * 1e6
        ppm_lower = stats.norm.cdf((lsl - mean) / sigma) * 1e6
        ppm_total = ppm_upper + ppm_lower
    else:
        ppm_total = None
    
    # Interpret Cpk
    if cpk >= 2.0:
        interpretation = 'Excellent (Six Sigma level)'
        classification = 'world_class'
    elif cpk >= 1.33:
        interpretation = 'Adequate (meets requirements)'
        classification = 'adequate'
    elif cpk >= 1.0:
        interpretation = 'Marginal (barely capable)'
        classification = 'marginal'
    else:
        interpretation = 'Inadequate (not capable)'
        classification = 'inadequate'
    
    result = {
        'cpk': float(cpk),
        'cpu': float(cpu) if cpu is not None else None,
        'cpl': float(cpl) if cpl is not None else None,
        'mean': float(mean),
        'sigma': float(sigma),
        'limiting_side': limiting_side,
        'interpretation': interpretation,
        'classification': classification,
        'capable': cpk >= 1.33,
        'n': len(values),
    }
    
    if cp is not None:
        result['cp'] = float(cp)
        result['centering_loss'] = float(centering_loss)
    
    if usl is not None:
        result['usl'] = float(usl)
    if lsl is not None:
        result['lsl'] = float(lsl)
    if ppm_total is not None:
        result['ppm_defective'] = float(ppm_total)
    
    return result

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
