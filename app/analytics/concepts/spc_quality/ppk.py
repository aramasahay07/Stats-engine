from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='ppk-001',
    topic_id='capability-topic',
    topic_slug='spc-quality',
    slug='ppk',
    title='Ppk (Process Performance Index)',
    concept_type='capability',
    level='intermediate',
    status='published',
    output_keys=['ppk', 'process_performance_index'],
    tags=['spc', 'capability', 'long_term'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Ppk - actual process performance (long-term, accounts for centering)."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    usl = params.get('usl')
    lsl = params.get('lsl')
    
    if not measure_column:
        raise ValueError('measure_column required')
    if usl is None and lsl is None:
        raise ValueError('At least one specification limit required')
    
    # Get data
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 2:
        return {'error': 'Insufficient data'}
    
    # Calculate mean and overall sigma
    mean = np.mean(values)
    sigma_overall = np.std(values, ddof=1)
    
    # Calculate Ppu and Ppl
    ppu = (usl - mean) / (3 * sigma_overall) if usl is not None else None
    ppl = (mean - lsl) / (3 * sigma_overall) if lsl is not None else None
    
    # Ppk is minimum
    if ppu is not None and ppl is not None:
        ppk = min(ppu, ppl)
    elif ppu is not None:
        ppk = ppu
    else:
        ppk = ppl
    
    # Interpret
    if ppk >= 1.67:
        interpretation = 'Excellent'
    elif ppk >= 1.33:
        interpretation = 'Adequate'
    elif ppk >= 1.0:
        interpretation = 'Marginal'
    else:
        interpretation = 'Inadequate'
    
    return {
        'ppk': float(ppk),
        'ppu': float(ppu) if ppu is not None else None,
        'ppl': float(ppl) if ppl is not None else None,
        'mean': float(mean),
        'sigma_overall': float(sigma_overall),
        'usl': float(usl) if usl is not None else None,
        'lsl': float(lsl) if lsl is not None else None,
        'interpretation': interpretation,
        'n': len(values),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
