from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='pp-001',
    topic_id='capability-topic',
    topic_slug='spc-quality',
    slug='pp',
    title='Pp (Process Performance)',
    concept_type='capability',
    level='intermediate',
    status='published',
    output_keys=['pp', 'process_performance'],
    tags=['spc', 'capability', 'long_term'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Pp - overall process performance (long-term variation)."""
    import numpy as np
    
    measure_column = params.get('measure_column')
    usl = params.get('usl')
    lsl = params.get('lsl')
    
    if not measure_column:
        raise ValueError('measure_column required')
    if usl is None or lsl is None:
        raise ValueError('Both USL and LSL required')
    
    # Get data
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 2:
        return {'error': 'Insufficient data'}
    
    # Use overall standard deviation (long-term)
    sigma_overall = np.std(values, ddof=1)
    
    # Calculate Pp
    tolerance = usl - lsl
    pp = tolerance / (6 * sigma_overall)
    
    # Interpret
    if pp >= 2.0:
        interpretation = 'Excellent'
    elif pp >= 1.33:
        interpretation = 'Adequate'
    elif pp >= 1.0:
        interpretation = 'Marginal'
    else:
        interpretation = 'Inadequate'
    
    return {
        'pp': float(pp),
        'usl': float(usl),
        'lsl': float(lsl),
        'tolerance': float(tolerance),
        'sigma_overall': float(sigma_overall),
        'sigma_long_term': float(sigma_overall),
        'n': len(values),
        'interpretation': interpretation,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
