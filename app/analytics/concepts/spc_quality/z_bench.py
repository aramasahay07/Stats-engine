from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='zbench-001',
    topic_id='six-sigma-topic',
    topic_slug='spc-quality',
    slug='z-bench',
    title='Z-Bench (Benchmark Z-score)',
    concept_type='metric',
    level='advanced',
    status='published',
    output_keys=['z_bench', 'benchmark_z'],
    tags=['spc', 'six_sigma', 'benchmark'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Z-bench - standard normal score for capability."""
    import numpy as np
    from scipy import stats
    
    measure_column = params.get('measure_column')
    usl = params.get('usl')
    lsl = params.get('lsl')
    
    if not measure_column:
        raise ValueError('measure_column required')
    if usl is None and lsl is None:
        raise ValueError('At least one spec limit required')
    
    # Get data
    query = f"SELECT {measure_column} FROM dataset WHERE {measure_column} IS NOT NULL"
    values = np.array([r[0] for r in ctx.con.execute(query).fetchall()])
    
    if len(values) < 2:
        return {'error': 'Insufficient data'}
    
    mean = np.mean(values)
    sigma = np.std(values, ddof=1)
    
    # Calculate Z-scores to spec limits
    z_usl = (usl - mean) / sigma if usl is not None else None
    z_lsl = (mean - lsl) / sigma if lsl is not None else None
    
    # Z-bench is the minimum (worst case)
    if z_usl is not None and z_lsl is not None:
        z_bench = min(z_usl, z_lsl)
        limiting = 'upper' if z_usl < z_lsl else 'lower'
    elif z_usl is not None:
        z_bench = z_usl
        limiting = 'upper'
    else:
        z_bench = z_lsl
        limiting = 'lower'
    
    # Calculate expected defect rate
    defect_rate = stats.norm.sf(z_bench)  # Survival function (1 - CDF)
    dpmo = defect_rate * 1_000_000
    
    return {
        'z_bench': float(z_bench),
        'z_usl': float(z_usl) if z_usl is not None else None,
        'z_lsl': float(z_lsl) if z_lsl is not None else None,
        'limiting_side': limiting,
        'dpmo': float(dpmo),
        'defect_rate': float(defect_rate),
        'sigma_level': float(z_bench),
        'mean': float(mean),
        'sigma': float(sigma),
        'n': len(values),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
