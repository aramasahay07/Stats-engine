from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='sigma-level-001',
    topic_id='six-sigma-topic',
    topic_slug='spc-quality',
    slug='sigma-level',
    title='Sigma Level (Process Sigma)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['sigma_level', 'process_sigma'],
    tags=['spc', 'six_sigma', 'quality'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Sigma Level from Cpk or DPMO."""
    from scipy import stats
    import numpy as np
    
    # Can calculate from Cpk or DPMO
    cpk = params.get('cpk')
    dpmo = params.get('dpmo')
    
    if cpk is not None:
        # Sigma level from Cpk (assuming 1.5 sigma shift)
        z_short = cpk * 3
        z_long = z_short + 1.5
        sigma_level = z_long
        
        # Calculate corresponding DPMO
        yield_rate = stats.norm.cdf(z_short)
        dpmo_calc = (1 - yield_rate) * 1_000_000
        
    elif dpmo is not None:
        # Sigma level from DPMO
        if dpmo <= 0:
            sigma_level = 6.0
            dpmo_calc = dpmo
        elif dpmo >= 999_999:
            sigma_level = 0
            dpmo_calc = dpmo
        else:
            yield_rate = 1 - (dpmo / 1_000_000)
            z_short = stats.norm.ppf(yield_rate)
            z_long = z_short + 1.5
            sigma_level = z_long
            dpmo_calc = dpmo
    else:
        raise ValueError('Need either cpk or dpmo parameter')
    
    # Sigma level lookup table
    sigma_benchmarks = {
        6.0: {'dpmo': 3.4, 'yield': 99.99966, 'quality': 'World Class'},
        5.0: {'dpmo': 233, 'yield': 99.977, 'quality': 'Excellent'},
        4.0: {'dpmo': 6210, 'yield': 99.379, 'quality': 'Above Average'},
        3.0: {'dpmo': 66807, 'yield': 93.32, 'quality': 'Industry Average'},
        2.0: {'dpmo': 308537, 'yield': 69.15, 'quality': 'Below Average'},
        1.0: {'dpmo': 690000, 'yield': 31.0, 'quality': 'Poor'},
    }
    
    # Find closest benchmark
    closest = min(sigma_benchmarks.keys(), key=lambda x: abs(x - sigma_level))
    benchmark = sigma_benchmarks[closest]
    
    return {
        'sigma_level': float(sigma_level),
        'process_sigma': float(sigma_level),
        'dpmo': float(dpmo_calc),
        'yield_pct': float((1 - dpmo_calc/1_000_000) * 100),
        'quality_level': benchmark['quality'],
        'cpk_equivalent': float(sigma_level - 1.5) / 3 if sigma_level >= 1.5 else 0,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
