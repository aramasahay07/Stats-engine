from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='dpmo-001',
    topic_id='six-sigma-topic',
    topic_slug='spc-quality',
    slug='dpmo',
    title='DPMO (Defects Per Million Opportunities)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['dpmo', 'defects_per_million'],
    tags=['spc', 'six_sigma', 'quality'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate DPMO - defects per million opportunities."""
    defects_column = params.get('defects_column')
    units_column = params.get('units_column')
    opportunities_per_unit = params.get('opportunities_per_unit', 1)
    
    # Alternative: provide defects, units, opportunities directly
    defects = params.get('defects')
    units = params.get('units')
    
    if defects_column and units_column:
        query = f"SELECT SUM({defects_column}), SUM({units_column}) FROM dataset"
        result = ctx.con.execute(query).fetchone()
        defects = result[0] if result[0] else 0
        units = result[1] if result[1] else 0
    elif defects is None or units is None:
        raise ValueError('Need either defects_column/units_column or defects/units')
    
    if units == 0:
        return {'error': 'No units to analyze'}
    
    # Calculate DPMO
    total_opportunities = units * opportunities_per_unit
    dpmo = (defects / total_opportunities) * 1_000_000 if total_opportunities > 0 else 0
    
    # Calculate defect rate
    dpu = defects / units if units > 0 else 0
    dpo = defects / total_opportunities if total_opportunities > 0 else 0
    
    # Estimate sigma level (assuming 1.5 sigma shift)
    from scipy import stats
    import numpy as np
    
    if dpmo > 0 and dpmo < 1_000_000:
        # Convert DPMO to Z-score
        yield_pct = 1 - (dpmo / 1_000_000)
        z_short = stats.norm.ppf(yield_pct)
        z_long = z_short + 1.5  # Add 1.5 sigma shift
        sigma_level = z_long
    else:
        sigma_level = 0 if dpmo >= 999_999 else 6.0
    
    # Interpret sigma level
    if sigma_level >= 6.0:
        quality = 'World Class (6σ)'
    elif sigma_level >= 5.0:
        quality = 'Excellent (5σ)'
    elif sigma_level >= 4.0:
        quality = 'Above Average (4σ)'
    elif sigma_level >= 3.0:
        quality = 'Industry Average (3σ)'
    else:
        quality = 'Below Average (<3σ)'
    
    return {
        'dpmo': float(dpmo),
        'defects': int(defects),
        'units': int(units),
        'opportunities_per_unit': int(opportunities_per_unit),
        'total_opportunities': int(total_opportunities),
        'dpu': float(dpu),  # Defects per unit
        'dpo': float(dpo),  # Defects per opportunity
        'yield_pct': float((1 - dpo) * 100),
        'sigma_level': float(sigma_level),
        'quality_level': quality,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
