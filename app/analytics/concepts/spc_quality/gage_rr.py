from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='grr-001',
    topic_id='msa-topic',
    topic_slug='spc-quality',
    slug='gage-rr',
    title='Gage R&R (Repeatability & Reproducibility)',
    concept_type='msa',
    level='advanced',
    status='published',
    output_keys=['gage_rr', 'grr'],
    tags=['spc', 'msa', 'measurement_system'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform Gage R&R study (ANOVA method)."""
    import numpy as np
    from scipy import stats as sp_stats
    
    measurement_column = params.get('measurement_column')
    part_column = params.get('part_column')
    operator_column = params.get('operator_column')
    tolerance = params.get('tolerance')  # Specification tolerance
    
    if not all([measurement_column, part_column, operator_column]):
        raise ValueError('measurement_column, part_column, and operator_column required')
    
    # Get data
    query = f"""
        SELECT {part_column}, {operator_column}, {measurement_column}
        FROM dataset
        WHERE {measurement_column} IS NOT NULL
        ORDER BY {part_column}, {operator_column}
    """
    data = ctx.con.execute(query).fetchall()
    
    if len(data) < 6:
        return {'error': 'Need at least 6 measurements'}
    
    parts = [r[0] for r in data]
    operators = [r[1] for r in data]
    measurements = np.array([r[2] for r in data])
    
    # Convert to structured data
    import pandas as pd
    df = pd.DataFrame({
        'part': parts,
        'operator': operators,
        'measurement': measurements
    })
    
    # Calculate variance components
    grand_mean = measurements.mean()
    
    # Part variance
    part_means = df.groupby('part')['measurement'].mean()
    var_part = part_means.var()
    
    # Operator variance  
    operator_means = df.groupby('operator')['measurement'].mean()
    var_operator = operator_means.var()
    
    # Repeatability (within operator)
    var_repeatability = df.groupby(['part', 'operator'])['measurement'].var().mean()
    
    # Reproducibility (between operators)
    var_reproducibility = var_operator
    
    # Total measurement system variation (Gage R&R)
    var_gage = var_repeatability + var_reproducibility
    
    # Total variation
    var_total = measurements.var()
    
    # Calculate percentages
    grr_pct = (np.sqrt(var_gage) / np.sqrt(var_total)) * 100 if var_total > 0 else 0
    repeatability_pct = (np.sqrt(var_repeatability) / np.sqrt(var_total)) * 100 if var_total > 0 else 0
    reproducibility_pct = (np.sqrt(var_reproducibility) / np.sqrt(var_total)) * 100 if var_total > 0 else 0
    part_variation_pct = (np.sqrt(var_part) / np.sqrt(var_total)) * 100 if var_total > 0 else 0
    
    # Interpret GR&R
    if grr_pct < 10:
        assessment = 'Acceptable measurement system'
        category = 'good'
    elif grr_pct < 30:
        assessment = 'Marginal measurement system (may be acceptable)'
        category = 'marginal'
    else:
        assessment = 'Unacceptable measurement system (needs improvement)'
        category = 'unacceptable'
    
    # Number of distinct categories
    ndc = int(np.floor(1.41 * np.sqrt(var_part / var_gage))) if var_gage > 0 else 0
    
    return {
        'grr_percent': float(grr_pct),
        'repeatability_percent': float(repeatability_pct),
        'reproducibility_percent': float(reproducibility_pct),
        'part_variation_percent': float(part_variation_pct),
        'assessment': assessment,
        'category': category,
        'ndc': ndc,
        'acceptable': grr_pct < 30,
        'n_parts': len(df['part'].unique()),
        'n_operators': len(df['operator'].unique()),
        'n_measurements': len(measurements),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
