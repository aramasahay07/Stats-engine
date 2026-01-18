from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='effect-size-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='effect-size',
    title='Effect Size',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['effect_size', 'cohens_d'],
    tags=['hypothesis_test', 'effect_size'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate effect size (Cohen's d) for group comparisons."""
    import numpy as np
    
    measure_column = params.get('measure_column', params.get('column'))
    group_column = params.get('group_column')
    
    if not measure_column or not group_column:
        raise ValueError('Both measure_column and group_column required')
    
    # Get two groups
    groups = ctx.con.execute(f"SELECT DISTINCT {group_column} FROM dataset WHERE {group_column} IS NOT NULL LIMIT 2").fetchall()
    
    if len(groups) < 2:
        return {'error': 'Need 2 groups for effect size'}
    
    g1, g2 = groups[0][0], groups[1][0]
    
    data1 = [r[0] for r in ctx.con.execute(f"SELECT {measure_column} FROM dataset WHERE {group_column}=? AND {measure_column} IS NOT NULL", [g1]).fetchall()]
    data2 = [r[0] for r in ctx.con.execute(f"SELECT {measure_column} FROM dataset WHERE {group_column}=? AND {measure_column} IS NOT NULL", [g2]).fetchall()]
    
    if len(data1) < 2 or len(data2) < 2:
        return {'error': 'Each group needs >= 2 observations'}
    
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else None
    
    # Interpret effect size
    if cohens_d is None:
        magnitude = 'undefined'
    elif abs(cohens_d) < 0.2:
        magnitude = 'negligible'
    elif abs(cohens_d) < 0.5:
        magnitude = 'small'
    elif abs(cohens_d) < 0.8:
        magnitude = 'medium'
    else:
        magnitude = 'large'
    
    return {
        'cohens_d': float(cohens_d) if cohens_d else None,
        'effect_size': float(cohens_d) if cohens_d else None,
        'magnitude': magnitude,
        'group1': str(g1),
        'mean1': float(mean1),
        'n1': n1,
        'group2': str(g2),
        'mean2': float(mean2),
        'n2': n2,
        'pooled_std': float(pooled_std),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
