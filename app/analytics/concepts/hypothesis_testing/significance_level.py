from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='alpha-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='significance-level',
    title='Significance Level',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['significance_level', 'alpha'],
    tags=['hypothesis_test'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Determine appropriate significance level and critical values."""
    from scipy import stats
    
    alpha = params.get('alpha', 0.05)
    test_type = params.get('test_type', 't')  # 't', 'z', 'f', 'chi2'
    df = params.get('df', params.get('degrees_of_freedom'))
    tail = params.get('tail', 'two')  # 'two', 'left', 'right'
    
    critical_values = {}
    
    if test_type == 't' and df:
        if tail == 'two':
            critical_values['t_critical'] = float(stats.t.ppf(1 - alpha/2, df))
        elif tail == 'right':
            critical_values['t_critical'] = float(stats.t.ppf(1 - alpha, df))
        else:
            critical_values['t_critical'] = float(stats.t.ppf(alpha, df))
    
    elif test_type == 'z':
        if tail == 'two':
            critical_values['z_critical'] = float(stats.norm.ppf(1 - alpha/2))
        elif tail == 'right':
            critical_values['z_critical'] = float(stats.norm.ppf(1 - alpha))
        else:
            critical_values['z_critical'] = float(stats.norm.ppf(alpha))
    
    elif test_type == 'chi2' and df:
        critical_values['chi2_critical'] = float(stats.chi2.ppf(1 - alpha, df))
    
    return {
        'alpha': float(alpha),
        'significance_level': float(alpha),
        'confidence_level': float(1 - alpha),
        'test_type': test_type,
        'tail': tail,
        'df': int(df) if df else None,
        **critical_values,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
