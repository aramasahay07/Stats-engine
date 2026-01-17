from __future__ import annotations

from typing import Any, Dict

from scipy import stats

META = ConceptMeta(
    id='e59f9621-91c2-45be-9fa0-573f981ec9dc',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='kurtosis',
    title='Kurtosis',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['kurtosis'],
    tags=['shape'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the kurtosis of a distribution."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = ctx.con.execute(query).fetchnumpy()[column]
    
    if len(data) < 4:
        return {'kurtosis': None, 'error': 'Need at least 4 values'}
    
    kurt = float(stats.kurtosis(data, bias=False, fisher=True))
    
    if abs(kurt) < 0.5:
        interpretation = 'mesokurtic (normal-like tails)'
    elif kurt > 0.5:
        interpretation = 'leptokurtic (heavy tails)'
    else:
        interpretation = 'platykurtic (light tails)'
    
    return {
        'kurtosis': kurt,
        'excess_kurtosis': kurt,
        'interpretation': interpretation,
        'valid_count': len(data),
        'measure': column
    }
