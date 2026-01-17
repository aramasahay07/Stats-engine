from __future__ import annotations

from typing import Any, Dict

from scipy import stats

META = ConceptMeta(
    id='397c63f1-f87c-4482-8ff2-e75ca46e8183',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='skewness',
    title='Skewness',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['skewness', 'skew'],
    tags=['shape'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the skewness of a distribution."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = ctx.con.execute(query).fetchnumpy()[column]
    
    if len(data) < 3:
        return {'skewness': None, 'error': 'Need at least 3 values'}
    
    skewness = float(stats.skew(data, bias=False))
    
    if abs(skewness) < 0.5:
        interpretation = 'approximately symmetric'
    elif skewness > 0.5:
        interpretation = 'right-skewed (positively skewed)'
    else:
        interpretation = 'left-skewed (negatively skewed)'
    
    return {
        'skewness': skewness,
        'skew': skewness,
        'interpretation': interpretation,
        'valid_count': len(data),
        'measure': column
    }
