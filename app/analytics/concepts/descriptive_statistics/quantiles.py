from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='193ba2a8-c764-4507-8ab3-90447495d578',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='quantiles',
    title='Quantiles',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['quantile'],
    tags=['distribution'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate quantiles of a numeric column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    probabilities = params.get('probabilities', [0.25, 0.5, 0.75])
    quantiles = {}
    
    for p in probabilities:
        query = f"SELECT QUANTILE({column}, {p}) FROM dataset WHERE {column} IS NOT NULL"
        result = ctx.con.execute(query).fetchone()
        quantiles[f'q{int(p*100)}'] = float(result[0]) if result[0] is not None else None
    
    return {
        'quantiles': quantiles,
        'probabilities': probabilities,
        'measure': column
    }
