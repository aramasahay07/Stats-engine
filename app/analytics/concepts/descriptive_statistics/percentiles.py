from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='4bbf3927-946c-49ef-a71c-9b05fe42348a',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='percentiles',
    title='Percentiles',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['percentile', 'pctl'],
    tags=['distribution'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate percentiles of a numeric column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    percentiles_list = params.get('percentiles', [10, 25, 50, 75, 90, 95, 99])
    percentiles = {}
    
    for p in percentiles_list:
        query = f"SELECT QUANTILE({column}, {p/100.0}) FROM dataset WHERE {column} IS NOT NULL"
        result = ctx.con.execute(query).fetchone()
        percentiles[f'p{p}'] = float(result[0]) if result[0] is not None else None
    
    return {
        'percentiles': percentiles,
        'percentile_values': percentiles_list,
        'measure': column
    }
