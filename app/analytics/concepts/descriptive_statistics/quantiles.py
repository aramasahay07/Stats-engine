from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='quantile-001',
    topic_id='desc-stats-topic',
    topic_slug='descriptive-statistics',
    slug='quantiles',
    title='Quantiles',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['quantiles', 'quantile'],
    tags=['descriptive', 'position'],
    quality_score=95,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate quantiles (quartiles, quintiles, deciles, etc.)."""
    column = params.get('column', params.get('measure_column'))
    q_type = params.get('type', 'quartiles')  # quartiles, quintiles, deciles, percentiles
    
    if not column:
        raise ValueError('column parameter required')
    
    # Determine quantile points based on type
    if q_type == 'quartiles':
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        labels = ['min', 'q1', 'q2', 'q3', 'max']
    elif q_type == 'quintiles':
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['min', 'q1', 'q2', 'q3', 'q4', 'max']
    elif q_type == 'deciles':
        quantiles = [i/10 for i in range(11)]
        labels = [f'd{i}' if i not in [0, 10] else ('min' if i==0 else 'max') for i in range(11)]
    else:  # custom
        quantiles = params.get('quantiles', [0, 0.25, 0.5, 0.75, 1.0])
        labels = [f'q{q}' for q in quantiles]
    
    # Build query
    quantile_queries = []
    for i, q in enumerate(quantiles):
        if q == 0:
            quantile_queries.append(f"MIN({column}) as {labels[i]}")
        elif q == 1:
            quantile_queries.append(f"MAX({column}) as {labels[i]}")
        else:
            quantile_queries.append(f"PERCENTILE_CONT({q}) WITHIN GROUP (ORDER BY {column}) as {labels[i]}")
    
    query = f"""
        SELECT 
            {', '.join(quantile_queries)},
            COUNT({column}) as n
        FROM dataset 
        WHERE {column} IS NOT NULL
    """
    
    result = ctx.con.execute(query).fetchone()
    
    n = int(result[len(quantiles)])
    
    if n < 1:
        return {'error': 'No valid data', 'n': n}
    
    # Extract quantile values
    quantile_values = {}
    for i, label in enumerate(labels):
        val = float(result[i]) if result[i] is not None else None
        quantile_values[label] = val
    
    return {
        **quantile_values,
        'type': q_type,
        'n_quantiles': len(quantiles),
        'n': n,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
