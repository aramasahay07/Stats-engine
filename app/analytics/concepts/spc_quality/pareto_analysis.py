from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='pareto-001',
    topic_id='quality-tools-topic',
    topic_slug='spc-quality',
    slug='pareto-analysis',
    title='Pareto Analysis (80/20 Rule)',
    concept_type='quality_tool',
    level='intro',
    status='published',
    output_keys=['pareto', 'pareto_analysis'],
    tags=['quality', 'tools', 'pareto'],
    quality_score=90,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform Pareto analysis to identify vital few vs trivial many."""
    import numpy as np
    
    category_column = params.get('category_column')
    value_column = params.get('value_column')  # Optional - defaults to count
    
    if not category_column:
        raise ValueError('category_column required')
    
    # Get data
    if value_column:
        query = f"SELECT {category_column}, SUM({value_column}) as total FROM dataset WHERE {category_column} IS NOT NULL GROUP BY {category_column} ORDER BY total DESC"
    else:
        query = f"SELECT {category_column}, COUNT(*) as total FROM dataset WHERE {category_column} IS NOT NULL GROUP BY {category_column} ORDER BY total DESC"
    
    data = ctx.con.execute(query).fetchall()
    
    if not data:
        return {'error': 'No data'}
    
    categories = [r[0] for r in data]
    values = np.array([r[1] for r in data])
    
    # Calculate cumulative percentages
    total = values.sum()
    percentages = (values / total) * 100
    cumulative = np.cumsum(percentages)
    
    # Find 80% threshold
    threshold_80 = np.where(cumulative >= 80)[0]
    n_vital_few = threshold_80[0] + 1 if len(threshold_80) > 0 else len(categories)
    
    # Create Pareto items
    pareto_items = []
    for i, (cat, val, pct, cum) in enumerate(zip(categories, values, percentages, cumulative)):
        pareto_items.append({
            'rank': i + 1,
            'category': str(cat),
            'value': float(val),
            'percent': float(pct),
            'cumulative_percent': float(cum),
            'vital_few': i < n_vital_few,
        })
    
    return {
        'pareto_items': pareto_items,
        'n_categories': len(categories),
        'n_vital_few': n_vital_few,
        'vital_few_percent': float((n_vital_few / len(categories)) * 100),
        'vital_few_contribution': float(cumulative[n_vital_few-1]) if n_vital_few <= len(cumulative) else 100.0,
        'total': float(total),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
