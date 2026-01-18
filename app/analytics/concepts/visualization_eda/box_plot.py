from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='box-plot-func',
    topic_id='topic-id',
    topic_slug='visualization-eda',
    slug='box-plot',
    title='Box Plot',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['box_plot'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Box Plot - fully functional implementation."""
    import numpy as np
    
    column = params.get('column')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    
    return {
        'min': float(np.min(data)),
        'q1': float(q1),
        'median': float(median),
        'q3': float(q3),
        'max': float(np.max(data)),
        'iqr': float(iqr),
        'lower_whisker': float(q1 - 1.5 * iqr),
        'upper_whisker': float(q3 + 1.5 * iqr),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
