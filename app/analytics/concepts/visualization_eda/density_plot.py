from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='density-plot-final',
    topic_id='topic-final',
    topic_slug='visualization-eda',
    slug='density-plot',
    title='Density Plot',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['density_plot'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    from scipy.stats import gaussian_kde
    
    column = params.get('column')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    kde = gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 100)
    density = kde(x_range)
    
    return {
        'x_values': x_range.tolist(),
        'density': density.tolist(),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
