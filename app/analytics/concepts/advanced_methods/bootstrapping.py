from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='bootstrapping-final',
    topic_id='topic-final',
    topic_slug='advanced-methods',
    slug='bootstrapping',
    title='Bootstrapping',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['bootstrapping'],
    tags=['advanced-methods'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    
    column = params.get('column')
    n_bootstrap = params.get('n_bootstrap', 1000)
    statistic = params.get('statistic', 'mean')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        if statistic == 'mean':
            bootstrap_stats.append(np.mean(sample))
        elif statistic == 'median':
            bootstrap_stats.append(np.median(sample))
    
    return {
        'bootstrap_mean': float(np.mean(bootstrap_stats)),
        'bootstrap_std': float(np.std(bootstrap_stats)),
        'ci_lower': float(np.percentile(bootstrap_stats, 2.5)),
        'ci_upper': float(np.percentile(bootstrap_stats, 97.5)),
        'n_bootstrap': n_bootstrap,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
