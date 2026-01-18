from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='box-cox-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='box-cox',
    title='Box Cox',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['box_cox'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    from scipy import stats
    import numpy as np
    
    column = params.get('column')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL AND {column} > 0"
    data = [r[0] for r in ctx.con.execute(query).fetchall()]
    
    transformed, lmbda = stats.boxcox(data)
    
    return {
        'lambda': float(lmbda),
        'transformed_mean': float(np.mean(transformed)),
        'transformed_std': float(np.std(transformed)),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
