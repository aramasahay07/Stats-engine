from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='feature-selection-final',
    topic_id='topic-final',
    topic_slug='predictive-ml',
    slug='feature-selection',
    title='Feature Selection',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['feature_selection'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif
    
    x_cols = params.get('x_columns', [])
    y_col = params.get('y_column')
    k = params.get('k', 5)
    
    if not isinstance(x_cols, list):
        x_cols = [x_cols]
    
    cols = x_cols + [y_col]
    query = f"SELECT {', '.join(cols)} FROM dataset WHERE {' AND '.join([f'{c} IS NOT NULL' for c in cols])}"
    data = np.array(ctx.con.execute(query).fetchall())
    
    X, y = data[:, :-1], data[:, -1]
    
    selector = SelectKBest(f_classif, k=min(k, len(x_cols)))
    selector.fit(X, y)
    
    scores = selector.scores_
    selected = selector.get_support()
    
    return {
        'selected_features': [x_cols[i] for i, s in enumerate(selected) if s],
        'feature_scores': {x_cols[i]: float(scores[i]) for i in range(len(x_cols))},
        'k': k,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
