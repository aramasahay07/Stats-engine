from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='decision-tree-final',
    topic_id='topic-final',
    topic_slug='predictive-ml',
    slug='decision-tree',
    title='Decision Tree',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['decision_tree'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    
    x_cols = params.get('x_columns', [])
    y_col = params.get('y_column')
    max_depth = params.get('max_depth', 3)
    
    if not isinstance(x_cols, list):
        x_cols = [x_cols]
    
    cols = x_cols + [y_col]
    query = f"SELECT {', '.join(cols)} FROM dataset WHERE {' AND '.join([f'{c} IS NOT NULL' for c in cols])}"
    data = np.array(ctx.con.execute(query).fetchall())
    
    X, y = data[:, :-1], data[:, -1]
    
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    
    return {
        'max_depth': max_depth,
        'n_leaves': int(model.get_n_leaves()),
        'feature_importances': {x_cols[i]: float(model.feature_importances_[i]) for i in range(len(x_cols))},
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
