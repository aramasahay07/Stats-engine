from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='random-forest-final',
    topic_id='topic-final',
    topic_slug='predictive-ml',
    slug='random-forest',
    title='Random Forest',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['random_forest'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    x_cols = params.get('x_columns', [])
    y_col = params.get('y_column')
    n_estimators = params.get('n_estimators', 100)
    
    if not isinstance(x_cols, list):
        x_cols = [x_cols]
    
    cols = x_cols + [y_col]
    query = f"SELECT {', '.join(cols)} FROM dataset WHERE {' AND '.join([f'{c} IS NOT NULL' for c in cols])}"
    data = np.array(ctx.con.execute(query).fetchall())
    
    X, y = data[:, :-1], data[:, -1]
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    return {
        'n_estimators': n_estimators,
        'feature_importances': {x_cols[i]: float(model.feature_importances_[i]) for i in range(len(x_cols))},
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
