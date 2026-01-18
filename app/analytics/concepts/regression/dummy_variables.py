from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='dummy-variables-final',
    topic_id='topic-final',
    topic_slug='regression',
    slug='dummy-variables',
    title='Dummy Variables',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['dummy_variables'],
    tags=['regression'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    cat_column = params.get('categorical_column')
    y_column = params.get('y_column')
    
    query = f"SELECT {cat_column}, {y_column} FROM dataset WHERE {cat_column} IS NOT NULL AND {y_column} IS NOT NULL"
    data = ctx.con.execute(query).fetchall()
    
    import pandas as pd
    df = pd.DataFrame(data, columns=['category', 'y'])
    
    dummies = pd.get_dummies(df['category'], prefix=cat_column)
    
    return {
        'dummy_variables': dummies.columns.tolist(),
        'n_categories': len(dummies.columns),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
