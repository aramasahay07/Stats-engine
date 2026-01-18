from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='pca-func',
    topic_id='topic-id',
    topic_slug='advanced-methods',
    slug='pca',
    title='Principal Component Analysis',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['pca'],
    tags=['advanced-methods'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Principal Component Analysis - fully functional implementation."""
    import numpy as np
    from sklearn.decomposition import PCA
    
    columns = params.get('columns', [])
    n_components = params.get('n_components', 2)
    
    if not isinstance(columns, list):
        columns = [columns]
    
    query = f"SELECT {', '.join(columns)} FROM dataset WHERE {' AND '.join([f'{c} IS NOT NULL' for c in columns])}"
    data = np.array(ctx.con.execute(query).fetchall())
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'n_components': n_components,
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
