from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='kmeans-func',
    topic_id='topic-id',
    topic_slug='advanced-methods',
    slug='kmeans',
    title='K-Means Clustering',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['kmeans'],
    tags=['advanced-methods'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """K-Means Clustering - fully functional implementation."""
    import numpy as np
    from sklearn.cluster import KMeans
    
    columns = params.get('columns', [])
    n_clusters = params.get('n_clusters', 3)
    
    if not isinstance(columns, list):
        columns = [columns]
    
    query = f"SELECT {', '.join(columns)} FROM dataset WHERE {' AND '.join([f'{c} IS NOT NULL' for c in columns])}"
    data = np.array(ctx.con.execute(query).fetchall())
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    
    return {
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'labels': labels.tolist(),
        'n_clusters': n_clusters,
        'inertia': float(kmeans.inertia_),
        'n': len(data),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
