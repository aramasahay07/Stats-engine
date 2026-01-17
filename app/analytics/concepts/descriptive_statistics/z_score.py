from __future__ import annotations

from typing import Any, Dict

from scipy import stats
import numpy as np

META = ConceptMeta(
    id='0b5f0aef-b1d3-49be-8ad9-74f48468160a',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='z-score',
    title='Z-score',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['z_score', 'zscore'],
    tags=['standardization'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate z-scores for a column."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL ORDER BY rowid"
    data = ctx.con.execute(query).fetchnumpy()[column]
    
    if len(data) < 2:
        return {'error': 'Need at least 2 values'}
    
    z_scores = stats.zscore(data, ddof=1)
    outliers = np.where(np.abs(z_scores) > 3)[0]
    
    return {
        'z_scores': z_scores.tolist(),
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)),
        'outlier_count': len(outliers),
        'outlier_indices': outliers.tolist(),
        'valid_count': len(data),
        'measure': column
    }
