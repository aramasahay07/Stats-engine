from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import stats

META = ConceptMeta(
    id='cfbc5ff5-a1ce-4ed4-a966-47c15bcbd3fc',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='outliers',
    title='Outliers',
    concept_type='procedure',
    level='intro',
    status='published',
    output_keys=['outliers', 'outlier'],
    tags=['quality', 'variation'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Detect outliers using IQR or Z-score method."""
    column = params.get('column', params.get('measure_column'))
    if not column:
        raise ValueError('column parameter is required')
    
    method = params.get('method', 'iqr')
    query = f"SELECT rowid, {column} FROM dataset WHERE {column} IS NOT NULL"
    result = ctx.con.execute(query).fetchall()
    
    if len(result) < 4:
        return {'error': 'Need at least 4 values'}
    
    row_ids = [r[0] for r in result]
    data = np.array([r[1] for r in result])
    
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (data < lower) | (data > upper)
        
        return {
            'outlier_count': int(np.sum(outlier_mask)),
            'outlier_indices': [row_ids[i] for i, x in enumerate(outlier_mask) if x],
            'outlier_values': data[outlier_mask].tolist(),
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'method': 'iqr',
            'measure': column
        }
    else:
        z_scores = np.abs(stats.zscore(data, ddof=1))
        outlier_mask = z_scores > 3
        
        return {
            'outlier_count': int(np.sum(outlier_mask)),
            'outlier_indices': [row_ids[i] for i, x in enumerate(outlier_mask) if x],
            'outlier_values': data[outlier_mask].tolist(),
            'method': 'zscore',
            'measure': column
        }
