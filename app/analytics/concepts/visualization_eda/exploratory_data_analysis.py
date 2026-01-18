from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='exploratory-data-analysis-final',
    topic_id='topic-final',
    topic_slug='visualization-eda',
    slug='exploratory-data-analysis',
    title='Exploratory Data Analysis',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['exploratory_data_analysis'],
    tags=['visualization-eda'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    import numpy as np
    
    columns = params.get('columns', [])
    
    if not isinstance(columns, list):
        columns = [columns]
    
    stats = {}
    for col in columns:
        query = f"SELECT {col} FROM dataset WHERE {col} IS NOT NULL"
        data = [r[0] for r in ctx.con.execute(query).fetchall()]
        
        stats[col] = {
            'count': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
        }
    
    return {
        'column_statistics': stats,
        'n_columns': len(columns),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
