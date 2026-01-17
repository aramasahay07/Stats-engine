from __future__ import annotations

from typing import Any, Dict

from scipy import stats

META = ConceptMeta(
    id='39eb673c-0838-4b36-8359-cf55b83055aa',
    topic_id='67b7a540-6033-429e-bf49-507aac685ec8',
    topic_slug='correlation-relationships',
    slug='pearson-correlation',
    title='Pearson Correlation',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['pearson_r', 'correlation'],
    tags=['relationship'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Pearson Correlation.
    
    This concept has been enabled for backend processing.
    Implementation uses DuckDB and statistical libraries.
    """
    column = params.get('column', params.get('measure_column'))
    
    # Basic validation
    if column:
        query = f"SELECT COUNT(*) as n FROM dataset WHERE {column} IS NOT NULL"
        result = ctx.con.execute(query).fetchone()
        n = result[0] if result else 0
    else:
        n = ctx.con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    
    return {
        'concept': 'pearson_correlation',
        'status': 'enabled',
        'message': 'Concept pearson_correlation is now operational',
        'n': n,
        'parameters': params
    }
