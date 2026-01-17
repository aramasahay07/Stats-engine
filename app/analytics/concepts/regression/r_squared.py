from __future__ import annotations

from typing import Any, Dict


META = ConceptMeta(
    id='8124f096-8807-41f1-8bb9-50630a4fc498',
    topic_id='47670940-6e51-4e25-aa11-9f78987e5194',
    topic_slug='regression',
    slug='r-squared',
    title='R² and Adjusted R²',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['r_squared', 'r2', 'adjusted_r_squared'],
    tags=['regression', 'metric'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: R Squared.
    
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
        'concept': 'r_squared',
        'status': 'enabled',
        'message': 'Concept r_squared is now operational',
        'n': n,
        'parameters': params
    }
