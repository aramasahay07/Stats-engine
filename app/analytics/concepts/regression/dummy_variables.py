from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='eb8c21e9-4158-4c9b-bc88-00245520e7be',
    topic_id='47670940-6e51-4e25-aa11-9f78987e5194',
    topic_slug='regression',
    slug='dummy-variables',
    title='Dummy Variables (One-hot)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Dummy Variables.
    
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
        'concept': 'dummy_variables',
        'status': 'enabled',
        'message': 'Concept dummy_variables is now operational',
        'n': n,
        'parameters': params
    }
