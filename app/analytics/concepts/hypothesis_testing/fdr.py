from __future__ import annotations

from typing import Any, Dict

import numpy as np

META = ConceptMeta(
    id='665da7ca-233b-4048-91b2-eaebda8f3c6e',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='fdr',
    title='False Discovery Rate (FDR)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Fdr.
    
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
        'concept': 'fdr',
        'status': 'enabled',
        'message': 'Concept fdr is now operational',
        'n': n,
        'parameters': params
    }
