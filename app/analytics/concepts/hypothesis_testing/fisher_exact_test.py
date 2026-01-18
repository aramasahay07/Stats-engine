from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='fisher-exact-001',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='fisher-exact-test',
    title='Fisher Exact Test',
    concept_type='test',
    level='intermediate',
    status='published',
    output_keys=['fisher_exact', 'odds_ratio'],
    tags=['hypothesis_test', 'categorical'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform Fisher's exact test for 2x2 contingency tables."""
    from scipy import stats
    import numpy as np
    
    row_var = params.get('row_variable')
    col_var = params.get('column_variable')
    alpha = params.get('alpha', 0.05)
    
    if not row_var or not col_var:
        raise ValueError('Both row_variable and column_variable required')
    
    query = f"""
        SELECT {row_var}, {col_var}, COUNT(*)
        FROM dataset
        WHERE {row_var} IS NOT NULL AND {col_var} IS NOT NULL
        GROUP BY {row_var}, {col_var}
    """
    
    data = ctx.con.execute(query).fetchall()
    
    # Build 2x2 table
    from collections import defaultdict
    table = defaultdict(lambda: defaultdict(int))
    for r, c, cnt in data:
        table[r][c] = cnt
    
    if len(table) != 2:
        return {'error': 'Fisher exact test requires exactly 2 rows', 'n_rows': len(table)}
    
    rows = sorted(table.keys())
    cols = sorted(set(c for row in table.values() for c in row.keys()))
    
    if len(cols) != 2:
        return {'error': 'Fisher exact test requires exactly 2 columns', 'n_cols': len(cols)}
    
    # Create 2x2 contingency table
    contingency = np.array([[table[r].get(c, 0) for c in cols] for r in rows])
    
    # Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact(contingency)
    
    return {
        'odds_ratio': float(odds_ratio),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'contingency_table': contingency.tolist(),
        'n': int(contingency.sum()),
        'alpha': float(alpha),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
