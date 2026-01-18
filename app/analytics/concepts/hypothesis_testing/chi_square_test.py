from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='e5f6a7b8-c9d0-1e2f-3a4b-5c6d7e8f9a0b',
    topic_id='75d6fdc4-410c-4e17-87c7-2f6f5aff7f98',
    topic_slug='hypothesis-testing',
    slug='chi-square-test',
    title='Chi-Square Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['chi_square_test', 'chi_square'],
    tags=['hypothesis_test', 'categorical'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform chi-square test of independence."""
    from scipy import stats
    import numpy as np
    from collections import defaultdict
    
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
    
    table = defaultdict(lambda: defaultdict(int))
    for r, c, cnt in data:
        table[r][c] = cnt
    
    rows = sorted(table.keys())
    cols = sorted(set(c for row in table.values() for c in row.keys()))
    
    contingency = np.array([[table[r].get(c, 0) for c in cols] for r in rows])
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    
    # Cramér's V
    n = contingency.sum()
    min_dim = min(len(rows) - 1, len(cols) - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    return {
        'chi_square': float(chi2),
        'p_value': float(p),
        'df': int(dof),
        'significant': p < alpha,
        'cramers_v': float(cramers_v),
        'n': int(n),
        'alpha': float(alpha),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
