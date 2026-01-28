from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='db03f307-91c1-41ca-9530-fa98e074dad7',
    topic_id='779f054c-3d03-43ab-a7a1-f7395f9ac718',
    topic_slug='hypothesis-testing',
    slug='chi-square-goodness-of-fit',
    title='Chi-Square Goodness of Fit',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['chi_square_goodness_of_fit'],
    tags=['hypothesis_test', 'chi_square'],
    quality_score=85,
)


async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Chi-square goodness-of-fit test for one categorical variable.

    Params:
      - category_column: categorical (or column)
      - expected_probs: list[float] (optional; must sum to 1 and match k categories)
      - alpha: float (default 0.05)
      - sample_limit: int (default 200000)
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import chisquare

    col = params.get("category_column") or params.get("column")
    if not col:
        raise ValueError("Provide category_column (or column)")

    alpha = float(params.get("alpha", 0.05))
    sample_limit = int(params.get("sample_limit", 200000))

    q = f"""
        SELECT {col}
        FROM dataset
        WHERE {col} IS NOT NULL
        LIMIT {sample_limit}
    """
    rows = ctx.con.execute(q).fetchall()
    if len(rows) < 3:
        return {"error": "Need at least 3 rows", "n": int(len(rows))}

    values = [r[0] for r in rows]
    s = pd.Series(values, dtype="object")
    counts = s.value_counts(dropna=True).sort_index()

    observed = counts.values.astype(float)
    labels = [str(i) for i in counts.index.tolist()]
    k = len(observed)
    if k < 2:
        return {"error": "Need at least 2 categories", "k": int(k)}

    expected_probs = params.get("expected_probs")
    if expected_probs is not None:
        if not isinstance(expected_probs, list) or len(expected_probs) != k:
            raise ValueError(f"expected_probs must be a list of length {k}")
        probs = np.array(expected_probs, dtype=float)
        if probs.sum() <= 0:
            raise ValueError("expected_probs must sum to 1")
        probs = probs / probs.sum()
        expected = probs * observed.sum()
    else:
        expected = np.ones(k, dtype=float) * (observed.sum() / k)

    chi2, p = chisquare(f_obs=observed, f_exp=expected)

    return {
        "category_column": col,
        "n": int(observed.sum()),
        "labels": labels,
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(k - 1),
        "significant": p < alpha,
        "reject_null": p < alpha,
        "alpha": float(alpha),
        "sample_limit": int(sample_limit),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
