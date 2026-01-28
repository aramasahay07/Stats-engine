from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='7caef142-f846-4720-8afb-ee49359cae01',
    topic_id='d26c17e5-31ec-4121-96ac-b13c70ab676d',
    topic_slug='hypothesis-testing',
    slug='wilcoxon-signed-rank',
    title='Wilcoxon Signed-Rank Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['w_statistic', 'p_value'],
    tags=['hypothesis_test', 'nonparametric', 'paired'],
    quality_score=85,
)


async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform Wilcoxon signed-rank test for paired samples.

    Params:
      - before_column: numeric
      - after_column: numeric
      - alternative: 'two-sided'|'less'|'greater' (default 'two-sided')
      - alpha: float (default 0.05)
      - sample_limit: int (default 200000)
    """
    from scipy import stats

    before_col = params.get("before_column") or params.get("before")
    after_col = params.get("after_column") or params.get("after")
    if not before_col or not after_col:
        raise ValueError("Provide 'before_column' and 'after_column'")

    alternative = str(params.get("alternative", "two-sided")).lower()
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    alpha = float(params.get("alpha", 0.05))
    sample_limit = int(params.get("sample_limit", 200000))

    q = f"""SELECT {before_col}, {after_col}
            FROM dataset
            WHERE {before_col} IS NOT NULL AND {after_col} IS NOT NULL
            LIMIT {sample_limit}"""
    rows = ctx.con.execute(q).fetchall()
    if len(rows) < 3:
        return {"error": "Need at least 3 paired observations", "n": int(len(rows))}

    x = [r[0] for r in rows]
    y = [r[1] for r in rows]

    w, p = stats.wilcoxon(x, y, alternative=alternative)
    return {
        "before_column": before_col,
        "after_column": after_col,
        "n": int(len(rows)),
        "w_statistic": float(w),
        "p_value": float(p),
        "alternative": alternative,
        "significant": p < alpha,
        "reject_null": p < alpha,
        "alpha": float(alpha),
        "sample_limit": int(sample_limit),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
