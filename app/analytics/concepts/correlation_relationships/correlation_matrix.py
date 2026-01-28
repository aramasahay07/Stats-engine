from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict, List, Union

META = ConceptMeta(
    id='d6117c3f-0892-4f2a-8423-51c66b7d3a45',
    topic_id='bdac1606-8d79-4af9-9935-b727677eb008',
    topic_slug='correlation-relationships',
    slug='correlation-matrix',
    title='Correlation Matrix',
    concept_type='analysis',
    level='intro',
    status='published',
    output_keys=['columns', 'matrix'],
    tags=['correlation', 'matrix'],
    quality_score=85,
)


def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i]
    return [str(x)]

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a Pearson correlation matrix across multiple numeric columns.

    Params:
      - columns: list[str] (required; at least 2)
      - method: 'pearson'|'spearman' (default 'pearson')
      - sample_limit: int (default 200000)
      - min_complete_rows: int (default 3)
    """
    import pandas as pd

    cols = _as_list(params.get("columns"))
    if len(cols) < 2:
        raise ValueError("Provide 'columns' as a list with at least 2 columns")

    method = str(params.get("method", "pearson")).lower()
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    sample_limit = int(params.get("sample_limit", 200000))
    min_complete_rows = int(params.get("min_complete_rows", 3))

    select_cols = ", ".join(cols)
    q = f"SELECT {select_cols} FROM dataset LIMIT {sample_limit}"
    rows = ctx.con.execute(q).fetchall()

    df = pd.DataFrame(rows, columns=cols)
    df = df.dropna(axis=0, how="any")
    if len(df) < min_complete_rows:
        return {
            "error": "Not enough complete rows after dropping nulls",
            "n_complete_rows": int(len(df)),
            "columns": cols,
        }

    corr = df.corr(method=method)
    # to native JSON types
    return {
        "method": method,
        "columns": list(corr.columns),
        "matrix": corr.values.tolist(),
        "n_complete_rows": int(len(df)),
        "sample_limit": int(sample_limit),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
