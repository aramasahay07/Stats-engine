from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='8e29e67c-6435-4524-803e-935eeae14721',
    topic_id='70b65abb-dae0-4ffd-93a9-5dba1509c206',
    topic_slug='time-series',
    slug='trend-analysis',
    title='Trend Analysis',
    concept_type='analysis',
    level='intro',
    status='published',
    output_keys=['slope', 'p_value', 'r_value'],
    tags=['time_series', 'trend'],
    quality_score=85,
)


async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fit a simple linear trend of observed values over time.

    Params:
      - time_column: required (datetime or numeric)
      - value_column: required (numeric)
      - sample_limit: int (default 200000)
    """
    import pandas as pd
    from scipy import stats

    time_col = params.get("time_column") or params.get("time_col")
    value_col = params.get("value_column") or params.get("observed_col") or params.get("value_col")
    if not time_col or not value_col:
        raise ValueError("Provide 'time_column' and 'value_column'")

    sample_limit = int(params.get("sample_limit", 200000))
    q = f"SELECT {time_col}, {value_col} FROM dataset WHERE {time_col} IS NOT NULL AND {value_col} IS NOT NULL LIMIT {sample_limit}"
    rows = ctx.con.execute(q).fetchall()
    if len(rows) < 3:
        return {"error": "Need at least 3 rows", "n": int(len(rows))}

    df = pd.DataFrame(rows, columns=["t", "y"])

    # Convert t to numeric
    if pd.api.types.is_datetime64_any_dtype(df["t"]) or isinstance(df["t"].iloc[0], (pd.Timestamp, )):
        tnum = pd.to_datetime(df["t"], errors="coerce").map(lambda x: x.toordinal() if pd.notnull(x) else None)
    else:
        # attempt numeric
        tnum = pd.to_numeric(df["t"], errors="coerce")

    y = pd.to_numeric(df["y"], errors="coerce")
    m = tnum.notnull() & y.notnull()
    tnum = tnum[m].astype(float)
    y = y[m].astype(float)

    if len(y) < 3:
        return {"error": "Need at least 3 valid numeric rows", "n": int(len(y))}

    lr = stats.linregress(tnum.values, y.values)
    return {
        "time_column": time_col,
        "value_column": value_col,
        "n": int(len(y)),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r_value": float(lr.rvalue),
        "r_squared": float(lr.rvalue ** 2),
        "p_value": float(lr.pvalue),
        "std_err": float(lr.stderr) if lr.stderr is not None else None,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
