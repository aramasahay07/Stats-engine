from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from app.engine.duckdb_engine import DuckDBEngine
from app.models.query import QuerySpec
from app.models.pipelines import PipelineStep
from app.engine.pipeline import ensure_pipeline_view
from app.db import registry
from app.config import settings


ALLOWED_FILTER_OPS = {
    "=", "!=", "<", "<=", ">", ">=",
    "IN", "NOT IN",
    "LIKE", "ILIKE",
    "IS", "IS NOT",
}


def _duckdb_col_types(con, view_name: str) -> dict[str, str]:
    """
    Returns {column_name: duckdb_type} for the registered parquet view.
    """
    rows = con.execute(f"DESCRIBE SELECT * FROM {view_name}").fetchall()
    # DuckDB DESCRIBE returns tuples like:
    # (column_name, column_type, null, key, default, extra)
    return {r[0]: str(r[1]).upper() for r in rows}


def _is_numeric_type(t: str) -> bool:
    t = (t or "").upper()
    return any(x in t for x in [
        "INT", "BIGINT", "SMALLINT", "TINYINT",
        "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL",
    ])


def _get_measure_name_expr(m):
    # supports either Pydantic model (m.name/m.expr) or dict (m["name"]/m["expr"])
    if isinstance(m, dict):
        return m.get("name"), m.get("expr")
    return getattr(m, "name", None), getattr(m, "expr", None)


def _make_measure_like(existing_measure, name: str, expr: str):
    # If measures are Pydantic objects, keep the same type.
    if isinstance(existing_measure, dict) or existing_measure is None:
        return {"name": name, "expr": expr}
    return type(existing_measure)(name=name, expr=expr)


def normalize_query_spec(spec, coltypes: dict[str, str]):
    """
    Prevent invalid aggregations like AVG(text_col).
    If a measure tries to aggregate a non-numeric column, we treat that column as a dimension:
      - add to select if missing
      - add to groupby if missing
      - drop the invalid measure
    If all measures are dropped, fall back to COUNT(*).
    """
    spec.select = spec.select or []
    spec.groupby = spec.groupby or []
    spec.measures = spec.measures or []

    def is_numeric_col(col: str) -> bool:
        return _is_numeric_type(coltypes.get(col, ""))

    fixed_measures = []
    for m in spec.measures:
        _, expr = _get_measure_name_expr(m)
        expr_u = (expr or "").upper().replace(" ", "")

        invalid = False

        # Only handle common numeric aggregations that break on text
        agg_fns = ["AVG(", "SUM(", "MEDIAN(", "STDDEV(", "STDDEV_SAMP(", "STDDEV_POP("]
        for fn in agg_fns:
            if expr_u.startswith(fn) and expr_u.endswith(")"):
                inside = (expr or "")[(expr or "").find("(") + 1:(expr or "").rfind(")")].strip()
                inside = inside.strip('"').strip("'")

                if inside and not is_numeric_col(inside):
                    # move to dimension
                    if inside not in spec.select:
                        spec.select.append(inside)
                    if inside not in spec.groupby:
                        spec.groupby.append(inside)
                    invalid = True
                break

        if not invalid:
            fixed_measures.append(m)

    # If we removed all measures, fallback to COUNT(*)
    if len(fixed_measures) == 0:
        fixed_measures = [
            _make_measure_like(
                spec.measures[0] if spec.measures else None,
                "count_rows",
                "COUNT(*)",
            )
        ]

    spec.measures = fixed_measures
    return spec


def _quote_ident(name: str) -> str:
    return f'"{name}"'


async def _ensure_parquet_local(user_id: str, dataset_id: str) -> Path:
    """Ensure the dataset parquet exists locally.

    Source of truth is Supabase Storage (datasets.parquet_ref). We download on-demand
    using the service role key (server-side) and cache under DATA_DIR.
    """
    p = Path(settings.data_dir) / "datasets" / user_id / dataset_id / "data.parquet"
    if p.exists():
        return p

    row = await registry.fetchrow(
        "SELECT parquet_ref FROM datasets WHERE dataset_id = $1::uuid AND user_id=$2",
        dataset_id, user_id,
    )
    if not row or not row.get("parquet_ref"):
        raise FileNotFoundError("Parquet artifact not found. Dataset parquet_ref is missing (still building?).")

    from app.services.storage_supabase import SupabaseStorage
    storage = SupabaseStorage()

    p.parent.mkdir(parents=True, exist_ok=True)
    file_bytes = await storage.download(row["parquet_ref"])
    p.write_bytes(file_bytes)

    return p

def _apply_xtransform_shortcut(spec: QuerySpec) -> QuerySpec:
    """
    Expand QuerySpec.xTransform into concrete transformer PipelineSteps.

    Long-term contract:
    - Always create a stable KEY column for grouping & sorting.
    - Optionally create LABEL columns for display:
        <base>_name, <base>_name_year, <base>_iso
      controlled by xTransform.format in {"name","name_year","iso","all"}.

    Uses existing transformer ops (NO missing-op errors):
      - date_trunc
      - date_part
      - format_datetime
      - add_conditional_column
      - duplicate_column
      - add_constant_column
      - text_merge

    Also rewrites select/groupby/order_by to use the KEY column so the backend
    actually groups/sorts by the bucketed field (not the raw timestamp).
    """
    xt = getattr(spec, "x_transform", None)
    xf = getattr(spec, "x_field", None)

    if not xt:
        return spec

    # Default xField to the first groupby if missing (common router pattern)
    if not xf:
        if spec.groupby:
            xf = spec.groupby[0]
        else:
            raise ValueError("xTransform provided but xField/groupby is missing")

    unit = (getattr(xt, "type", None) or "").lower().strip()
    fmt = (getattr(xt, "format", None) or "").lower().strip()

    # Base output name: either explicit "as" or <xField>_<unit>
    base = getattr(xt, "as_name", None) or f"{xf}_{unit}"
    key_col = base

    # Label columns
    name_col = f"{base}_name"
    name_year_col = f"{base}_name_year"
    iso_col = f"{base}_iso"

    want_all = (fmt == "all")
    want_name = want_all or (fmt == "name")
    want_name_year = want_all or (fmt == "name_year")
    want_iso = want_all or (fmt == "iso")

    steps = list(spec.transforms or [])

    # ---------------------------------------------------------------------
    # 1) KEY column (stable grouping/sorting)
    # ---------------------------------------------------------------------
    if unit in ("weekday", "weekday_weekend"):
        # DuckDB: EXTRACT(dow FROM ts) -> 0=Sunday .. 6=Saturday
        steps.append(PipelineStep(op="date_part", args={"column": xf, "part": "dow", "as": key_col}))
    else:
        # Default: time bucketing
        steps.append(PipelineStep(op="date_trunc", args={"column": xf, "unit": unit, "as": key_col}))

    # ---------------------------------------------------------------------
    # 2) LABEL columns (optional)
    # ---------------------------------------------------------------------
    if want_name or want_name_year or want_iso:
        if unit == "month":
            if want_name:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%B", "as": name_col}))
            if want_name_year:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%b %Y", "as": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%Y-%m", "as": iso_col}))

        elif unit == "hour":
            if want_name:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%I %p", "as": name_col}))  # "01 PM"
            if want_name_year:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%b %d %Y %I %p", "as": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%Y-%m-%d %H:00", "as": iso_col}))

        elif unit == "day":
            if want_name:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%b %d", "as": name_col}))
            if want_name_year:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%b %d %Y", "as": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%Y-%m-%d", "as": iso_col}))

        elif unit == "week":
            if want_name:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "Wk of %b %d", "as": name_col}))
            if want_name_year:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "Wk of %b %d %Y", "as": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%Y-%m-%d", "as": iso_col}))

        elif unit == "year":
            # Year is straightforward; reuse via duplicate_column when possible
            steps.append(PipelineStep(op="format_datetime", args={"column": key_col, "format": "%Y", "as": name_col}))
            if want_name_year:
                steps.append(PipelineStep(op="duplicate_column", args={"source": name_col, "name": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="duplicate_column", args={"source": name_col, "name": iso_col}))

        elif unit == "quarter":
            # Avoid relying on %q (not always supported); build labels from date parts.
            y_tmp = f"{base}__year"
            q_tmp = f"{base}__quarter"
            q_lbl = f"{base}__q_lbl"
            q_const = f"{base}__Q"

            steps.append(PipelineStep(op="date_part", args={"column": key_col, "part": "year", "as": y_tmp}))
            steps.append(PipelineStep(op="date_part", args={"column": key_col, "part": "quarter", "as": q_tmp}))

            steps.append(PipelineStep(op="add_constant_column", args={"name": q_const, "value": "Q"}))
            steps.append(PipelineStep(op="text_merge", args={"columns_in": [q_const, q_tmp], "sep": "", "name": q_lbl}))

            if want_name:
                steps.append(PipelineStep(op="duplicate_column", args={"source": q_lbl, "name": name_col}))
            if want_name_year:
                steps.append(PipelineStep(op="text_merge", args={"columns_in": [q_lbl, y_tmp], "sep": " ", "name": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="text_merge", args={"columns_in": [y_tmp, q_tmp], "sep": "-Q", "name": iso_col}))

        elif unit == "weekday":
            # KEY is numeric DOW; label from original timestamp for correct locale names
            if want_name:
                steps.append(PipelineStep(op="format_datetime", args={"column": xf, "format": "%A", "as": name_col}))
            if want_name_year:
                steps.append(PipelineStep(op="format_datetime", args={"column": xf, "format": "%a", "as": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="duplicate_column", args={"source": key_col, "name": iso_col}))

        elif unit == "weekday_weekend":
            # Weekend vs Weekday derived from numeric DOW (0,6)
            steps.append(PipelineStep(op="add_conditional_column", args={
                "name": name_col,
                "conditions": [
                    {"when": f"{_quote_ident(key_col)} IN (0, 6)", "then": "Weekend"},
                ],
                "else": "Weekday",
            }))
            if want_name_year:
                steps.append(PipelineStep(op="duplicate_column", args={"source": name_col, "name": name_year_col}))
            if want_iso:
                steps.append(PipelineStep(op="duplicate_column", args={"source": name_col, "name": iso_col}))

        # If unit is unknown, we still keep KEY (date_trunc) and skip labels safely.

    # Apply steps
    spec.transforms = steps

    # ---------------------------------------------------------------------
    # 3) Rewrite select/groupby/order_by to use KEY (stable grouping)
    # ---------------------------------------------------------------------
    def _rewrite(cols: list[str]) -> list[str]:
        return [key_col if c == xf else c for c in (cols or [])]

    spec.select = _rewrite(spec.select)
    spec.groupby = _rewrite(spec.groupby)

    if spec.order_by:
        for o in spec.order_by:
            if o.col == xf:
                o.col = key_col

    # Ensure labels are returned when requested (but not grouped)
    if fmt in ("all", "name", "name_year", "iso"):
        to_add = []
        if want_name:
            to_add.append(name_col)
        if want_name_year:
            to_add.append(name_year_col)
        if want_iso:
            to_add.append(iso_col)

        existing = set(spec.select or [])
        spec.select = list(spec.select or [])
        for c in to_add:
            if c not in existing and c not in (spec.groupby or []):
                spec.select.append(c)
                existing.add(c)

    return spec



def build_query_sql(view: str, spec: QuerySpec) -> str:
    select_parts = []
    selected_cols = set()

    # 1️⃣ Always include GROUP BY dimensions in SELECT (chart-safe)
    for c in (spec.groupby or []):
        if c not in selected_cols:
            select_parts.append(_quote_ident(c))
            selected_cols.add(c)

    # 2️⃣ Include explicitly requested SELECT columns (avoid duplicates)
    for c in (spec.select or []):
        if c not in selected_cols:
            select_parts.append(_quote_ident(c))
            selected_cols.add(c)

    # 3️⃣ Measures
    for m in (spec.measures or []):
        select_parts.append(f"({m.expr}) AS {_quote_ident(m.name)}")

    if not select_parts:
        select_parts = ["*"]

    sql = f"SELECT {', '.join(select_parts)} FROM {view}"

    # 4️⃣ WHERE filters
    if spec.filters:
        clauses = []
        for f in spec.filters:
            op = (f.op or "").strip().upper()
            if op not in ALLOWED_FILTER_OPS:
                raise ValueError(f"Unsupported filter operator: {f.op}")

            col = _quote_ident(f.col)

            if op in ("IN", "NOT IN"):
                if not isinstance(f.value, (list, tuple)) or not f.value:
                    raise ValueError("IN / NOT IN requires a non-empty list")
                vals = []
                for v in f.value:
                    if isinstance(v, (int, float)):
                        vals.append(str(v))
                    elif v is None:
                        vals.append("NULL")
                    else:
                        s = str(v).replace("'", "''")
                        vals.append(f"'{s}'")
                clauses.append(f"{col} {op} ({', '.join(vals)})")

            elif op in ("IS", "IS NOT"):
                if f.value is None:
                    rhs = "NULL"
                else:
                    rhs = str(f.value).upper()
                    if rhs not in ("NULL", "TRUE", "FALSE"):
                        s = str(f.value).replace("'", "''")
                        rhs = f"'{s}'"
                clauses.append(f"{col} {op} {rhs}")

            else:
                if isinstance(f.value, (int, float)):
                    val = str(f.value)
                elif f.value is None:
                    val = "NULL"
                else:
                    s = str(f.value).replace("'", "''")
                    val = f"'{s}'"
                clauses.append(f"{col} {op} {val}")

        sql += " WHERE " + " AND ".join(clauses)

    # 5️⃣ GROUP BY (must match SELECT dimensions exactly)
    if spec.groupby:
        gb = ", ".join(_quote_ident(c) for c in spec.groupby)
        sql += f" GROUP BY {gb}"

    # 6️⃣ ORDER BY (expects columns already rewritten by xTransform shortcut)
    if spec.order_by:
        ob = ", ".join(
            f"{_quote_ident(o.col)} {o.dir.upper()}"
            for o in spec.order_by
        )
        sql += f" ORDER BY {ob}"

    # 7️⃣ LIMIT (safety-clamped earlier)
    sql += f" LIMIT {int(spec.limit)}"

    return sql



async def run_query(user_id: str, dataset_id: str, spec: QuerySpec) -> Dict[str, Any]:
    # Safety clamp: never stream huge raw datasets into JSON.
    # For large result sets, use /query/export instead.
    max_rows = int(__import__("os").getenv("MAX_QUERY_ROWS", "10000"))
    if spec.limit is None or spec.limit <= 0:
        spec.limit = min(1000, max_rows)
    if spec.limit > max_rows:
        spec.limit = max_rows

    parquet_local = await _ensure_parquet_local(user_id, dataset_id)

    eng = DuckDBEngine(user_id)
    con = eng.connect()
    try:
        # 1) Register parquet as a DuckDB view
        view = eng.register_parquet(con, dataset_id, parquet_local)

        # 2) Expand xTransform shortcut into real transformer step + rewrite groupby/select/order_by
        spec = _apply_xtransform_shortcut(spec)

        # 3) If transforms exist (explicit or created by xTransform), compile pipeline view
        if spec.transforms:
            # NOTE: if your ensure_pipeline_view signature differs, adjust here
            view = ensure_pipeline_view(con, dataset_id, view, spec.transforms)

        # 4) Inspect types AFTER transforms
        coltypes = _duckdb_col_types(con, view)

        # 5) Normalize spec (avoid AVG(text), etc.)
        spec = normalize_query_spec(spec, coltypes)

        # 6) Build & execute query
        sql = build_query_sql(view, spec)
        df = con.execute(sql).fetchdf()

        return {
            "columns": list(df.columns),
            "data": json.loads(df.to_json(orient="records", date_format="iso")),
            "row_count": len(df),
        }

    finally:
        con.close()


async def export_query(user_id: str, dataset_id: str, spec: QuerySpec, fmt: str = "parquet") -> Dict[str, Any]:
    """Run a query in DuckDB and export results to a local file + Supabase Storage.

    Use this for very large result sets (e.g., 1M+ rows) instead of returning JSON.
    """
    fmt = (fmt or "parquet").lower()
    if fmt not in ("parquet", "csv"):
        raise ValueError("fmt must be parquet or csv")

    parquet_local = await _ensure_parquet_local(user_id, dataset_id)

    eng = DuckDBEngine(user_id)
    con = eng.connect()
    try:
        view = eng.register_parquet(con, dataset_id, parquet_local)

        # ✅ Expand xTransform shortcut into real transformer step + rewrite select/groupby/order_by
        spec = _apply_xtransform_shortcut(spec)

        # ✅ Apply pipeline transforms (including labels via format_datetime/add_conditional_column/etc.)
        if spec.transforms:
            view = ensure_pipeline_view(con, dataset_id, view, spec.transforms)

        # ✅ Normalize AFTER transforms (types may change)
        coltypes = _duckdb_col_types(con, view)
        spec = normalize_query_spec(spec, coltypes)

        sql = build_query_sql(view, spec)

        export_dir = Path(settings.data_dir) / "exports" / user_id / dataset_id
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        local_out = export_dir / f"query_{ts}.{fmt}"

        if fmt == "parquet":
            con.execute(f"COPY ({sql}) TO '{local_out.as_posix()}' (FORMAT PARQUET)")
            content_type = "application/octet-stream"
        else:
            con.execute(f"COPY ({sql}) TO '{local_out.as_posix()}' (HEADER, DELIMITER ',')")
            content_type = "text/csv"

        # row count
        row_count = con.execute(f"SELECT COUNT(*) AS c FROM ({sql}) t").fetchone()[0]

    finally:
        con.close()

    from app.services.storage_supabase import SupabaseStorage
    storage = SupabaseStorage()
    remote_path = f"exports/{user_id}/{dataset_id}/{local_out.name}"
    await storage.upload_file(local_out, remote_path, content_type)

    return {"remote_path": remote_path, "row_count": int(row_count), "format": fmt}



async def run_query_operation(user_id: str, dataset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility path for the datalab-chat edge function.

    The edge function currently sends a tool-call payload shaped like:
      { operation: "aggregate"|"filter"|"distinct"|"crosstab"|"describe"|"percentile", ... }

    We translate that payload into a QuerySpec and execute via DuckDB.
    """
    op = str(payload.get("operation") or "").strip().lower()
    if not op:
        raise ValueError("Missing operation")

    # Map common shapes
    if op in ("aggregate", "filter"):
        metrics = payload.get("metrics") or payload.get("measures") or []
        measures = []
        for m in metrics:
            # expected { column, func, alias? }
            col = m.get("column")
            func = str(m.get("func") or "").lower()
            alias = m.get("alias") or f"{func}_{col}"
            
            # Use conservative allowlist, map to DuckDB SQL
            func_map = {
                # Existing
                "count": "COUNT",
                "sum": "SUM",
                "mean": "AVG",
                "avg": "AVG",
                "min": "MIN",
                "max": "MAX",
                "std": "STDDEV_SAMP",
                "nunique": "__NUNIQUE__",

                # ADD THESE (DuckDB-native)
                "median": "MEDIAN",
                "mode": "MODE",
                "variance": "VAR_SAMP",   # DuckDB supports VAR_SAMP / VAR_POP
                "var": "VAR_SAMP",
                "var_samp": "VAR_SAMP",
                "var_pop": "VAR_POP",
            }

            # Percentiles: accept keys like percentile_25, percentile_75, percentile_90
            if func.startswith("percentile_"):
                try:
                    p = int(func.split("_", 1)[1])
                except Exception:
                    raise ValueError(f"Unsupported percentile format: {func}")
                if p < 0 or p > 100:
                    raise ValueError(f"Percentile out of range (0-100): {p}")
                q = p / 100.0
                expr = f"QUANTILE_CONT({_quote_ident(col)}, {q})"

            elif func in func_map:
                if func == "nunique":
                    expr = f"COUNT(DISTINCT {_quote_ident(col)})"
                else:
                    expr = f"{func_map[func]}({_quote_ident(col)})"

            else:
                raise ValueError(f"Unsupported agg func: {func}")

            measures.append({"name": alias, "expr": expr})


        # Build QuerySpec using the pydantic model types (pydantic will coerce dicts -> Measure)
        spec = QuerySpec(
            select=list(payload.get("select") or []),
            measures=[{"name": m["name"], "expr": m["expr"]} for m in measures],
            groupby=list(payload.get("group_by") or payload.get("groupby") or []),
            filters=[
                {"col": f.get("column") or f.get("col"), "op": f.get("op"), "value": f.get("value")}
                for f in (payload.get("filters") or payload.get("filter") or [])
            ],
            order_by=[],
            limit=int(payload.get("limit") or 1000),
        )
        return await run_query(user_id, dataset_id, spec)

    if op == "distinct":
        cols = payload.get("columns") or payload.get("select") or payload.get("group_by")
        if not cols:
            raise ValueError("distinct requires columns")
        spec = QuerySpec(select=list(cols), measures=[], groupby=[], filters=[], order_by=[], limit=int(payload.get("limit") or 1000))
        # Distinct implemented by SQL builder when select only; we emulate by setting groupby=select
        spec.groupby = list(cols)
        return await run_query(user_id, dataset_id, spec)

    if op == "crosstab":
        gb = payload.get("group_by") or []
        if len(gb) != 2:
            raise ValueError("crosstab requires exactly 2 group_by columns")
        # Basic crosstab count
        spec = QuerySpec(
            select=list(gb),
            measures=[{"name": "count", "expr": "COUNT(*)"}],
            groupby=list(gb),
            filters=[
                {"col": f.get("column") or f.get("col"), "op": f.get("op"), "value": f.get("value")}
                for f in (payload.get("filters") or [])
            ],
            order_by=[],
            limit=int(payload.get("limit") or 10000),
        )
        return await run_query(user_id, dataset_id, spec)

    if op in ("describe", "percentile"):
        # Route to stats endpoint behavior using QuerySpec for raw data pull would be expensive.
        # For now, compute via DuckDB aggregate SQL.
        parquet_local = await _ensure_parquet_local(user_id, dataset_id)
        eng = DuckDBEngine(user_id)
        con = eng.connect()
        view = eng.register_parquet(con, dataset_id, parquet_local)

        cols = payload.get("columns") or payload.get("select") or []
        if not cols:
            # If no columns specified, pick numeric columns from schema_json if available
            row = await registry.fetchrow(
                "SELECT schema_json FROM datasets WHERE dataset_id = $1::uuid AND user_id=$2",
                dataset_id, user_id,
            )
            schema = row["schema_json"] if row and row.get("schema_json") else []
            cols = [
                c.get("name")
                for c in schema
                if str(c.get("type") or "").lower() in ("integer", "bigint", "double", "real", "float")
                and c.get("name")
            ]

        if not cols:
            raise ValueError("No columns provided for describe")

        if op == "describe":
            results = {}
            for c in cols:
                q = (
                    f"SELECT COUNT({_quote_ident(c)}) AS count, "
                    f"AVG({_quote_ident(c)}) AS mean, "
                    f"STDDEV_SAMP({_quote_ident(c)}) AS std, "
                    f"MIN({_quote_ident(c)}) AS min, "
                    f"MAX({_quote_ident(c)}) AS max "
                    f"FROM {view}"
                )
                r = con.execute(q).fetchone()
                results[c] = {
                    "count": int(r[0]),
                    "mean": float(r[1]) if r[1] is not None else None,
                    "std": float(r[2]) if r[2] is not None else None,
                    "min": float(r[3]) if r[3] is not None else None,
                    "max": float(r[4]) if r[4] is not None else None,
                }
            con.close()
            return {
                "columns": ["column", "count", "mean", "std", "min", "max"],
                "data": [{"column": k, **v} for k, v in results.items()],
                "row_count": len(results),
            }

        # percentile
        pct = float(payload.get("percentile") or payload.get("p") or 0.5)
        if not (0.0 < pct < 1.0):
            raise ValueError("percentile must be between 0 and 1")

        data = []
        for c in cols:
            q = f"SELECT quantile_cont({_quote_ident(c)}, {pct}) AS p FROM {view}"
            r = con.execute(q).fetchone()
            data.append({"column": c, "percentile": pct, "value": float(r[0]) if r[0] is not None else None})
        con.close()
        return {"columns": ["column", "percentile", "value"], "data": data, "row_count": len(data)}

    raise ValueError(f"Unsupported operation: {op}")

