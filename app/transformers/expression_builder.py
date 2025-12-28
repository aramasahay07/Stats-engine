from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Union


@dataclass(frozen=True)
class ExprError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def q_ident(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ExprError("Invalid column name")
    if '"' in name:
        raise ExprError(f"Invalid identifier: {name}")
    return f'"{name}"'


def q_lit(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).replace("'", "''")
    return f"'{s}'"


_ALLOWED_BINOPS = {"+", "-", "*", "/", "%"}
_ALLOWED_CMPOPS = {"=", "!=", ">", ">=", "<", "<=", "like", "ilike"}
_ALLOWED_LOGICAL = {"and", "or"}
_ALLOWED_UNARY = {"not", "-"}

# DuckDB-safe function whitelist (extend as needed)
_ALLOWED_FUNCS = {
    # numeric
    "abs": 1,
    "round": (1, 2),
    "coalesce": (2, 10),
    "sqrt": 1,
    "log": (1, 2),
    # text
    "lower": 1,
    "upper": 1,
    "trim": 1,
    "replace": 3,
    "concat": (2, 20),
    "length": 1,
    # datetime
    "strftime": 2,  # (timestamp, format)
    "date_trunc": 2,  # (unit, timestamp)
}


def _arity_ok(expected, got: int) -> bool:
    if isinstance(expected, int):
        return got == expected
    lo, hi = expected
    return lo <= got <= hi


def to_sql(expr: Any, *, allowed_columns: Optional[Set[str]] = None) -> str:
    """Convert a JSON AST expression into a SAFE SQL expression string.

    Supported node shapes:
      - {"col": "MyColumn"}
      - {"val": <literal>}
      - {"op": "+", "left": <expr>, "right": <expr>}
      - {"op": ">=", "left": <expr>, "right": <expr>}
      - {"op": "and", "args": [<expr>, <expr>, ...]}
      - {"op": "not", "arg": <expr>}
      - {"op": "in", "left": <expr>, "right": {"val": [..]}}
      - {"op": "between", "value": <expr>, "low": <expr>, "high": <expr>, "inclusive": true}
      - {"op": "func", "name": "strftime", "args": [..]}
      - shorthand func: {"op":"strftime","args":[..]}

    This intentionally does NOT accept raw SQL strings.
    """
    if isinstance(expr, dict):
        if "col" in expr:
            col = expr["col"]
            if allowed_columns is not None and col not in allowed_columns:
                raise ExprError(f"Unknown column: {col}")
            return q_ident(col)
        if "val" in expr:
            v = expr["val"]
            # IN expects list literal sometimes; handled below
            if isinstance(v, list):
                return "(" + ", ".join(q_lit(x) for x in v) + ")"
            return q_lit(v)

        op = expr.get("op")
        if not op:
            raise ExprError("Expression missing op")

        # logical n-ary
        if op in _ALLOWED_LOGICAL:
            args = expr.get("args")
            if not isinstance(args, list) or len(args) < 2:
                raise ExprError(f"{op} requires args list")
            parts = [to_sql(a, allowed_columns=allowed_columns) for a in args]
            joiner = " AND " if op == "and" else " OR "
            return "(" + joiner.join(parts) + ")"

        # unary
        if op in _ALLOWED_UNARY:
            arg = expr.get("arg")
            if arg is None:
                raise ExprError(f"{op} requires arg")
            a = to_sql(arg, allowed_columns=allowed_columns)
            if op == "not":
                return f"(NOT {a})"
            return f"(-{a})"

        # between
        if op == "between":
            v = to_sql(expr.get("value"), allowed_columns=allowed_columns)
            low = to_sql(expr.get("low"), allowed_columns=allowed_columns)
            high = to_sql(expr.get("high"), allowed_columns=allowed_columns)
            inclusive = bool(expr.get("inclusive", True))
            if inclusive:
                return f"({v} BETWEEN {low} AND {high})"
            return f"({v} > {low} AND {v} < {high})"

        # IN
        if op == "in":
            left = to_sql(expr.get("left"), allowed_columns=allowed_columns)
            right = expr.get("right")
            if not isinstance(right, dict) or "val" not in right or not isinstance(right["val"], list):
                raise ExprError("in requires right={val:[...]} ")
            lst = ", ".join(q_lit(x) for x in right["val"])
            return f"({left} IN ({lst}))"

        # binary ops
        if op in _ALLOWED_BINOPS or op in _ALLOWED_CMPOPS:
            left = to_sql(expr.get("left"), allowed_columns=allowed_columns)
            right = to_sql(expr.get("right"), allowed_columns=allowed_columns)
            sop = op.upper() if op in {"like", "ilike"} else op
            return f"({left} {sop} {right})"

        # function call
        if op == "func":
            name = (expr.get("name") or "").lower()
            args = expr.get("args")
        else:
            # shorthand: {op:"strftime", args:[...]}
            name = str(op).lower()
            args = expr.get("args")

        if name in _ALLOWED_FUNCS:
            if not isinstance(args, list):
                raise ExprError(f"{name} requires args list")
            expected = _ALLOWED_FUNCS[name]
            if not _arity_ok(expected, len(args)):
                raise ExprError(f"{name} wrong number of args")
            sql_args = [to_sql(a, allowed_columns=allowed_columns) for a in args]
            # DuckDB date_trunc expects unit as string literal, so allow {val:"month"}
            return f"{name.upper()}({', '.join(sql_args)})" if name not in {"strftime", "date_trunc"} else f"{name}({', '.join(sql_args)})"

        raise ExprError(f"Unsupported op/function: {op}")

    raise ExprError("Expression must be an object")
