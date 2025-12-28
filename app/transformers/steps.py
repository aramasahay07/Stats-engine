"""
Enhanced Transformers Library - 100+ Data Transformation Operations

Categories:
1. Table Shaping (7 transformers)
2. Row Operations (10 transformers)
3. Filtering (4 transformers: 2 legacy + 2 safe)
4. Computed Columns (4 transformers: 2 legacy + 2 safe)
5. Data Cleaning (6 transformers)
6. Text Operations (12 transformers)
7. Datetime Operations (10 transformers)
8. Statistical Operations (20 transformers)
9. Window Functions (12 transformers)
10. Aggregation (8 transformers)
11. Data Quality (7 transformers)

Total: 100 transformers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.transformers.base import Transformer
from app.transformers.expression_builder import to_sql, q_ident, q_lit


def wrap(prior_sql: str) -> str:
    """Wrap SQL in parentheses with alias for use in subqueries."""
    return f"({prior_sql}) AS _t"


# ============================================================================
# 1. TABLE SHAPING (7 transformers)
# ============================================================================

class SelectColumns(Transformer):
    """Select specific columns to keep."""
    op = "select_columns"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        cols = args.get("columns") or []
        if not cols:
            return prior_sql
        quoted = ", ".join(q_ident(c) for c in cols)
        return f"SELECT {quoted} FROM {wrap(prior_sql)}"


class DropColumns(Transformer):
    """Drop columns by specifying which to keep."""
    op = "drop_columns"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        keep = args.get("keep") or []
        if not keep:
            return prior_sql
        quoted = ", ".join(q_ident(c) for c in keep)
        return f"SELECT {quoted} FROM {wrap(prior_sql)}"


class RenameColumns(Transformer):
    """Rename columns using a mapping."""
    op = "rename_columns"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        mapping = args.get("mapping") or {}
        cols = args.get("select") or args.get("columns") or []
        if not cols:
            return prior_sql
        exprs = []
        for c in cols:
            new = mapping.get(c, c)
            if new == c:
                exprs.append(q_ident(c))
            else:
                exprs.append(f"{q_ident(c)} AS {q_ident(new)}")
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class ReorderColumns(Transformer):
    """Reorder columns in specific order."""
    op = "reorder_columns"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        cols = args.get("columns") or []
        if not cols:
            return prior_sql
        quoted = ", ".join(q_ident(c) for c in cols)
        return f"SELECT {quoted} FROM {wrap(prior_sql)}"


class AddConstantColumn(Transformer):
    """Add a column with constant value."""
    op = "add_constant_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        name = args.get("name")
        value = args.get("value")
        if not name:
            return prior_sql
        return f"SELECT *, {q_lit(value)} AS {q_ident(name)} FROM {wrap(prior_sql)}"


class DuplicateColumn(Transformer):
    """Duplicate a column with new name."""
    op = "duplicate_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        source = args.get("source")
        name = args.get("name")
        if not source or not name:
            return prior_sql
        return f"SELECT *, {q_ident(source)} AS {q_ident(name)} FROM {wrap(prior_sql)}"


class MoveColumn(Transformer):
    """Move column to specific position (reselect all with new order)."""
    op = "move_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        position = args.get("position", "last")  # first, last, after_column
        columns = args.get("all_columns") or []  # Need full column list
        
        if not column or not columns or column not in columns:
            return prior_sql
        
        # Remove column from current position
        cols = [c for c in columns if c != column]
        
        # Insert at new position
        if position == "first":
            cols.insert(0, column)
        elif position == "last":
            cols.append(column)
        elif isinstance(position, dict) and "after" in position:
            after_col = position["after"]
            if after_col in cols:
                idx = cols.index(after_col) + 1
                cols.insert(idx, column)
            else:
                cols.append(column)
        
        quoted = ", ".join(q_ident(c) for c in cols)
        return f"SELECT {quoted} FROM {wrap(prior_sql)}"


# ============================================================================
# 2. ROW OPERATIONS (10 transformers)
# ============================================================================

class SortRows(Transformer):
    """Sort rows by one or more columns."""
    op = "sort_rows"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        sort = args.get("sort") or []
        if not sort:
            return prior_sql
        parts = []
        for s in sort:
            col = s.get("column")
            if not col:
                continue
            direction = (s.get("direction") or "asc").upper()
            nulls = s.get("nulls")
            piece = f"{q_ident(col)} {direction}"
            if nulls:
                piece += f" NULLS {str(nulls).upper()}"
            parts.append(piece)
        if not parts:
            return prior_sql
        return f"SELECT * FROM {wrap(prior_sql)} ORDER BY {', '.join(parts)}"


class LimitRows(Transformer):
    """Limit to first N rows."""
    op = "limit_rows"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        n = args.get("n")
        if n is None:
            return prior_sql
        return f"SELECT * FROM {wrap(prior_sql)} LIMIT {int(n)}"


class OffsetRows(Transformer):
    """Skip first N rows."""
    op = "offset_rows"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        n = args.get("n")
        if n is None:
            return prior_sql
        return f"SELECT * FROM {wrap(prior_sql)} OFFSET {int(n)}"


class SampleRows(Transformer):
    """Random sample of rows."""
    op = "sample_rows"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        n = args.get("n")
        frac = args.get("frac")
        if n is None and frac is None:
            return prior_sql
        if n is not None:
            return f"SELECT * FROM {wrap(prior_sql)} ORDER BY random() LIMIT {int(n)}"
        pct = float(frac) * 100.0
        return f"SELECT * FROM {wrap(prior_sql)} USING SAMPLE {pct} PERCENT (bernoulli)"


class DistinctRows(Transformer):
    """Select distinct rows."""
    op = "distinct_rows"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        cols = args.get("columns") or []
        if cols:
            quoted = ", ".join(q_ident(c) for c in cols)
            return f"SELECT DISTINCT {quoted} FROM {wrap(prior_sql)}"
        return f"SELECT DISTINCT * FROM {wrap(prior_sql)}"


class RemoveDuplicates(Transformer):
    """Remove duplicates keeping first occurrence."""
    op = "remove_duplicates"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        by = args.get("by") or []
        order_by = args.get("order_by") or []
        if not by:
            return f"SELECT DISTINCT * FROM {wrap(prior_sql)}"
        part = ", ".join(q_ident(c) for c in by)
        if order_by:
            ob = ", ".join(f"{q_ident(o['column'])} {(o.get('direction') or 'asc').upper()}" for o in order_by if o.get('column'))
        else:
            ob = part
        return (
            "SELECT * EXCLUDE(_rn) FROM ("
            f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {part} ORDER BY {ob}) AS _rn "
            f"FROM {wrap(prior_sql)}"
            ") WHERE _rn = 1"
        )


class AddIndexColumn(Transformer):
    """Add row number index column."""
    op = "add_index_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        name = args.get("name") or "index"
        start = int(args.get("start", 1))
        return (
            f"SELECT *, ({start} - 1) + ROW_NUMBER() OVER () AS {q_ident(name)} "
            f"FROM {wrap(prior_sql)}"
        )


class TopN(Transformer):
    """Select top N rows by column value."""
    op = "top_n"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        n = args.get("n", 10)
        column = args.get("column")
        direction = args.get("direction", "desc").upper()
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT * FROM {wrap(prior_sql)}
            ORDER BY {q_ident(column)} {direction}
            LIMIT {int(n)}
        """


class BottomN(Transformer):
    """Select bottom N rows by column value."""
    op = "bottom_n"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        n = args.get("n", 10)
        column = args.get("column")
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT * FROM {wrap(prior_sql)}
            ORDER BY {q_ident(column)} ASC
            LIMIT {int(n)}
        """


class RandomSample(Transformer):
    """Random sample with seed for reproducibility."""
    op = "random_sample"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        n = args.get("n")
        seed = args.get("seed")
        
        if n is None:
            return prior_sql
        
        if seed is not None:
            return f"SELECT setseed({float(seed)}), * FROM {wrap(prior_sql)} ORDER BY random() LIMIT {int(n)}"
        else:
            return f"SELECT * FROM {wrap(prior_sql)} ORDER BY random() LIMIT {int(n)}"


# ============================================================================
# 3. FILTERING (4 transformers: 2 legacy + 2 safe)
# ============================================================================

class FilterRows(Transformer):
    """Filter rows using raw SQL WHERE clause (LEGACY - use filter_rows_safe for user input)."""
    op = "filter_rows"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        where = args.get("where")
        if not where:
            return prior_sql
        return f"SELECT * FROM {wrap(prior_sql)} WHERE {where}"


class FilterRowsSafe(Transformer):
    """Filter rows using safe expression tree (prevents SQL injection)."""
    op = "filter_rows_safe"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        expr = args.get("expr")
        allowed = set(args.get("allowed_columns") or []) or None
        if not expr:
            return prior_sql
        where_sql = to_sql(expr, allowed_columns=allowed)
        return f"SELECT * FROM {wrap(prior_sql)} WHERE {where_sql}"


class FilterTopPercent(Transformer):
    """Filter to top N percent by column value."""
    op = "filter_top_percent"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        percent = args.get("percent", 10)
        column = args.get("column")
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT * FROM (
                SELECT *, PERCENT_RANK() OVER (ORDER BY {q_ident(column)} DESC) AS _pct_rank
                FROM {wrap(prior_sql)}
            ) WHERE _pct_rank <= {float(percent) / 100.0}
        """


class FilterByRange(Transformer):
    """Filter rows where column is within range."""
    op = "filter_by_range"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        min_val = args.get("min")
        max_val = args.get("max")
        
        if not column:
            return prior_sql
        
        conditions = []
        if min_val is not None:
            conditions.append(f"{q_ident(column)} >= {q_lit(min_val)}")
        if max_val is not None:
            conditions.append(f"{q_ident(column)} <= {q_lit(max_val)}")
        
        if not conditions:
            return prior_sql
        
        where_clause = " AND ".join(conditions)
        return f"SELECT * FROM {wrap(prior_sql)} WHERE {where_clause}"


# ============================================================================
# 4. COMPUTED COLUMNS (4 transformers: 2 legacy + 2 safe)
# ============================================================================

class AddComputedColumn(Transformer):
    """Add computed column using raw SQL expression (LEGACY - use add_computed_safe for user input)."""
    op = "add_computed_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        name = args.get("name")
        expr = args.get("expr")
        if not name or not expr:
            return prior_sql
        return f"SELECT *, ({expr}) AS {q_ident(name)} FROM {wrap(prior_sql)}"


class AddComputedSafe(Transformer):
    """Add computed column using safe expression tree (prevents SQL injection)."""
    op = "add_computed_safe"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        name = args.get("name")
        expr = args.get("expr")
        allowed = set(args.get("allowed_columns") or []) or None
        if not name or not expr:
            return prior_sql
        expr_sql = to_sql(expr, allowed_columns=allowed)
        return f"SELECT *, ({expr_sql}) AS {q_ident(name)} FROM {wrap(prior_sql)}"


class AddConditionalColumn(Transformer):
    """Add column based on CASE WHEN conditions."""
    op = "add_conditional_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        name = args.get("name")
        conditions = args.get("conditions") or []  # [{when: expr, then: value}, ...]
        else_value = args.get("else")
        
        if not name or not conditions:
            return prior_sql
        
        case_parts = ["CASE"]
        for cond in conditions:
            when = cond.get("when")
            then = cond.get("then")
            if when and then is not None:
                case_parts.append(f"WHEN {when} THEN {q_lit(then)}")
        
        if else_value is not None:
            case_parts.append(f"ELSE {q_lit(else_value)}")
        case_parts.append("END")
        
        case_expr = " ".join(case_parts)
        return f"SELECT *, ({case_expr}) AS {q_ident(name)} FROM {wrap(prior_sql)}"


class AddMathColumn(Transformer):
    """Add column with mathematical operation between columns."""
    op = "add_math_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        name = args.get("name")
        operation = args.get("operation")  # add, subtract, multiply, divide, power, modulo
        columns = args.get("columns") or []
        
        if not name or not operation or len(columns) < 2:
            return prior_sql
        
        op_map = {
            "add": "+", "subtract": "-", "multiply": "*", 
            "divide": "/", "power": "**", "modulo": "%"
        }
        
        sql_op = op_map.get(operation)
        if not sql_op:
            return prior_sql
        
        expr = f" {sql_op} ".join(q_ident(c) for c in columns)
        return f"SELECT *, ({expr}) AS {q_ident(name)} FROM {wrap(prior_sql)}"


# ============================================================================
# 5. DATA CLEANING (6 transformers)
# ============================================================================

class DropNulls(Transformer):
    """Drop rows with NULL values in specified columns."""
    op = "drop_nulls"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        if not columns:
            return prior_sql
        cond = " AND ".join([f"{q_ident(c)} IS NOT NULL" for c in columns])
        return f"SELECT * FROM {wrap(prior_sql)} WHERE {cond}"


class FillNulls(Transformer):
    """Fill NULL values with specified values."""
    op = "fill_nulls"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        fills = args.get("fills") or []
        columns = args.get("columns") or []
        
        if not fills or not columns:
            return prior_sql
        
        fill_map = {f["column"]: f["value"] for f in fills if "column" in f and "value" in f}
        exprs = []
        
        for c in columns:
            if c in fill_map:
                exprs.append(f"COALESCE({q_ident(c)}, {q_lit(fill_map[c])}) AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class ReplaceValues(Transformer):
    """Replace specific values in columns."""
    op = "replace_values"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        replacements = args.get("replacements") or []
        columns = args.get("columns") or []
        
        if not replacements or not columns:
            return prior_sql
        
        replace_map = {}
        for r in replacements:
            col = r.get("column")
            if col:
                if col not in replace_map:
                    replace_map[col] = []
                replace_map[col].append((r.get("old"), r.get("new")))
        
        exprs = []
        for c in columns:
            if c in replace_map:
                case_expr = f"CASE {q_ident(c)}"
                for old_val, new_val in replace_map[c]:
                    case_expr += f" WHEN {q_lit(old_val)} THEN {q_lit(new_val)}"
                case_expr += f" ELSE {q_ident(c)} END"
                exprs.append(f"{case_expr} AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class ChangeType(Transformer):
    """Convert column data types."""
    op = "change_type"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        conversions = args.get("conversions") or []
        columns = args.get("columns") or []
        
        if not conversions or not columns:
            return prior_sql
        
        type_map = {c["column"]: c["to_type"] for c in conversions if "column" in c and "to_type" in c}
        exprs = []
        
        for c in columns:
            if c in type_map:
                exprs.append(f"CAST({q_ident(c)} AS {type_map[c]}) AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class Coalesce(Transformer):
    """Return first non-NULL value from list of columns."""
    op = "coalesce"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        name = args.get("name")
        columns = args.get("columns") or []
        
        if not name or not columns:
            return prior_sql
        
        col_list = ", ".join(q_ident(c) for c in columns)
        return f"SELECT *, COALESCE({col_list}) AS {q_ident(name)} FROM {wrap(prior_sql)}"


class CleanWhitespace(Transformer):
    """Remove extra whitespace from text columns."""
    op = "clean_whitespace"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        all_columns = args.get("all_columns") or []
        
        if not columns or not all_columns:
            return prior_sql
        
        exprs = []
        for c in all_columns:
            if c in columns:
                exprs.append(f"TRIM(REGEXP_REPLACE({q_ident(c)}, '\s+', ' ', 'g')) AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


# ============================================================================
# 6. TEXT OPERATIONS (12 transformers)
# ============================================================================

class TextTrim(Transformer):
    """Trim whitespace from text columns."""
    op = "text_trim"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        all_columns = args.get("all_columns") or []
        
        if not columns or not all_columns:
            return prior_sql
        
        exprs = []
        for c in all_columns:
            if c in columns:
                exprs.append(f"TRIM({q_ident(c)}) AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class TextLower(Transformer):
    """Convert text to lowercase."""
    op = "text_lower"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        all_columns = args.get("all_columns") or []
        
        if not columns or not all_columns:
            return prior_sql
        
        exprs = []
        for c in all_columns:
            if c in columns:
                exprs.append(f"LOWER({q_ident(c)}) AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class TextUpper(Transformer):
    """Convert text to uppercase."""
    op = "text_upper"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        all_columns = args.get("all_columns") or []
        
        if not columns or not all_columns:
            return prior_sql
        
        exprs = []
        for c in all_columns:
            if c in columns:
                exprs.append(f"UPPER({q_ident(c)}) AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class TextReplace(Transformer):
    """Find and replace text in columns."""
    op = "text_replace"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        replacements = args.get("replacements") or []
        columns = args.get("columns") or []
        
        if not replacements or not columns:
            return prior_sql
        
        replace_map = {}
        for r in replacements:
            col = r.get("column")
            if col:
                if col not in replace_map:
                    replace_map[col] = []
                replace_map[col].append((r.get("old"), r.get("new", "")))
        
        exprs = []
        for c in columns:
            if c in replace_map:
                expr = q_ident(c)
                for old, new in replace_map[c]:
                    expr = f"REPLACE({expr}, {q_lit(old)}, {q_lit(new)})"
                exprs.append(f"{expr} AS {q_ident(c)}")
            else:
                exprs.append(q_ident(c))
        
        return f"SELECT {', '.join(exprs)} FROM {wrap(prior_sql)}"


class TextSplit(Transformer):
    """Split text column into multiple columns."""
    op = "text_split"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        sep = args.get("sep")
        parts = int(args.get("parts", 2))
        prefix = args.get("prefix") or f"{column}_part"
        
        if not column or sep is None:
            return prior_sql
        
        new_cols = [
            f"split_part({q_ident(column)}, {q_lit(sep)}, {i}) AS {q_ident(f'{prefix}{i}')}"
            for i in range(1, parts + 1)
        ]
        
        return f"SELECT *, {', '.join(new_cols)} FROM {wrap(prior_sql)}"


class TextMerge(Transformer):
    """Concatenate columns into new column."""
    op = "text_merge"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        cols_in = args.get("columns_in") or []
        sep = args.get("sep", "")
        name = args.get("name")
        
        if not cols_in or not name:
            return prior_sql
        
        pieces = []
        for i, c in enumerate(cols_in):
            if i > 0 and sep:
                pieces.append(q_lit(sep))
            pieces.append(f"coalesce(cast({q_ident(c)} as varchar), '')")
        
        expr = " || ".join(pieces) if pieces else "''"
        return f"SELECT *, ({expr}) AS {q_ident(name)} FROM {wrap(prior_sql)}"


class TextLength(Transformer):
    """Add column with text length."""
    op = "text_length"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        name = args.get("name") or f"{column}_length"
        
        if not column:
            return prior_sql
        
        return f"SELECT *, LENGTH({q_ident(column)}) AS {q_ident(name)} FROM {wrap(prior_sql)}"


class TextSubstring(Transformer):
    """Extract substring from text column."""
    op = "text_substring"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        start = args.get("start", 1)
        length = args.get("length")
        name = args.get("name") or f"{column}_sub"
        
        if not column:
            return prior_sql
        
        if length is not None:
            expr = f"SUBSTRING({q_ident(column)}, {int(start)}, {int(length)})"
        else:
            expr = f"SUBSTRING({q_ident(column)}, {int(start)})"
        
        return f"SELECT *, {expr} AS {q_ident(name)} FROM {wrap(prior_sql)}"


class TextPad(Transformer):
    """Pad text to specified length."""
    op = "text_pad"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        length = args.get("length")
        side = args.get("side", "left")  # left or right
        char = args.get("char", " ")
        name = args.get("name") or f"{column}_padded"
        
        if not column or not length:
            return prior_sql
        
        if side == "left":
            expr = f"LPAD({q_ident(column)}, {int(length)}, {q_lit(char)})"
        else:
            expr = f"RPAD({q_ident(column)}, {int(length)}, {q_lit(char)})"
        
        return f"SELECT *, {expr} AS {q_ident(name)} FROM {wrap(prior_sql)}"


class TextContains(Transformer):
    """Add boolean column indicating if text contains substring."""
    op = "text_contains"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        pattern = args.get("pattern")
        name = args.get("name") or f"{column}_contains"
        
        if not column or not pattern:
            return prior_sql
        
        expr = f"({q_ident(column)} LIKE '%' || {q_lit(pattern)} || '%')"
        return f"SELECT *, {expr} AS {q_ident(name)} FROM {wrap(prior_sql)}"


class TextStartsWith(Transformer):
    """Add boolean column indicating if text starts with prefix."""
    op = "text_starts_with"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        prefix = args.get("prefix")
        name = args.get("name") or f"{column}_starts"
        
        if not column or not prefix:
            return prior_sql
        
        expr = f"({q_ident(column)} LIKE {q_lit(prefix)} || '%')"
        return f"SELECT *, {expr} AS {q_ident(name)} FROM {wrap(prior_sql)}"


class TextEndsWith(Transformer):
    """Add boolean column indicating if text ends with suffix."""
    op = "text_ends_with"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        suffix = args.get("suffix")
        name = args.get("name") or f"{column}_ends"
        
        if not column or not suffix:
            return prior_sql
        
        expr = f"({q_ident(column)} LIKE '%' || {q_lit(suffix)})"
        return f"SELECT *, {expr} AS {q_ident(name)} FROM {wrap(prior_sql)}"


# ============================================================================
# 7. DATETIME OPERATIONS (10 transformers)
# ============================================================================

class DateFromText(Transformer):
    """Parse text to timestamp."""
    op = "date_from_text"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        fmt = args.get("format")
        as_name = args.get("as") or column
        
        if not column or not fmt:
            return prior_sql
        
        return f"SELECT *, CAST(strptime({q_ident(column)}, {q_lit(fmt)}) AS TIMESTAMP) AS {q_ident(as_name)} FROM {wrap(prior_sql)}"


class DatePart(Transformer):
    """Extract date part (year, month, day, etc.)."""
    op = "date_part"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        part = (args.get("part") or "").lower()
        as_name = args.get("as") or f"{column}_{part}"
        
        if not column or not part:
            return prior_sql
        
        return f"SELECT *, EXTRACT({part} FROM {q_ident(column)}) AS {q_ident(as_name)} FROM {wrap(prior_sql)}"


class DateTrunc(Transformer):
    """Truncate date to unit (month, day, hour, etc.)."""
    op = "date_trunc"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        unit = args.get("unit")
        as_name = args.get("as") or f"{column}_{unit}"
        
        if not column or not unit:
            return prior_sql
        
        return f"SELECT *, date_trunc({q_lit(unit)}, {q_ident(column)}) AS {q_ident(as_name)} FROM {wrap(prior_sql)}"


class FormatDatetime(Transformer):
    """Format datetime as text."""
    op = "format_datetime"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        fmt = args.get("format")
        as_name = args.get("as") or f"{column}_fmt"
        
        if not column or not fmt:
            return prior_sql
        
        return f"SELECT *, strftime({q_ident(column)}, {q_lit(fmt)}) AS {q_ident(as_name)} FROM {wrap(prior_sql)}"


class DateDiff(Transformer):
    """Calculate difference between two dates."""
    op = "date_diff"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        start_col = args.get("start_column")
        end_col = args.get("end_column")
        unit = args.get("unit", "day")  # day, hour, minute, second
        name = args.get("name") or "date_diff"
        
        if not start_col or not end_col:
            return prior_sql
        
        return f"""
            SELECT *, 
            DATE_DIFF({q_lit(unit)}, {q_ident(start_col)}, {q_ident(end_col)}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class DateAdd(Transformer):
    """Add interval to date."""
    op = "date_add"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        value = args.get("value")
        unit = args.get("unit", "day")
        name = args.get("name") or f"{column}_plus"
        
        if not column or value is None:
            return prior_sql
        
        return f"""
            SELECT *, 
            {q_ident(column)} + INTERVAL {int(value)} {unit} AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class DateSubtract(Transformer):
    """Subtract interval from date."""
    op = "date_subtract"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        value = args.get("value")
        unit = args.get("unit", "day")
        name = args.get("name") or f"{column}_minus"
        
        if not column or value is None:
            return prior_sql
        
        return f"""
            SELECT *, 
            {q_ident(column)} - INTERVAL {int(value)} {unit} AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class AgeCalculation(Transformer):
    """Calculate age from birthdate."""
    op = "age_calculation"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        birthdate_col = args.get("birthdate_column")
        reference_date = args.get("reference_date", "CURRENT_DATE")
        name = args.get("name") or "age"
        
        if not birthdate_col:
            return prior_sql
        
        return f"""
            SELECT *, 
            DATE_DIFF('year', {q_ident(birthdate_col)}, {reference_date}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class QuarterFromDate(Transformer):
    """Extract quarter from date."""
    op = "quarter_from_date"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        name = args.get("name") or f"{column}_quarter"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *, 
            QUARTER({q_ident(column)}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class WeekOfYear(Transformer):
    """Extract week of year from date."""
    op = "week_of_year"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        name = args.get("name") or f"{column}_week"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *, 
            WEEK({q_ident(column)}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


# ============================================================================
# 8. STATISTICAL OPERATIONS (20 transformers)
# ============================================================================

class Percentile(Transformer):
    """Calculate percentile of column."""
    op = "percentile"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        percentile = args.get("percentile", 0.5)
        name = args.get("name") or f"{column}_p{int(percentile*100)}"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *, 
            PERCENTILE_CONT({float(percentile)}) WITHIN GROUP (ORDER BY {q_ident(column)}) OVER () AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class Quartiles(Transformer):
    """Add Q1, Q2 (median), Q3 columns."""
    op = "quartiles"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        prefix = args.get("prefix") or column
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {q_ident(column)}) OVER () AS {q_ident(f'{prefix}_q1')},
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {q_ident(column)}) OVER () AS {q_ident(f'{prefix}_q2')},
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {q_ident(column)}) OVER () AS {q_ident(f'{prefix}_q3')}
            FROM {wrap(prior_sql)}
        """


class ZScore(Transformer):
    """Calculate z-score (standard score)."""
    op = "z_score"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        name = args.get("name") or f"{column}_zscore"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *,
            ({q_ident(column)} - AVG({q_ident(column)}) OVER ()) / 
            NULLIF(STDDEV_SAMP({q_ident(column)}) OVER (), 0) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class Normalize(Transformer):
    """Normalize column to 0-1 range (min-max normalization)."""
    op = "normalize"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        name = args.get("name") or f"{column}_normalized"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *,
            ({q_ident(column)} - MIN({q_ident(column)}) OVER ()) / 
            NULLIF(MAX({q_ident(column)}) OVER () - MIN({q_ident(column)}) OVER (), 0) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class StandardDeviation(Transformer):
    """Add column with standard deviation."""
    op = "standard_deviation"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        population = args.get("population", False)
        name = args.get("name") or f"{column}_std"
        
        if not column:
            return prior_sql
        
        func = "STDDEV_POP" if population else "STDDEV_SAMP"
        
        return f"""
            SELECT *,
            {func}({q_ident(column)}) OVER () AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class Variance(Transformer):
    """Add column with variance."""
    op = "variance"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        population = args.get("population", False)
        name = args.get("name") or f"{column}_var"
        
        if not column:
            return prior_sql
        
        func = "VAR_POP" if population else "VAR_SAMP"
        
        return f"""
            SELECT *,
            {func}({q_ident(column)}) OVER () AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class Correlation(Transformer):
    """Calculate correlation between two columns."""
    op = "correlation"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        x_column = args.get("x_column")
        y_column = args.get("y_column")
        name = args.get("name") or f"corr_{x_column}_{y_column}"
        
        if not x_column or not y_column:
            return prior_sql
        
        return f"""
            SELECT *,
            CORR({q_ident(x_column)}, {q_ident(y_column)}) OVER () AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class Covariance(Transformer):
    """Calculate covariance between two columns."""
    op = "covariance"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        x_column = args.get("x_column")
        y_column = args.get("y_column")
        population = args.get("population", False)
        name = args.get("name") or f"cov_{x_column}_{y_column}"
        
        if not x_column or not y_column:
            return prior_sql
        
        func = "COVAR_POP" if population else "COVAR_SAMP"
        
        return f"""
            SELECT *,
            {func}({q_ident(x_column)}, {q_ident(y_column)}) OVER () AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class Binning(Transformer):
    """Bin continuous values into discrete bins."""
    op = "binning"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        bins = args.get("bins", 5)
        method = args.get("method", "equal_width")  # equal_width or equal_frequency
        name = args.get("name") or f"{column}_bin"
        
        if not column:
            return prior_sql
        
        if method == "equal_frequency":
            # Percentile-based bins
            return f"""
                SELECT *,
                NTILE({int(bins)}) OVER (ORDER BY {q_ident(column)}) AS {q_ident(name)}
                FROM {wrap(prior_sql)}
            """
        else:
            # Equal width bins
            return f"""
                SELECT *,
                WIDTH_BUCKET({q_ident(column)}, 
                    MIN({q_ident(column)}) OVER (), 
                    MAX({q_ident(column)}) OVER (), 
                    {int(bins)}) AS {q_ident(name)}
                FROM {wrap(prior_sql)}
            """


class OutlierDetection(Transformer):
    """Flag outliers using IQR method."""
    op = "outlier_detection"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        multiplier = args.get("multiplier", 1.5)
        name = args.get("name") or f"{column}_is_outlier"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *,
            CASE 
                WHEN {q_ident(column)} < q1 - {float(multiplier)} * (q3 - q1) THEN true
                WHEN {q_ident(column)} > q3 + {float(multiplier)} * (q3 - q1) THEN true
                ELSE false
            END AS {q_ident(name)}
            FROM (
                SELECT *,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {q_ident(column)}) OVER () AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {q_ident(column)}) OVER () AS q3
                FROM {wrap(prior_sql)}
            )
        """


class MovingAverage(Transformer):
    """Calculate moving average."""
    op = "moving_average"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        window = args.get("window", 3)
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_ma{window}"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            AVG({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)} 
                ROWS BETWEEN {int(window)-1} PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class ExponentialMovingAverage(Transformer):
    """Calculate exponential moving average (approximation)."""
    op = "exponential_moving_average"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        alpha = args.get("alpha", 0.3)  # Smoothing factor
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_ema"
        
        if not column or not order_by:
            return prior_sql
        
        # Simplified EMA using recursive CTE would be complex
        # Using weighted moving average as approximation
        return f"""
            SELECT *,
            SUM({q_ident(column)} * POWER({float(1-alpha)}, rn - 1)) OVER w / 
            SUM(POWER({float(1-alpha)}, rn - 1)) OVER w AS {q_ident(name)}
            FROM (
                SELECT *, ROW_NUMBER() OVER (ORDER BY {q_ident(order_by)}) AS rn
                FROM {wrap(prior_sql)}
            )
            WINDOW w AS (ORDER BY {q_ident(order_by)} ROWS BETWEEN 10 PRECEDING AND CURRENT ROW)
        """


class CumulativeSum(Transformer):
    """Calculate cumulative sum."""
    op = "cumulative_sum"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        partition_by = args.get("partition_by")  # Optional
        name = args.get("name") or f"{column}_cumsum"
        
        if not column or not order_by:
            return prior_sql
        
        partition_clause = ""
        if partition_by:
            parts = ", ".join(q_ident(c) for c in partition_by) if isinstance(partition_by, list) else q_ident(partition_by)
            partition_clause = f"PARTITION BY {parts}"
        
        return f"""
            SELECT *,
            SUM({q_ident(column)}) OVER (
                {partition_clause}
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class CumulativeProduct(Transformer):
    """Calculate cumulative product (using log transformation)."""
    op = "cumulative_product"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_cumprod"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            EXP(SUM(LN({q_ident(column)})) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RankColumn(Transformer):
    """Add rank column."""
    op = "rank_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        order_by = args.get("order_by")
        direction = args.get("direction", "desc").upper()
        method = args.get("method", "rank")  # rank, dense_rank, row_number
        partition_by = args.get("partition_by")
        name = args.get("name") or "rank"
        
        if not order_by:
            return prior_sql
        
        partition_clause = ""
        if partition_by:
            parts = ", ".join(q_ident(c) for c in partition_by) if isinstance(partition_by, list) else q_ident(partition_by)
            partition_clause = f"PARTITION BY {parts}"
        
        func_map = {
            "rank": "RANK",
            "dense_rank": "DENSE_RANK",
            "row_number": "ROW_NUMBER"
        }
        
        func = func_map.get(method, "RANK")
        
        return f"""
            SELECT *,
            {func}() OVER (
                {partition_clause}
                ORDER BY {q_ident(order_by)} {direction}
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class PercentRank(Transformer):
    """Calculate percent rank (0 to 1)."""
    op = "percent_rank"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        order_by = args.get("order_by")
        direction = args.get("direction", "desc").upper()
        name = args.get("name") or f"{order_by}_pct_rank"
        
        if not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            PERCENT_RANK() OVER (ORDER BY {q_ident(order_by)} {direction}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class Mode(Transformer):
    """Calculate mode (most frequent value)."""
    op = "mode"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        name = args.get("name") or f"{column}_mode"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *,
            MODE() WITHIN GROUP (ORDER BY {q_ident(column)}) OVER () AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RollingStdDev(Transformer):
    """Calculate rolling standard deviation."""
    op = "rolling_std_dev"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        window = args.get("window", 3)
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_rolling_std"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            STDDEV_SAMP({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN {int(window)-1} PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class SimpleLinearRegression(Transformer):
    """Calculate simple linear regression slope and intercept."""
    op = "simple_linear_regression"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        x_column = args.get("x_column")
        y_column = args.get("y_column")
        predict_column = args.get("predict_column") or f"{y_column}_predicted"
        
        if not x_column or not y_column:
            return prior_sql
        
        return f"""
            SELECT *,
            REGR_SLOPE({q_ident(y_column)}, {q_ident(x_column)}) OVER () AS _slope,
            REGR_INTERCEPT({q_ident(y_column)}, {q_ident(x_column)}) OVER () AS _intercept,
            REGR_INTERCEPT({q_ident(y_column)}, {q_ident(x_column)}) OVER () + 
            REGR_SLOPE({q_ident(y_column)}, {q_ident(x_column)}) OVER () * {q_ident(x_column)} 
            AS {q_ident(predict_column)}
            FROM {wrap(prior_sql)}
        """


# ============================================================================
# 9. WINDOW FUNCTIONS (12 transformers)
# ============================================================================

class LagColumn(Transformer):
    """Access value from previous row."""
    op = "lag_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        offset = args.get("offset", 1)
        order_by = args.get("order_by")
        default = args.get("default")
        name = args.get("name") or f"{column}_lag{offset}"
        
        if not column or not order_by:
            return prior_sql
        
        default_clause = ""
        if default is not None:
            default_clause = f", {q_lit(default)}"
        
        return f"""
            SELECT *,
            LAG({q_ident(column)}, {int(offset)}{default_clause}) OVER (
                ORDER BY {q_ident(order_by)}
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class LeadColumn(Transformer):
    """Access value from next row."""
    op = "lead_column"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        offset = args.get("offset", 1)
        order_by = args.get("order_by")
        default = args.get("default")
        name = args.get("name") or f"{column}_lead{offset}"
        
        if not column or not order_by:
            return prior_sql
        
        default_clause = ""
        if default is not None:
            default_clause = f", {q_lit(default)}"
        
        return f"""
            SELECT *,
            LEAD({q_ident(column)}, {int(offset)}{default_clause}) OVER (
                ORDER BY {q_ident(order_by)}
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class FirstValue(Transformer):
    """Get first value in window."""
    op = "first_value"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        partition_by = args.get("partition_by")
        name = args.get("name") or f"{column}_first"
        
        if not column or not order_by:
            return prior_sql
        
        partition_clause = ""
        if partition_by:
            parts = ", ".join(q_ident(c) for c in partition_by) if isinstance(partition_by, list) else q_ident(partition_by)
            partition_clause = f"PARTITION BY {parts}"
        
        return f"""
            SELECT *,
            FIRST_VALUE({q_ident(column)}) OVER (
                {partition_clause}
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class LastValue(Transformer):
    """Get last value in window."""
    op = "last_value"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        partition_by = args.get("partition_by")
        name = args.get("name") or f"{column}_last"
        
        if not column or not order_by:
            return prior_sql
        
        partition_clause = ""
        if partition_by:
            parts = ", ".join(q_ident(c) for c in partition_by) if isinstance(partition_by, list) else q_ident(partition_by)
            partition_clause = f"PARTITION BY {parts}"
        
        return f"""
            SELECT *,
            LAST_VALUE({q_ident(column)}) OVER (
                {partition_clause}
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class NthValue(Transformer):
    """Get Nth value in window."""
    op = "nth_value"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        n = args.get("n", 1)
        order_by = args.get("order_by")
        partition_by = args.get("partition_by")
        name = args.get("name") or f"{column}_nth{n}"
        
        if not column or not order_by:
            return prior_sql
        
        partition_clause = ""
        if partition_by:
            parts = ", ".join(q_ident(c) for c in partition_by) if isinstance(partition_by, list) else q_ident(partition_by)
            partition_clause = f"PARTITION BY {parts}"
        
        return f"""
            SELECT *,
            NTH_VALUE({q_ident(column)}, {int(n)}) OVER (
                {partition_clause}
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RunningTotal(Transformer):
    """Calculate running total (alias for cumulative_sum)."""
    op = "running_total"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        partition_by = args.get("partition_by")
        name = args.get("name") or f"{column}_running_total"
        
        if not column or not order_by:
            return prior_sql
        
        partition_clause = ""
        if partition_by:
            parts = ", ".join(q_ident(c) for c in partition_by) if isinstance(partition_by, list) else q_ident(partition_by)
            partition_clause = f"PARTITION BY {parts}"
        
        return f"""
            SELECT *,
            SUM({q_ident(column)}) OVER (
                {partition_clause}
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RunningMin(Transformer):
    """Calculate running minimum."""
    op = "running_min"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_running_min"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            MIN({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RunningMax(Transformer):
    """Calculate running maximum."""
    op = "running_max"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_running_max"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            MAX({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RunningAverage(Transformer):
    """Calculate running average."""
    op = "running_average"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_running_avg"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            AVG({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RollingMin(Transformer):
    """Calculate rolling minimum."""
    op = "rolling_min"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        window = args.get("window", 3)
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_rolling_min"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            MIN({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN {int(window)-1} PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RollingMax(Transformer):
    """Calculate rolling maximum."""
    op = "rolling_max"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        window = args.get("window", 3)
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_rolling_max"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            MAX({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN {int(window)-1} PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RollingSum(Transformer):
    """Calculate rolling sum."""
    op = "rolling_sum"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        window = args.get("window", 3)
        order_by = args.get("order_by")
        name = args.get("name") or f"{column}_rolling_sum"
        
        if not column or not order_by:
            return prior_sql
        
        return f"""
            SELECT *,
            SUM({q_ident(column)}) OVER (
                ORDER BY {q_ident(order_by)}
                ROWS BETWEEN {int(window)-1} PRECEDING AND CURRENT ROW
            ) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


# ============================================================================
# 10. AGGREGATION (8 transformers)
# ============================================================================

class GroupAggregate(Transformer):
    """Group by columns and calculate aggregates."""
    op = "group_aggregate"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        group_by = args.get("group_by") or []
        aggs = args.get("aggs") or []
        
        if not aggs:
            return prior_sql
        
        gb_sql = ", ".join(q_ident(c) for c in group_by) if group_by else ""
        agg_exprs = []
        
        for a in aggs:
            fn = (a.get("fn") or "").lower()
            col = a.get("column", "*")
            alias = a.get("as")
            
            if not fn or not alias:
                continue
            
            if col == "*" and fn == "count":
                agg_exprs.append(f"COUNT(*) AS {q_ident(alias)}")
            else:
                agg_exprs.append(f"{fn.upper()}({q_ident(col)}) AS {q_ident(alias)}")
        
        if not agg_exprs:
            return prior_sql
        
        select_list = []
        if gb_sql:
            select_list.append(gb_sql)
        select_list.extend(agg_exprs)
        
        sql = f"SELECT {', '.join(select_list)} FROM {wrap(prior_sql)}"
        if gb_sql:
            sql += f" GROUP BY {gb_sql}"
        
        return sql


class WeightedAverage(Transformer):
    """Calculate weighted average."""
    op = "weighted_average"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        value_column = args.get("value_column")
        weight_column = args.get("weight_column")
        group_by = args.get("group_by") or []
        name = args.get("name") or "weighted_avg"
        
        if not value_column or not weight_column:
            return prior_sql
        
        gb_sql = ""
        if group_by:
            gb_sql = ", ".join(q_ident(c) for c in group_by)
            select_cols = f"{gb_sql}, "
            group_clause = f" GROUP BY {gb_sql}"
        else:
            select_cols = ""
            group_clause = ""
        
        return f"""
            SELECT {select_cols}
            SUM({q_ident(value_column)} * {q_ident(weight_column)}) / 
            SUM({q_ident(weight_column)}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
            {group_clause}
        """


class CountDistinct(Transformer):
    """Count distinct values."""
    op = "count_distinct"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        group_by = args.get("group_by") or []
        name = args.get("name") or f"{column}_distinct_count"
        
        if not column:
            return prior_sql
        
        gb_sql = ""
        if group_by:
            gb_sql = ", ".join(q_ident(c) for c in group_by)
            select_cols = f"{gb_sql}, "
            group_clause = f" GROUP BY {gb_sql}"
        else:
            select_cols = ""
            group_clause = ""
        
        return f"""
            SELECT {select_cols}
            COUNT(DISTINCT {q_ident(column)}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
            {group_clause}
        """


class StringAgg(Transformer):
    """Concatenate strings from group."""
    op = "string_agg"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        separator = args.get("separator", ", ")
        group_by = args.get("group_by") or []
        name = args.get("name") or f"{column}_concat"
        
        if not column or not group_by:
            return prior_sql
        
        gb_sql = ", ".join(q_ident(c) for c in group_by)
        
        return f"""
            SELECT {gb_sql},
            STRING_AGG({q_ident(column)}, {q_lit(separator)}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
            GROUP BY {gb_sql}
        """


class Join(Transformer):
    """Join with another dataset."""
    op = "join"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        right_sql = args.get("right_sql")
        on = args.get("on") or []
        how = (args.get("how") or "left").upper()
        
        if not right_sql or not on:
            return prior_sql
        
        conds = []
        for j in on:
            l = j.get("left")
            r = j.get("right")
            if l and r:
                conds.append(f"l.{q_ident(l)} = r.{q_ident(r)}")
        
        if not conds:
            return prior_sql
        
        sel_left = args.get("select_left") or ["*"]
        sel_right = args.get("select_right") or []
        
        select_parts = []
        if sel_left == ["*"]:
            select_parts.append("l.*")
        else:
            select_parts.extend([f"l.{q_ident(c)}" for c in sel_left])
        
        select_parts.extend([f"r.{q_ident(c)} AS {q_ident('right_' + c)}" for c in sel_right])
        
        return (
            f"SELECT {', '.join(select_parts)} FROM ({prior_sql}) l "
            f"{how} JOIN ({right_sql}) r ON "
            + " AND ".join(conds)
        )


class UnionAll(Transformer):
    """Union with another dataset."""
    op = "union_all"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        other_sql = args.get("other_sql")
        distinct = bool(args.get("distinct", False))
        
        if not other_sql:
            return prior_sql
        
        op = "UNION" if distinct else "UNION ALL"
        return f"SELECT * FROM {wrap(prior_sql)} {op} SELECT * FROM ({other_sql}) AS _u"


class Pivot(Transformer):
    """Pivot data from rows to columns."""
    op = "pivot"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        index = args.get("index") or []
        pivot_col = args.get("pivot_col")
        value_col = args.get("value_col")
        agg_fn = (args.get("agg_fn") or "sum").upper()
        values = args.get("values") or []
        
        if not index or not pivot_col or not value_col or not values:
            return prior_sql
        
        idx_sql = ", ".join(q_ident(c) for c in index)
        exprs = []
        
        for v in values:
            exprs.append(
                f"{agg_fn}(CASE WHEN {q_ident(pivot_col)} = {q_lit(v)} "
                f"THEN {q_ident(value_col)} ELSE NULL END) AS {q_ident(str(v))}"
            )
        
        return f"SELECT {idx_sql}, {', '.join(exprs)} FROM {wrap(prior_sql)} GROUP BY {idx_sql}"


class Unpivot(Transformer):
    """Unpivot data from columns to rows."""
    op = "unpivot"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        id_cols = args.get("id_cols") or []
        value_cols = args.get("value_cols") or []
        var_name = args.get("var_name") or "attribute"
        value_name = args.get("value_name") or "value"
        
        if not id_cols or not value_cols:
            return prior_sql
        
        selects = []
        id_sql = ", ".join(q_ident(c) for c in id_cols)
        
        for vc in value_cols:
            selects.append(
                f"SELECT {id_sql}, {q_lit(vc)} AS {q_ident(var_name)}, "
                f"{q_ident(vc)} AS {q_ident(value_name)} FROM {wrap(prior_sql)}"
            )
        
        return " UNION ALL ".join(selects)


# ============================================================================
# 11. DATA QUALITY (7 transformers)
# ============================================================================

class DataValidation(Transformer):
    """Add validation flag column based on conditions."""
    op = "data_validation"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        validations = args.get("validations") or []
        name = args.get("name") or "is_valid"
        
        if not validations:
            return prior_sql
        
        conditions = []
        for v in validations:
            condition = v.get("condition")
            if condition:
                conditions.append(f"({condition})")
        
        if not conditions:
            return prior_sql
        
        combined = " AND ".join(conditions)
        
        return f"""
            SELECT *,
            CASE WHEN {combined} THEN true ELSE false END AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class FindDuplicates(Transformer):
    """Flag duplicate rows."""
    op = "find_duplicates"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        name = args.get("name") or "is_duplicate"
        
        if not columns:
            return prior_sql
        
        cols_sql = ", ".join(q_ident(c) for c in columns)
        
        return f"""
            SELECT *,
            CASE WHEN COUNT(*) OVER (PARTITION BY {cols_sql}) > 1 THEN true ELSE false END AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class ValueFrequency(Transformer):
    """Add column showing frequency of each value."""
    op = "value_frequency"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        name = args.get("name") or f"{column}_frequency"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *,
            COUNT(*) OVER (PARTITION BY {q_ident(column)}) AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class OutlierFlag(Transformer):
    """Flag outliers using Z-score method."""
    op = "outlier_flag"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        threshold = args.get("threshold", 3.0)
        name = args.get("name") or f"{column}_is_outlier"
        
        if not column:
            return prior_sql
        
        return f"""
            SELECT *,
            CASE 
                WHEN ABS(({q_ident(column)} - AVG({q_ident(column)}) OVER ()) / 
                     NULLIF(STDDEV_SAMP({q_ident(column)}) OVER (), 0)) > {float(threshold)}
                THEN true
                ELSE false
            END AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class MissingValueFlag(Transformer):
    """Add flag indicating if any specified columns are NULL."""
    op = "missing_value_flag"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        name = args.get("name") or "has_missing"
        
        if not columns:
            return prior_sql
        
        null_checks = " OR ".join([f"{q_ident(c)} IS NULL" for c in columns])
        
        return f"""
            SELECT *,
            CASE WHEN {null_checks} THEN true ELSE false END AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class DataTypeCheck(Transformer):
    """Add flag indicating if column can be cast to specified type."""
    op = "data_type_check"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        column = args.get("column")
        target_type = args.get("target_type")
        name = args.get("name") or f"{column}_is_{target_type}"
        
        if not column or not target_type:
            return prior_sql
        
        return f"""
            SELECT *,
            CASE 
                WHEN TRY_CAST({q_ident(column)} AS {target_type}) IS NOT NULL THEN true
                ELSE false
            END AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


class RowQualityScore(Transformer):
    """Calculate row quality score based on completeness."""
    op = "row_quality_score"

    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        columns = args.get("columns") or []
        name = args.get("name") or "quality_score"
        
        if not columns:
            return prior_sql
        
        # Count non-null columns / total columns
        non_null_checks = " + ".join([
            f"CASE WHEN {q_ident(c)} IS NOT NULL THEN 1 ELSE 0 END"
            for c in columns
        ])
        
        return f"""
            SELECT *,
            ({non_null_checks}) * 100.0 / {len(columns)} AS {q_ident(name)}
            FROM {wrap(prior_sql)}
        """


# ============================================================================
# MASTER LIST - ALL 100+ TRANSFORMERS
# ============================================================================

ALL_TRANSFORMERS: List[Transformer] = [
    # 1. Table Shaping (7)
    SelectColumns(), DropColumns(), RenameColumns(), ReorderColumns(),
    AddConstantColumn(), DuplicateColumn(), MoveColumn(),
    
    # 2. Row Operations (10)
    SortRows(), LimitRows(), OffsetRows(), SampleRows(), DistinctRows(),
    RemoveDuplicates(), AddIndexColumn(), TopN(), BottomN(), RandomSample(),
    
    # 3. Filtering (4)
    FilterRows(), FilterRowsSafe(), FilterTopPercent(), FilterByRange(),
    
    # 4. Computed Columns (4)
    AddComputedColumn(), AddComputedSafe(), AddConditionalColumn(), AddMathColumn(),
    
    # 5. Data Cleaning (6)
    DropNulls(), FillNulls(), ReplaceValues(), ChangeType(), Coalesce(), CleanWhitespace(),
    
    # 6. Text Operations (12)
    TextTrim(), TextLower(), TextUpper(), TextReplace(), TextSplit(), TextMerge(),
    TextLength(), TextSubstring(), TextPad(), TextContains(), TextStartsWith(), TextEndsWith(),
    
    # 7. Datetime Operations (10)
    DateFromText(), DatePart(), DateTrunc(), FormatDatetime(), DateDiff(),
    DateAdd(), DateSubtract(), AgeCalculation(), QuarterFromDate(), WeekOfYear(),
    
    # 8. Statistical Operations (20)
    Percentile(), Quartiles(), ZScore(), Normalize(), StandardDeviation(),
    Variance(), Correlation(), Covariance(), Binning(), OutlierDetection(),
    MovingAverage(), ExponentialMovingAverage(), CumulativeSum(), CumulativeProduct(),
    RankColumn(), PercentRank(), Mode(), RollingStdDev(), SimpleLinearRegression(),
    
    # 9. Window Functions (12)
    LagColumn(), LeadColumn(), FirstValue(), LastValue(), NthValue(),
    RunningTotal(), RunningMin(), RunningMax(), RunningAverage(),
    RollingMin(), RollingMax(), RollingSum(),
    
    # 10. Aggregation (8)
    GroupAggregate(), WeightedAverage(), CountDistinct(), StringAgg(),
    Join(), UnionAll(), Pivot(), Unpivot(),
    
    # 11. Data Quality (7)
    DataValidation(), FindDuplicates(), ValueFrequency(), OutlierFlag(),
    MissingValueFlag(), DataTypeCheck(), RowQualityScore(),
]
