from __future__ import annotations

from typing import List

from app.models.process_mining import ColumnMapping, ProcessDataShape
from app.services.process_mining.common import ensure_unique, humanize_activity, qident, qstring


def _resource_expr(column: str | None) -> str:
    if not column:
        return "NULL::VARCHAR AS resource"
    return f"CAST({qident(column)} AS VARCHAR) AS resource"


def _cost_expr(column: str | None) -> str:
    if not column:
        return "NULL::DOUBLE AS cost"
    return f"TRY_CAST({qident(column)} AS DOUBLE) AS cost"


def _attribute_exprs(attribute_columns: List[str]) -> List[str]:
    return [f"{qident(column)} AS {qident(column)}" for column in ensure_unique(attribute_columns)]


def build_canonical_event_log_view(
    con,
    base_view: str,
    mapping: ColumnMapping,
    shape: ProcessDataShape,
) -> str:
    attribute_exprs = _attribute_exprs(mapping.attribute_columns)
    attr_sql = (", " + ", ".join(attribute_exprs)) if attribute_exprs else ""

    if shape.format == "wide":
        case_id_column = shape.case_id_column or mapping.case_id_column
        unions: list[str] = []
        for pivot_column in shape.pivot_columns:
            unions.append(
                f"""
                SELECT
                    CAST({qident(case_id_column)} AS VARCHAR) AS case_id,
                    {qstring(humanize_activity(pivot_column))} AS activity,
                    TRY_CAST({qident(pivot_column)} AS TIMESTAMP) AS event_ts,
                    {_resource_expr(mapping.resource_column)},
                    {_cost_expr(mapping.cost_column)}
                    {attr_sql}
                FROM {base_view}
                """
            )

        raw_sql = " UNION ALL ".join(unions)
        con.execute(f"CREATE OR REPLACE TEMP VIEW pm_event_log_raw AS {raw_sql}")
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW pm_event_log_raw AS
            SELECT
                CAST({qident(mapping.case_id_column)} AS VARCHAR) AS case_id,
                NULLIF(TRIM(CAST({qident(mapping.activity_column)} AS VARCHAR)), '') AS activity,
                TRY_CAST({qident(mapping.timestamp_column)} AS TIMESTAMP) AS event_ts,
                {_resource_expr(mapping.resource_column)},
                {_cost_expr(mapping.cost_column)}
                {attr_sql}
            FROM {base_view}
            """
        )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW pm_event_log_filtered AS
        SELECT *
        FROM pm_event_log_raw
        WHERE case_id IS NOT NULL
          AND activity IS NOT NULL
          AND event_ts IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW pm_event_log AS
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY case_id
                ORDER BY event_ts, activity
            ) AS event_index
        FROM pm_event_log_filtered
        """
    )
    return "pm_event_log"
