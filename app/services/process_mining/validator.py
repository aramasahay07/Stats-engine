from __future__ import annotations

from fastapi import HTTPException

from app.models.process_mining import ColumnMapping, ProcessDataShape
from app.services.process_mining.common import qident


def _view_columns(con, view_name: str) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {view_name}").fetchall()
    return {str(row[0]) for row in rows}


def validate_pre_shape(con, view_name: str, mapping: ColumnMapping, shape: ProcessDataShape) -> None:
    columns = _view_columns(con, view_name)

    if shape.format == "wide":
        case_id_column = shape.case_id_column or mapping.case_id_column
        required = [case_id_column, *shape.pivot_columns]
        if not shape.pivot_columns:
            raise HTTPException(
                status_code=422,
                detail="Wide format requires at least one pivot column in shape.pivot_columns.",
            )
    else:
        required = [mapping.case_id_column, mapping.activity_column, mapping.timestamp_column]

    optional = [
        mapping.resource_column,
        mapping.cost_column,
        *(mapping.attribute_columns or []),
    ]

    missing = [column for column in [*required, *optional] if column and column not in columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"These columns are missing from the dataset: {', '.join(sorted(missing))}.",
        )

    if shape.format == "long":
        parseable = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {view_name}
            WHERE {qident(mapping.timestamp_column)} IS NOT NULL
              AND TRY_CAST({qident(mapping.timestamp_column)} AS TIMESTAMP) IS NOT NULL
            """
        ).fetchone()[0]
        if int(parseable or 0) == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Column '{mapping.timestamp_column}' does not contain parseable timestamps. "
                    "Please map a different timestamp column or clean the data first."
                ),
            )

    if shape.format == "wide":
        case_id_column = shape.case_id_column or mapping.case_id_column
        usable_events = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {view_name}
            WHERE {qident(case_id_column)} IS NOT NULL
              AND (
                {" OR ".join(f"TRY_CAST({qident(col)} AS TIMESTAMP) IS NOT NULL" for col in shape.pivot_columns)}
              )
            """
        ).fetchone()[0]
        if int(usable_events or 0) == 0:
            raise HTTPException(
                status_code=422,
                detail="None of the wide-format pivot columns contain parseable timestamps.",
            )


def validate_canonical_event_log(con, event_log_view: str) -> None:
    total_cases = con.execute(f"SELECT COUNT(DISTINCT case_id) FROM {event_log_view}").fetchone()[0]
    if int(total_cases or 0) == 0:
        raise HTTPException(
            status_code=422,
            detail="The selected mapping produced zero valid cases after shaping.",
        )

    multi_event_cases = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT case_id
            FROM {event_log_view}
            GROUP BY case_id
            HAVING COUNT(*) >= 2
        ) t
        """
    ).fetchone()[0]
    if int(multi_event_cases or 0) == 0:
        raise HTTPException(
            status_code=422,
            detail="At least one case needs two or more events for process mining.",
        )

    blank_activities = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {event_log_view}
        WHERE activity IS NULL OR TRIM(activity) = ''
        """
    ).fetchone()[0]
    if int(blank_activities or 0) > 0:
        raise HTTPException(
            status_code=422,
            detail="Some events have blank activity names after shaping. Please clean the source columns.",
        )
