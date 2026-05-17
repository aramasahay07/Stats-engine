from __future__ import annotations

from app.models.process_mining import ColumnMapping, ProcessDataShape
from app.services.process_mining.shaper import build_canonical_event_log_view
from app.services.process_mining.validator import validate_pre_shape


def test_wide_to_long_shaping_builds_expected_activities(duckdb_con, wide_event_log_df):
    duckdb_con.register("wide_df", wide_event_log_df)
    duckdb_con.execute("CREATE VIEW base_view AS SELECT * FROM wide_df")

    mapping = ColumnMapping(
        case_id_column="case_id",
        activity_column="unused_activity",
        timestamp_column="unused_ts",
        attribute_columns=["department"],
    )
    shape = ProcessDataShape(
        format="wide",
        case_id_column="case_id",
        pivot_columns=["admission_ts", "review_ts", "discharge_ts"],
    )

    validate_pre_shape(duckdb_con, "base_view", mapping, shape)
    event_log_view = build_canonical_event_log_view(duckdb_con, "base_view", mapping, shape)

    rows = duckdb_con.execute(
        f"SELECT case_id, activity FROM {event_log_view} ORDER BY case_id, event_index"
    ).fetchall()

    assert rows == [
        ("c1", "Admission Ts"),
        ("c1", "Review Ts"),
        ("c1", "Discharge Ts"),
        ("c2", "Admission Ts"),
        ("c2", "Discharge Ts"),
    ]
