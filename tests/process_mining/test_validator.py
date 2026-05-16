from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException

from app.models.process_mining import ColumnMapping, ProcessDataShape
from app.services.process_mining.shaper import build_canonical_event_log_view
from app.services.process_mining.validator import validate_canonical_event_log, validate_pre_shape


def test_validate_pre_shape_rejects_missing_columns(duckdb_con):
    df = pd.DataFrame([{"case_id": "c1", "activity": "A"}])
    duckdb_con.register("df", df)
    duckdb_con.execute("CREATE VIEW base_view AS SELECT * FROM df")

    with pytest.raises(HTTPException) as exc:
        validate_pre_shape(
            duckdb_con,
            "base_view",
            ColumnMapping(case_id_column="case_id", activity_column="activity", timestamp_column="missing_ts"),
            ProcessDataShape(format="long"),
        )

    assert exc.value.status_code == 422
    assert "missing_ts" in str(exc.value.detail)


def test_validate_canonical_event_log_requires_multi_event_case(duckdb_con):
    df = pd.DataFrame([{"case_id": "c1", "activity": "A", "event_ts": "2024-01-01 08:00:00"}])
    duckdb_con.register("df", df)
    duckdb_con.execute("CREATE VIEW base_view AS SELECT * FROM df")

    event_log_view = build_canonical_event_log_view(
        duckdb_con,
        "base_view",
        ColumnMapping(case_id_column="case_id", activity_column="activity", timestamp_column="event_ts"),
        ProcessDataShape(format="long"),
    )

    with pytest.raises(HTTPException) as exc:
        validate_canonical_event_log(duckdb_con, event_log_view)

    assert exc.value.status_code == 422
    assert "two or more events" in str(exc.value.detail)
