from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd
import pytest


@pytest.fixture
def long_event_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"case_id": "c1", "activity": "A", "event_ts": datetime(2024, 1, 1, 8, 0), "department": "X"},
            {"case_id": "c1", "activity": "B", "event_ts": datetime(2024, 1, 1, 10, 0), "department": "X"},
            {"case_id": "c1", "activity": "C", "event_ts": datetime(2024, 1, 1, 11, 0), "department": "Y"},
            {"case_id": "c2", "activity": "A", "event_ts": datetime(2024, 1, 2, 9, 0), "department": "X"},
            {"case_id": "c2", "activity": "B", "event_ts": datetime(2024, 1, 2, 13, 0), "department": "Y"},
            {"case_id": "c2", "activity": "B", "event_ts": datetime(2024, 1, 2, 15, 0), "department": "Y"},
            {"case_id": "c2", "activity": "C", "event_ts": datetime(2024, 1, 2, 18, 0), "department": "Y"},
            {"case_id": "c3", "activity": "A", "event_ts": datetime(2024, 1, 3, 7, 0), "department": "Z"},
            {"case_id": "c3", "activity": "D", "event_ts": datetime(2024, 1, 3, 8, 0), "department": "Z"},
        ]
    )


@pytest.fixture
def wide_event_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "c1",
                "admission_ts": "2024-01-01 08:00:00",
                "review_ts": "2024-01-01 10:00:00",
                "discharge_ts": "2024-01-01 12:00:00",
                "department": "North",
            },
            {
                "case_id": "c2",
                "admission_ts": "2024-01-02 09:00:00",
                "review_ts": None,
                "discharge_ts": "2024-01-02 14:00:00",
                "department": "South",
            },
        ]
    )


@pytest.fixture
def duckdb_con():
    con = duckdb.connect(":memory:")
    try:
        yield con
    finally:
        con.close()
