from __future__ import annotations

import pytest

from app.models.process_mining import ColumnMapping, ProcessDataShape
from app.services.process_mining.metrics import (
    build_edge_duration_map,
    compute_bottlenecks,
    compute_process_map_edges,
    compute_process_map_nodes,
    compute_rework_loops,
    compute_summary,
    compute_variants,
    create_direct_follows_view,
)
from app.services.process_mining.shaper import build_canonical_event_log_view
from app.services.process_mining.validator import validate_canonical_event_log, validate_pre_shape


def test_core_metrics_are_computed_from_long_event_log(duckdb_con, long_event_log_df):
    duckdb_con.register("long_df", long_event_log_df)
    duckdb_con.execute("CREATE VIEW base_view AS SELECT * FROM long_df")

    mapping = ColumnMapping(
        case_id_column="case_id",
        activity_column="activity",
        timestamp_column="event_ts",
        attribute_columns=["department"],
    )
    shape = ProcessDataShape(format="long")

    validate_pre_shape(duckdb_con, "base_view", mapping, shape)
    event_log_view = build_canonical_event_log_view(duckdb_con, "base_view", mapping, shape)
    validate_canonical_event_log(duckdb_con, event_log_view)
    direct_follows_view = create_direct_follows_view(duckdb_con, event_log_view)

    summary = compute_summary(duckdb_con, event_log_view)
    nodes = compute_process_map_nodes(duckdb_con, event_log_view, direct_follows_view)
    edges = compute_process_map_edges(duckdb_con, direct_follows_view)
    variants = compute_variants(duckdb_con, event_log_view)
    bottlenecks = compute_bottlenecks(duckdb_con, direct_follows_view)
    rework_loops = compute_rework_loops(duckdb_con, event_log_view)
    edge_map = build_edge_duration_map(edges)

    assert summary.total_cases == 3
    assert summary.total_events == 9
    assert summary.unique_activities == 4
    assert summary.variant_count == 3
    assert summary.rework_rate == pytest.approx(1 / 3)
    assert summary.avg_cycle_time_hours == pytest.approx(13 / 3)
    assert summary.median_cycle_time_hours == pytest.approx(3.0)

    assert nodes[0].activity == "A"
    assert any(edge.from_activity == "A" and edge.to_activity == "B" and edge.frequency == 2 for edge in edges)
    assert any(variant.activities == ["A", "B", "C"] for variant in variants)
    assert bottlenecks[0].from_activity == "A"
    assert bottlenecks[0].to_activity == "B"
    assert rework_loops[0].activity == "B"
    assert rework_loops[0].affected_cases == 1
    assert edge_map["A->B"]["avg"] == pytest.approx(3.0)
