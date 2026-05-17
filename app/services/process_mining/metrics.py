from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from app.models.process_mining import (
    CaseRecord,
    ProcessBottleneck,
    ProcessEdge,
    ProcessEdgeDuration,
    ProcessMiningSummary,
    ProcessNode,
    ProcessVariant,
    ReworkLoop,
)
from app.services.process_mining.common import ensure_unique, qident, qstring

SEQUENCE_DELIM = "|||"
EDGE_DELIM = "||"


def create_direct_follows_view(con, event_log_view: str) -> str:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pm_direct_follows AS
        WITH ordered AS (
            SELECT
                case_id,
                activity,
                event_ts,
                event_index,
                LEAD(activity) OVER (
                    PARTITION BY case_id
                    ORDER BY event_ts, activity, event_index
                ) AS next_activity,
                LEAD(event_ts) OVER (
                    PARTITION BY case_id
                    ORDER BY event_ts, activity, event_index
                ) AS next_event_ts
            FROM {event_log_view}
        )
        SELECT
            case_id,
            activity AS source,
            next_activity AS target,
            EXTRACT(EPOCH FROM (next_event_ts - event_ts)) / 3600.0 AS wait_hours
        FROM ordered
        WHERE next_activity IS NOT NULL
          AND next_event_ts IS NOT NULL
        """
    )
    return "pm_direct_follows"


def create_case_summary_view(
    con,
    event_log_view: str,
    attribute_columns: Iterable[str],
    sla_hours: Optional[float],
) -> str:
    attr_columns = ensure_unique(attribute_columns)
    attr_selects = [
        f"COALESCE(MIN(CAST({qident(column)} AS VARCHAR)), '') AS {qident(column)}"
        for column in attr_columns
    ]
    attr_sql = (",\n                " + ",\n                ".join(attr_selects)) if attr_selects else ""
    sla_expr = (
        f"CASE WHEN EXTRACT(EPOCH FROM (MAX(event_ts) - MIN(event_ts))) / 3600.0 > {float(sla_hours)} THEN TRUE ELSE FALSE END"
        if sla_hours is not None
        else "FALSE"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW pm_case_summary AS
        WITH repeats AS (
            SELECT case_id, activity, COUNT(*) AS activity_count
            FROM {event_log_view}
            GROUP BY case_id, activity
            HAVING COUNT(*) > 1
        )
        SELECT
            e.case_id,
            STRING_AGG(e.activity, {qstring(SEQUENCE_DELIM)} ORDER BY e.event_index) AS sequence,
            EXTRACT(EPOCH FROM (MAX(e.event_ts) - MIN(e.event_ts))) / 3600.0 AS cycle_time,
            MIN(e.event_ts) AS start_ts,
            STRFTIME(MIN(e.event_ts), '%Y-%m-%dT%H:%M:%SZ') AS start_iso,
            COALESCE(SUM(e.cost), 0.0) AS amount,
            EXISTS(SELECT 1 FROM repeats r WHERE r.case_id = e.case_id) AS has_rework,
            {sla_expr} AS sla_breached
            {attr_sql}
        FROM {event_log_view} e
        GROUP BY e.case_id
        """
    )
    return "pm_case_summary"


def compute_summary(con, case_summary_view: str) -> ProcessMiningSummary:
    row = con.execute(
        f"""
        WITH variants AS (
            SELECT COUNT(DISTINCT sequence) AS variant_count
            FROM {case_summary_view}
        )
        SELECT
            COUNT(*) AS total_cases,
            (SELECT COUNT(*) FROM pm_event_log) AS total_events,
            (SELECT COUNT(DISTINCT activity) FROM pm_event_log) AS unique_activities,
            COALESCE(AVG(cycle_time), 0.0) AS average_cycle_time,
            COALESCE(MEDIAN(cycle_time), 0.0) AS median_cycle_time,
            COALESCE(AVG(CASE WHEN has_rework THEN 1.0 ELSE 0.0 END), 0.0) AS rework_rate,
            (SELECT variant_count FROM variants) AS variant_count,
            COALESCE(AVG(CASE WHEN sla_breached THEN 1.0 ELSE 0.0 END), 0.0) AS sla_breach_rate
        FROM {case_summary_view}
        """
    ).fetchone()

    return ProcessMiningSummary(
        total_cases=int(row[0] or 0),
        total_events=int(row[1] or 0),
        unique_activities=int(row[2] or 0),
        average_cycle_time=float(row[3] or 0.0),
        median_cycle_time=float(row[4] or 0.0),
        rework_rate=float(row[5] or 0.0),
        variant_count=int(row[6] or 0),
        sla_breach_rate=float(row[7] or 0.0),
    )


def compute_process_map_nodes(con, direct_follows_view: str) -> List[ProcessNode]:
    rows = con.execute(
        f"""
        WITH node_freq AS (
            SELECT activity AS id, COUNT(*) AS frequency
            FROM pm_event_log
            GROUP BY activity
        ),
        node_duration AS (
            SELECT source AS id, AVG(wait_hours) AS avg_duration
            FROM {direct_follows_view}
            GROUP BY source
        )
        SELECT
            node_freq.id,
            node_freq.id AS label,
            node_freq.frequency,
            COALESCE(node_duration.avg_duration, 0.0) AS avg_duration
        FROM node_freq
        LEFT JOIN node_duration USING (id)
        ORDER BY node_freq.frequency DESC, node_freq.id
        """
    ).fetchall()

    return [
        ProcessNode(
            id=str(row[0]),
            label=str(row[1]),
            frequency=int(row[2]),
            avg_duration=float(row[3] or 0.0),
        )
        for row in rows
    ]


def compute_process_map_edges(con, direct_follows_view: str) -> List[ProcessEdge]:
    rows = con.execute(
        f"""
        SELECT
            source,
            target,
            COUNT(*) AS frequency,
            COALESCE(AVG(wait_hours), 0.0) AS avg_duration,
            COALESCE(MEDIAN(wait_hours), 0.0) AS median_duration
        FROM {direct_follows_view}
        GROUP BY source, target
        ORDER BY frequency DESC, source, target
        """
    ).fetchall()

    return [
        ProcessEdge(
            source=str(row[0]),
            target=str(row[1]),
            frequency=int(row[2]),
            avg_duration=float(row[3] or 0.0),
            median_duration=float(row[4] or 0.0),
        )
        for row in rows
    ]


def compute_variants(con, case_summary_view: str, limit: int = 20) -> List[ProcessVariant]:
    rows = con.execute(
        f"""
        WITH variant_rollup AS (
            SELECT
                sequence,
                COUNT(*) AS case_count,
                AVG(cycle_time) AS average_cycle_time
            FROM {case_summary_view}
            GROUP BY sequence
        ),
        total AS (
            SELECT COUNT(*) AS total_cases
            FROM {case_summary_view}
        ),
        ranked AS (
            SELECT
                DENSE_RANK() OVER (ORDER BY case_count DESC, sequence) AS variant_id,
                sequence,
                case_count,
                case_count * 100.0 / total.total_cases AS percentage,
                average_cycle_time
            FROM variant_rollup, total
        )
        SELECT variant_id, sequence, case_count, percentage, average_cycle_time
        FROM ranked
        ORDER BY variant_id
        LIMIT {int(limit)}
        """
    ).fetchall()

    return [
        ProcessVariant(
            variant_id=int(row[0]),
            path=str(row[1]).split(SEQUENCE_DELIM) if row[1] else [],
            case_count=int(row[2]),
            percentage=float(row[3] or 0.0),
            average_cycle_time=float(row[4] or 0.0),
        )
        for row in rows
    ]


def compute_case_records(
    con,
    case_summary_view: str,
    variants: List[ProcessVariant],
) -> List[CaseRecord]:
    variant_map = {SEQUENCE_DELIM.join(variant.path): variant.variant_id for variant in variants}

    rows = con.execute(f"SELECT * FROM {case_summary_view} ORDER BY case_id").fetchdf()
    records: list[CaseRecord] = []
    for row in rows.to_dict(orient="records"):
        sequence = str(row.get("sequence") or "")
        path = sequence.split(SEQUENCE_DELIM) if sequence else []
        records.append(
            CaseRecord(
                case_id=str(row.get("case_id")),
                payer=str(row.get("payer") or ""),
                department=str(row.get("department") or ""),
                variant_id=int(variant_map.get(sequence, 0)),
                path=path,
                cycle_time=float(row.get("cycle_time") or 0.0),
                has_rework=bool(row.get("has_rework")),
                sla_breached=bool(row.get("sla_breached")),
                denial_reason=str(row.get("denial_reason")) if row.get("denial_reason") not in (None, "") else None,
                amount=float(row.get("amount") or 0.0),
                start_iso=str(row.get("start_iso") or ""),
            )
        )
    return records


def compute_bottlenecks(con, direct_follows_view: str, limit: int = 15) -> List[ProcessBottleneck]:
    rows = con.execute(
        f"""
        SELECT
            source,
            target,
            COALESCE(AVG(wait_hours), 0.0) AS average_wait_time,
            COALESCE(MEDIAN(wait_hours), 0.0) AS median_wait_time,
            COALESCE(QUANTILE_CONT(wait_hours, 0.9), 0.0) AS p90_wait_time,
            COUNT(*) AS case_count
        FROM {direct_follows_view}
        GROUP BY source, target
        ORDER BY average_wait_time DESC, case_count DESC, source, target
        LIMIT {int(limit)}
        """
    ).fetchall()

    return [
        ProcessBottleneck(
            from_activity=str(row[0]),
            to_activity=str(row[1]),
            average_wait_time=float(row[2] or 0.0),
            median_wait_time=float(row[3] or 0.0),
            p90_wait_time=float(row[4] or 0.0),
            case_count=int(row[5]),
        )
        for row in rows
    ]


def compute_rework_loops(con) -> List[ReworkLoop]:
    rows = con.execute(
        """
        WITH repeats AS (
            SELECT
                case_id,
                activity,
                COUNT(*) AS activity_count
            FROM pm_event_log
            GROUP BY case_id, activity
            HAVING COUNT(*) > 1
        ),
        totals AS (
            SELECT COUNT(DISTINCT case_id) AS total_cases
            FROM pm_event_log
        )
        SELECT
            activity,
            MAX(activity_count) AS repeat_count,
            COUNT(DISTINCT case_id) AS affected_cases,
            COUNT(DISTINCT case_id) * 100.0 / totals.total_cases AS percentage_of_cases
        FROM repeats, totals
        GROUP BY activity, totals.total_cases
        ORDER BY affected_cases DESC, repeat_count DESC, activity
        """
    ).fetchall()

    return [
        ReworkLoop(
            activity=str(row[0]),
            repeat_count=int(row[1]),
            affected_cases=int(row[2]),
            percentage_of_cases=float(row[3] or 0.0),
        )
        for row in rows
    ]


def build_edge_duration_map(edges: List[ProcessEdge]) -> Dict[str, ProcessEdgeDuration]:
    return {
        f"{edge.source}{EDGE_DELIM}{edge.target}": ProcessEdgeDuration(
            avg=edge.avg_duration,
            median=edge.median_duration,
        )
        for edge in edges
    }
