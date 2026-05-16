from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.process_mining import Bottleneck, ProcessMapEdge, ProcessMapNode, ProcessSummary, ProcessVariant, ReworkLoop
from app.services.process_mining.common import qstring

SEQUENCE_DELIM = "|||"


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
            activity AS from_activity,
            next_activity AS to_activity,
            event_ts,
            next_event_ts,
            EXTRACT(EPOCH FROM (next_event_ts - event_ts)) / 3600.0 AS wait_hours
        FROM ordered
        WHERE next_activity IS NOT NULL
          AND next_event_ts IS NOT NULL
        """
    )
    return "pm_direct_follows"


def compute_summary(con, event_log_view: str, goals: Optional[Dict[str, Any]] = None) -> ProcessSummary:
    goals = goals or {}
    sla_hours = goals.get("sla_hours")

    row = con.execute(
        f"""
        WITH case_spans AS (
            SELECT
                case_id,
                EXTRACT(EPOCH FROM (MAX(event_ts) - MIN(event_ts))) / 3600.0 AS cycle_time_hours
            FROM {event_log_view}
            GROUP BY case_id
        ),
        rework_cases AS (
            SELECT DISTINCT case_id
            FROM (
                SELECT case_id, activity, COUNT(*) AS n
                FROM {event_log_view}
                GROUP BY case_id, activity
                HAVING COUNT(*) > 1
            ) t
        ),
        variants AS (
            SELECT
                case_id,
                STRING_AGG(activity, {qstring(SEQUENCE_DELIM)} ORDER BY event_index) AS sequence
            FROM {event_log_view}
            GROUP BY case_id
        )
        SELECT
            (SELECT COUNT(DISTINCT case_id) FROM {event_log_view}) AS total_cases,
            (SELECT COUNT(*) FROM {event_log_view}) AS total_events,
            (SELECT COUNT(DISTINCT activity) FROM {event_log_view}) AS unique_activities,
            (SELECT AVG(cycle_time_hours) FROM case_spans) AS avg_cycle_time_hours,
            (SELECT MEDIAN(cycle_time_hours) FROM case_spans) AS median_cycle_time_hours,
            (
                SELECT
                    CASE
                        WHEN (SELECT COUNT(*) FROM case_spans) = 0 THEN NULL
                        ELSE COUNT(*) * 1.0 / (SELECT COUNT(*) FROM case_spans)
                    END
                FROM rework_cases
            ) AS rework_rate,
            (SELECT COUNT(DISTINCT sequence) FROM variants) AS variant_count
        """
    ).fetchone()

    sla_breach_rate = None
    if sla_hours is not None:
        breach = con.execute(
            f"""
            WITH case_spans AS (
                SELECT
                    case_id,
                    EXTRACT(EPOCH FROM (MAX(event_ts) - MIN(event_ts))) / 3600.0 AS cycle_time_hours
                FROM {event_log_view}
                GROUP BY case_id
            )
            SELECT
                CASE
                    WHEN COUNT(*) = 0 THEN NULL
                    ELSE AVG(CASE WHEN cycle_time_hours > {float(sla_hours)} THEN 1.0 ELSE 0.0 END)
                END
            FROM case_spans
            """
        ).fetchone()[0]
        sla_breach_rate = float(breach) if breach is not None else None

    return ProcessSummary(
        total_cases=int(row[0] or 0),
        total_events=int(row[1] or 0),
        unique_activities=int(row[2] or 0),
        avg_cycle_time_hours=float(row[3]) if row[3] is not None else None,
        median_cycle_time_hours=float(row[4]) if row[4] is not None else None,
        rework_rate=float(row[5]) if row[5] is not None else None,
        variant_count=int(row[6] or 0),
        sla_breach_rate=sla_breach_rate,
    )


def compute_process_map_nodes(con, event_log_view: str, direct_follows_view: str) -> List[ProcessMapNode]:
    rows = con.execute(
        f"""
        WITH freq AS (
            SELECT
                activity,
                COUNT(*) AS frequency,
                COUNT(DISTINCT case_id) AS case_frequency
            FROM {event_log_view}
            GROUP BY activity
        ),
        durations AS (
            SELECT
                from_activity AS activity,
                AVG(wait_hours) AS avg_duration_in_state_hours
            FROM {direct_follows_view}
            GROUP BY from_activity
        )
        SELECT
            freq.activity,
            freq.frequency,
            freq.case_frequency,
            durations.avg_duration_in_state_hours
        FROM freq
        LEFT JOIN durations USING (activity)
        ORDER BY freq.frequency DESC, freq.activity
        """
    ).fetchall()

    return [
        ProcessMapNode(
            activity=str(row[0]),
            frequency=int(row[1]),
            case_frequency=int(row[2]),
            avg_duration_in_state_hours=float(row[3]) if row[3] is not None else None,
        )
        for row in rows
    ]


def compute_process_map_edges(con, direct_follows_view: str) -> List[ProcessMapEdge]:
    rows = con.execute(
        f"""
        SELECT
            from_activity,
            to_activity,
            COUNT(*) AS frequency,
            COUNT(DISTINCT case_id) AS case_frequency,
            AVG(wait_hours) AS avg_wait_hours,
            MEDIAN(wait_hours) AS median_wait_hours,
            QUANTILE_CONT(wait_hours, 0.9) AS p90_wait_hours
        FROM {direct_follows_view}
        GROUP BY from_activity, to_activity
        ORDER BY frequency DESC, from_activity, to_activity
        """
    ).fetchall()

    return [
        ProcessMapEdge(
            from_activity=str(row[0]),
            to_activity=str(row[1]),
            frequency=int(row[2]),
            case_frequency=int(row[3]),
            avg_wait_hours=float(row[4]) if row[4] is not None else None,
            median_wait_hours=float(row[5]) if row[5] is not None else None,
            p90_wait_hours=float(row[6]) if row[6] is not None else None,
        )
        for row in rows
    ]


def compute_variants(con, event_log_view: str, limit: int = 20) -> List[ProcessVariant]:
    rows = con.execute(
        f"""
        WITH case_sequences AS (
            SELECT
                case_id,
                STRING_AGG(activity, {qstring(SEQUENCE_DELIM)} ORDER BY event_index) AS sequence,
                EXTRACT(EPOCH FROM (MAX(event_ts) - MIN(event_ts))) / 3600.0 AS cycle_time_hours
            FROM {event_log_view}
            GROUP BY case_id
        ),
        total AS (
            SELECT COUNT(*) AS total_cases
            FROM case_sequences
        )
        SELECT
            sequence,
            COUNT(*) AS case_count,
            COUNT(*) * 100.0 / total.total_cases AS percentage,
            AVG(cycle_time_hours) AS avg_cycle_time_hours
        FROM case_sequences, total
        GROUP BY sequence, total.total_cases
        ORDER BY case_count DESC, sequence
        LIMIT {int(limit)}
        """
    ).fetchall()

    return [
        ProcessVariant(
            activities=str(row[0]).split(SEQUENCE_DELIM) if row[0] else [],
            case_count=int(row[1]),
            percentage=float(row[2]),
            avg_cycle_time_hours=float(row[3]) if row[3] is not None else None,
        )
        for row in rows
    ]


def compute_bottlenecks(con, direct_follows_view: str, limit: int = 15) -> List[Bottleneck]:
    rows = con.execute(
        f"""
        SELECT
            from_activity,
            to_activity,
            COUNT(*) AS frequency,
            AVG(wait_hours) AS avg_wait_hours,
            MEDIAN(wait_hours) AS median_wait_hours,
            QUANTILE_CONT(wait_hours, 0.9) AS p90_wait_hours
        FROM {direct_follows_view}
        GROUP BY from_activity, to_activity
        ORDER BY avg_wait_hours DESC NULLS LAST, frequency DESC
        LIMIT {int(limit)}
        """
    ).fetchall()

    return [
        Bottleneck(
            from_activity=str(row[0]),
            to_activity=str(row[1]),
            frequency=int(row[2]),
            avg_wait_hours=float(row[3]) if row[3] is not None else None,
            median_wait_hours=float(row[4]) if row[4] is not None else None,
            p90_wait_hours=float(row[5]) if row[5] is not None else None,
        )
        for row in rows
    ]


def compute_rework_loops(con, event_log_view: str) -> List[ReworkLoop]:
    rows = con.execute(
        f"""
        WITH repeated AS (
            SELECT
                case_id,
                activity,
                COUNT(*) AS activity_count
            FROM {event_log_view}
            GROUP BY case_id, activity
            HAVING COUNT(*) > 1
        ),
        total AS (
            SELECT COUNT(DISTINCT case_id) AS total_cases
            FROM {event_log_view}
        )
        SELECT
            activity,
            COUNT(DISTINCT case_id) AS affected_cases,
            COUNT(DISTINCT case_id) * 100.0 / total.total_cases AS affected_case_pct,
            SUM(activity_count - 1) AS repeat_events
        FROM repeated, total
        GROUP BY activity, total.total_cases
        ORDER BY affected_cases DESC, repeat_events DESC, activity
        """
    ).fetchall()

    return [
        ReworkLoop(
            activity=str(row[0]),
            affected_cases=int(row[1]),
            affected_case_pct=float(row[2]),
            repeat_events=int(row[3]),
        )
        for row in rows
    ]


def build_edge_duration_map(edges: List[ProcessMapEdge]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for edge in edges:
        out[f"{edge.from_activity}->{edge.to_activity}"] = {
            "avg": edge.avg_wait_hours,
            "median": edge.median_wait_hours,
            "p90": edge.p90_wait_hours,
        }
    return out
