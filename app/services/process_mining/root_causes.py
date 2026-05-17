from __future__ import annotations

from typing import Iterable, List

from app.models.process_mining import RootCauseFinding
from app.services.process_mining.common import ensure_unique, qident


def analyze_root_causes(
    con,
    case_summary_view: str,
    attribute_columns: Iterable[str],
    limit: int = 25,
) -> List[RootCauseFinding]:
    columns = ensure_unique(attribute_columns)
    findings: list[RootCauseFinding] = []

    base_row = con.execute(
        f"""
        SELECT
            COALESCE(AVG(CASE WHEN has_rework THEN 1.0 ELSE 0.0 END), 0.0) AS rework_baseline,
            COALESCE(AVG(CASE WHEN sla_breached THEN 1.0 ELSE 0.0 END), 0.0) AS sla_baseline
        FROM {case_summary_view}
        """
    ).fetchone()
    rework_baseline = float(base_row[0] or 0.0)
    sla_baseline = float(base_row[1] or 0.0)

    for column in columns:
        rows = con.execute(
            f"""
            SELECT
                COALESCE(NULLIF(TRIM(CAST({qident(column)} AS VARCHAR)), ''), '(Blank)') AS category,
                COUNT(*) AS case_count,
                COALESCE(AVG(CASE WHEN has_rework THEN 1.0 ELSE 0.0 END), 0.0) AS rework_rate,
                COALESCE(AVG(CASE WHEN sla_breached THEN 1.0 ELSE 0.0 END), 0.0) AS sla_rate
            FROM {case_summary_view}
            GROUP BY 1
            HAVING COUNT(*) >= 5
            """
        ).fetchall()

        for category, case_count, rework_rate, sla_rate in rows:
            if rework_baseline > 0:
                rework_lift = float(rework_rate) / rework_baseline
                if rework_lift >= 1.3:
                    findings.append(
                        RootCauseFinding(
                            attribute=column,
                            category=str(category),
                            outcome="rework",
                            rate=float(rework_rate),
                            baseline=rework_baseline,
                            lift=float(rework_lift),
                            case_count=int(case_count),
                        )
                    )
            if sla_baseline > 0:
                sla_lift = float(sla_rate) / sla_baseline
                if sla_lift >= 1.3:
                    findings.append(
                        RootCauseFinding(
                            attribute=column,
                            category=str(category),
                            outcome="sla_breach",
                            rate=float(sla_rate),
                            baseline=sla_baseline,
                            lift=float(sla_lift),
                            case_count=int(case_count),
                        )
                    )

    findings.sort(key=lambda item: (-item.lift, -item.case_count, item.attribute, item.category))
    return findings[:limit]
