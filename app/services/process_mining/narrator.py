from __future__ import annotations

from typing import List

from app.models.process_mining import (
    ProcessAIInsights,
    ProcessBottleneck,
    ProcessMiningSummary,
    ProcessVariant,
    ReworkLoop,
)


def build_ai_insights(
    summary: ProcessMiningSummary,
    variants: List[ProcessVariant],
    bottlenecks: List[ProcessBottleneck],
    rework_loops: List[ReworkLoop],
) -> ProcessAIInsights:
    findings: list[str] = []
    actions: list[str] = []

    if summary.total_cases > 0:
        findings.append(
            f"Average cycle time is {summary.average_cycle_time:.1f} hours across {summary.total_cases} cases."
        )
    if bottlenecks:
        slowest = bottlenecks[0]
        findings.append(
            f"The slowest directly-follows step is {slowest.from_activity} to {slowest.to_activity} "
            f"at {slowest.average_wait_time:.1f} hours on average."
        )
        actions.append(
            f"Review the handoff from {slowest.from_activity} to {slowest.to_activity} first."
        )
    if rework_loops:
        top_loop = rework_loops[0]
        findings.append(
            f"Rework is concentrated in {top_loop.activity}, affecting {top_loop.affected_cases} cases."
        )
        actions.append(f"Standardize the {top_loop.activity} step to reduce repeat work.")
    if variants:
        top_variant = variants[0]
        findings.append(
            f"The most common path appears in {top_variant.case_count} cases "
            f"({top_variant.percentage:.1f}% of observed cases)."
        )

    if not actions:
        actions.append("Review the most common path and longest waits to identify the first improvement target.")

    summary_line = (
        f"Process mining completed on {summary.total_events} events across {summary.total_cases} cases. "
        f"Found {summary.unique_activities} activities and {summary.variant_count} distinct variants."
    )
    return ProcessAIInsights(
        executive_summary=summary_line,
        key_findings=findings,
        recommended_actions=actions,
    )
