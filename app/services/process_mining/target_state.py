from __future__ import annotations

from typing import List, Optional, Tuple

from app.models.process_mining import ProcessEdge, ProcessMiningSummary, ProcessNode, ReworkLoop, TargetState, ProjectedImpact


def _edge_hours_saved(avg_duration: float, factor: float) -> float:
    return max(avg_duration * factor, 0.0)


def build_target_state(
    summary: ProcessMiningSummary,
    nodes: List[ProcessNode],
    edges: List[ProcessEdge],
    rework_loops: List[ReworkLoop],
    protected_activities: List[str],
    total_cases: int,
) -> tuple[TargetState, list[dict], Optional[ProcessEdge], str]:
    protected = set(protected_activities or [])
    edge_annotations: list[dict] = []
    assumptions: list[str] = []
    target_edges = [edge.model_copy(deep=True) for edge in edges]
    removed_nodes: list[str] = []

    constraint_edge: Optional[ProcessEdge] = None
    constraint_rationale = ""

    eligible_edges = [
        edge for edge in target_edges
        if edge.source not in protected and edge.target not in protected
    ]
    if eligible_edges:
        constraint_edge = max(eligible_edges, key=lambda edge: edge.avg_duration)
        saved = _edge_hours_saved(constraint_edge.avg_duration, 0.5)
        edge_annotations.append(
            {
                "source": constraint_edge.source,
                "target": constraint_edge.target,
                "kinds": ["constraint"],
                "rationale": f"This is the slowest non-protected handoff at {constraint_edge.avg_duration:.2f} hours.",
                "action": "compress",
                "hours_saved_per_case": saved,
                "cases_affected": constraint_edge.frequency,
                "title": f"Compress {constraint_edge.source} to {constraint_edge.target}",
            }
        )
        constraint_edge.avg_duration = max(constraint_edge.avg_duration - saved, 0.0)
        constraint_edge.median_duration = max(constraint_edge.median_duration - saved * 0.5, 0.0)
        assumptions.append("Constraint edge average duration can be reduced by 50% in the target state.")
        constraint_rationale = edge_annotations[-1]["rationale"]

    for loop in rework_loops:
        matching = next(
            (edge for edge in target_edges if edge.source == loop.activity or edge.target == loop.activity),
            None,
        )
        if matching is None:
            continue
        saved = matching.avg_duration * 0.6
        edge_annotations.append(
            {
                "source": matching.source,
                "target": matching.target,
                "kinds": ["rework"],
                "rationale": f"{loop.activity} repeats in {loop.affected_cases} cases and should be standardized.",
                "action": "standardize",
                "hours_saved_per_case": saved,
                "cases_affected": loop.affected_cases,
                "title": f"Standardize {loop.activity}",
            }
        )
        assumptions.append("Rework loops can be reduced by 60% through standardization.")

    low_frequency_threshold = max(total_cases * 0.05, 1)
    filtered_edges: list[ProcessEdge] = []
    for edge in target_edges:
        if edge.frequency < low_frequency_threshold and edge.source not in protected and edge.target not in protected:
            edge_annotations.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kinds": ["deviation"],
                    "rationale": "This edge appears in fewer than 5% of cases and can likely be removed from the target path.",
                    "action": "remove",
                    "hours_saved_per_case": edge.avg_duration,
                    "cases_affected": edge.frequency,
                    "title": f"Remove low-frequency path {edge.source} to {edge.target}",
                }
            )
            if edge.target not in removed_nodes:
                removed_nodes.append(edge.target)
            continue
        filtered_edges.append(edge)

    total_saved_per_case = sum(float(item.get("hours_saved_per_case") or 0.0) for item in edge_annotations)
    total_rework_cases = sum(loop.affected_cases for loop in rework_loops)
    projected_average_cycle = max(summary.average_cycle_time - total_saved_per_case, 0.0)
    projected_median_cycle = max(summary.median_cycle_time - total_saved_per_case * 0.5, 0.0)
    projected_rework_rate = max(summary.rework_rate * 0.4, 0.0) if rework_loops else summary.rework_rate
    projected_sla_breach = (
        max(summary.sla_breach_rate - (summary.average_cycle_time - projected_average_cycle) / max(summary.average_cycle_time, 1.0), 0.0)
        if summary.sla_breach_rate is not None
        else None
    )

    projected_summary = ProcessMiningSummary(
        total_cases=summary.total_cases,
        total_events=summary.total_events,
        unique_activities=summary.unique_activities,
        average_cycle_time=projected_average_cycle,
        median_cycle_time=projected_median_cycle,
        rework_rate=projected_rework_rate,
        variant_count=summary.variant_count,
        sla_breach_rate=projected_sla_breach,
    )

    cycle_time_reduction_pct = (
        ((summary.average_cycle_time - projected_average_cycle) / summary.average_cycle_time) * 100.0
        if summary.average_cycle_time > 0
        else 0.0
    )
    rework_reduction_pct = (
        ((summary.rework_rate - projected_rework_rate) / summary.rework_rate) * 100.0
        if summary.rework_rate > 0
        else 0.0
    )

    target_state = TargetState(
        nodes=nodes,
        edges=filtered_edges,
        node_annotations=[],
        edge_annotations=edge_annotations,
        removed_nodes=removed_nodes,
        added_edges=[],
        projected_summary=projected_summary,
        projected_impact=ProjectedImpact(
            cycle_time_reduction_pct=cycle_time_reduction_pct,
            rework_reduction_pct=rework_reduction_pct,
            throughput_uplift_pct=cycle_time_reduction_pct,
        ),
        assumptions=assumptions,
    )

    return target_state, edge_annotations, constraint_edge, constraint_rationale
