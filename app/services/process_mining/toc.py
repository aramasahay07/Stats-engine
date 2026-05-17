from __future__ import annotations

from typing import Optional

from app.models.process_mining import ProcessEdge, TocAnalysis, TocConstraintEdge, TocStep


def build_toc_analysis(
    constraint_edge: Optional[ProcessEdge],
    rationale: str,
    projected_throughput_impact_pct: float,
) -> Optional[TocAnalysis]:
    if constraint_edge is None:
        return None

    steps = [
        TocStep(step=1, name="Identify", description=f"Identify {constraint_edge.source} to {constraint_edge.target} as the current constraint."),
        TocStep(step=2, name="Exploit", description="Reduce idle time and interruptions on the constraint before adding capacity."),
        TocStep(step=3, name="Subordinate", description="Align upstream and downstream work to protect the constrained handoff."),
        TocStep(step=4, name="Elevate", description="Add staffing, scheduling, or automation only after local waste is removed."),
        TocStep(step=5, name="Repeat", description="After improving this edge, review the map again for the next constraint."),
    ]
    return TocAnalysis(
        constraint_edge=TocConstraintEdge(source=constraint_edge.source, target=constraint_edge.target),
        constraint_rationale=rationale,
        steps=steps,
        projected_throughput_impact_pct=projected_throughput_impact_pct,
        next_constraint=None,
    )
