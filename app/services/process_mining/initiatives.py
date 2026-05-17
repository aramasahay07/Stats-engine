from __future__ import annotations

from typing import Iterable, List, Optional

from app.models.process_mining import Initiative, InitiativeEdge


def build_initiatives(
    edge_annotations: Iterable[dict],
    cost_per_hour: Optional[float] = None,
) -> List[Initiative]:
    initiatives: list[Initiative] = []
    effort_map = {
        "compress": "Medium",
        "parallelize": "High",
        "remove": "Low",
        "standardize": "Medium",
    }

    for idx, annotation in enumerate(edge_annotations, start=1):
        action = annotation.get("action") or "compress"
        source = str(annotation.get("source") or "")
        target = str(annotation.get("target") or "")
        hours_saved_per_case = float(annotation.get("hours_saved_per_case") or 0.0)
        cases_affected = int(annotation.get("cases_affected") or 0)
        total_hours_saved = float(hours_saved_per_case * cases_affected)
        cost_savings = total_hours_saved * float(cost_per_hour) if cost_per_hour is not None else None

        initiatives.append(
            Initiative(
                id=f"initiative-{idx}",
                title=str(annotation.get("title") or f"{action.title()} {source} to {target}"),
                edge=InitiativeEdge(source=source, target=target),
                action=action,  # type: ignore[arg-type]
                hours_saved_per_case=hours_saved_per_case,
                cases_affected=cases_affected,
                total_hours_saved=total_hours_saved,
                cost_savings=cost_savings,
                effort=effort_map.get(action, "Medium"),  # type: ignore[arg-type]
                rationale=str(annotation.get("rationale") or ""),
            )
        )

    initiatives.sort(key=lambda item: (-item.total_hours_saved, item.id))
    return initiatives
