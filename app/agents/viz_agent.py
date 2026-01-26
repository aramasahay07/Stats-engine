from __future__ import annotations

from typing import Any, Dict

from app.agents.models import ChartRequest, ChartSpec


class VizAgent:
    """Create Vega-Lite chart specifications for visualization."""

    def build_spec(self, request: ChartRequest) -> ChartSpec:
        mark = self._resolve_mark(request.chart_type)
        encoding: Dict[str, Any] = {}

        if request.x:
            encoding["x"] = {"field": request.x, "type": "quantitative"}
        if request.y:
            encoding["y"] = {"field": request.y, "type": "quantitative"}
        if request.color:
            encoding["color"] = {"field": request.color, "type": "nominal"}

        spec: Dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": mark,
            "encoding": encoding,
        }

        if request.data is not None:
            spec["data"] = {"values": request.data}

        if request.title:
            spec["title"] = request.title

        return ChartSpec(title=request.title, spec=spec)

    def _resolve_mark(self, chart_type: str) -> Dict[str, Any]:
        chart_type = chart_type.lower()
        if chart_type in {"scatter", "point"}:
            return {"type": "point", "tooltip": True}
        if chart_type in {"line", "timeseries"}:
            return {"type": "line", "tooltip": True}
        if chart_type in {"bar", "column"}:
            return {"type": "bar", "tooltip": True}
        return {"type": chart_type, "tooltip": True}
