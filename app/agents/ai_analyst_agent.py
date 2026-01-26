from __future__ import annotations

from typing import Any, Dict

from app.agents.models import StatsRequest, StatsResult
from app.services.stats_service import run_stats


class AIAnalystAgent:
    """Select statistical analyses and execute them through the stats service."""

    def select_analysis(self, request: StatsRequest) -> str:
        if request.analysis:
            return request.analysis

        question = (request.question or "").lower()
        if "correlation" in question:
            return "correlation"
        if "anova" in question:
            return "anova_oneway"
        if "t-test" in question or "ttest" in question:
            return "ttest_2samp"
        if "regression" in question:
            return "regression_ols"
        if "trend" in question or "time series" in question:
            return "trend_analysis"

        return "descriptives"

    async def run(
        self,
        user_id: str,
        dataset_id: str,
        request: StatsRequest,
    ) -> StatsResult:
        analysis = self.select_analysis(request)
        params: Dict[str, Any] = request.params
        result, cached = await run_stats(
            user_id,
            dataset_id,
            analysis,
            params,
            where=request.where,
            pipeline_id=request.pipeline_id,
        )

        return StatsResult(
            analysis=analysis,
            params=params,
            result=result,
            cached=cached,
        )
