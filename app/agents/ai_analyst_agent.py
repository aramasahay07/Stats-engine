from __future__ import annotations

from typing import Any, Dict, Optional

from app.agents.models import StatsRequest, StatsResult
from app.services.stats_service import run_stats


# -----------------------------------------------------------------------------
# Slug allowlist (loaded once) so the agent never returns a non-existent concept
# -----------------------------------------------------------------------------
_META_BY_SLUG: Optional[Dict[str, Any]] = None


def _get_meta_by_slug() -> Dict[str, Any]:
    """
    Returns {slug -> ConceptMeta} from the backend concepts registry.
    Cached for process lifetime.
    """
    global _META_BY_SLUG
    if _META_BY_SLUG is None:
        from app.analytics.concepts.registry import meta_by_slug

        _META_BY_SLUG = meta_by_slug()
    return _META_BY_SLUG


def _is_supported_slug(slug: str) -> bool:
    return slug in _get_meta_by_slug()


class AIAnalystAgent:
    """
    Select statistical analyses as BACKEND CONCEPT SLUGS (kebab-case) and execute
    them through the stats service.

    Contract:
      - If request.analysis is provided: treat it as authoritative (should be a slug).
      - If request.analysis is missing: choose a safe concept slug based on question + params.
      - Never return legacy snake_case names (e.g., ttest_2samp, anova_oneway, trend_analysis).
    """

    def select_analysis(self, request: StatsRequest) -> str:
        # 0) If caller already provided analysis, do NOT override.
        #    Assume it's already a concept slug.
        if request.analysis:
            return request.analysis

        q = (request.question or "").strip().lower()
        p = request.params or {}

        def has_any(words: list[str]) -> bool:
            return any(w in q for w in words)

        # ---------------------------------------------------------------------
        # 1) If params contain strong hints, prioritize those (most reliable)
        # ---------------------------------------------------------------------
        # Correlation: x + y present
        if (p.get("x") and p.get("y")) or (p.get("x_column") and p.get("y_column")):
            method = str(p.get("method", "")).lower()
            if "spearman" in q or method == "spearman":
                return "spearman-correlation"
            if "kendall" in q or method == "kendall":
                return "kendall-tau"
            return "pearson-correlation"

        # Group comparison: group_column + value_column present
        group_col = p.get("group_column") or p.get("group_col") or p.get("group")
        value_col = (
            p.get("value_column")
            or p.get("value_col")
            or p.get("column")
            or p.get("measure_column")
        )
        if group_col and value_col:
            # Nonparametric explicit
            if has_any(["kruskal"]):
                return "kruskal-wallis"
            if has_any(["mann whitney", "mann-whitney"]):
                return "mann-whitney-u"
            if has_any(["wilcoxon"]):
                return "wilcoxon-signed-rank"

            if "anova" in q:
                return "anova-one-way"

            # Default group comparison
            return "two-sample-t-test"

        # Paired data: before/after present
        if (p.get("before_column") or p.get("before")) and (p.get("after_column") or p.get("after")):
            if has_any(["wilcoxon"]):
                return "wilcoxon-signed-rank"
            return "paired-t-test"

        # Time series: time + value present
        if (p.get("time_column") or p.get("time_col")) and (
            p.get("value_column") or p.get("observed_col") or p.get("value_col")
        ):
            if has_any(["moving average", "moving-average"]) or str(p.get("method", "")).lower() == "moving_average":
                return "moving-averages"
            return "trend-analysis"

        # ---------------------------------------------------------------------
        # 2) Strong explicit keyword requests (even without params)
        # ---------------------------------------------------------------------
        if has_any(["correlation matrix", "corr matrix"]):
            return "correlation-matrix"

        if has_any(["kruskal"]):
            return "kruskal-wallis"
        if has_any(["mann whitney", "mann-whitney"]):
            return "mann-whitney-u"
        if has_any(["wilcoxon"]):
            return "wilcoxon-signed-rank"

        if has_any(["chi square", "chi-square", "chisquare"]):
            # Independence vs Goodness-of-fit heuristic
            if has_any(["goodness", "goodness-of-fit", "expected", "distribution", "fit to"]):
                return "chi-square-goodness-of-fit"
            return "chi-square-independence"

        if "anova" in q:
            return "anova-one-way"

        if has_any(["t-test", "t test", "ttest"]):
            if has_any(["paired", "before and after", "before/after", "pre and post", "pre/post"]):
                return "paired-t-test"
            if has_any(["one-sample", "one sample", "vs 0", "vs zero", "target mean"]):
                return "one-sample-t-test"
            return "two-sample-t-test"

        if "kendall" in q:
            return "kendall-tau"
        if "spearman" in q:
            return "spearman-correlation"
        if has_any(["pearson", "correlation", "relationship", "associated", "association"]):
            return "pearson-correlation"

        if has_any(["moving average", "moving-average"]):
            return "moving-averages"
        if has_any(["trend", "over time", "time series", "timeseries"]):
            return "trend-analysis"

        if has_any(["ols", "regression", "predict", "model", "explain", "drivers"]):
            return "regression-ols"

        # ---------------------------------------------------------------------
        # 3) “Insights” / vague requests -> choose a safe default bundle concept
        # ---------------------------------------------------------------------
        if has_any([
            "insight", "insights", "summary", "summarize", "overview", "high level",
            "key takeaways", "what stands out", "anything interesting", "explore", "eda",
            "analyze my data", "analyze this", "tell me about", "what can you tell me",
            "profile", "data profiling", "quality check", "data quality",
            "describe", "distribution", "summary stats", "basic stats", "report"
        ]):
            return "detailed-descriptives"

        # ---------------------------------------------------------------------
        # 4) Simple descriptives fallbacks
        # ---------------------------------------------------------------------
        if "mean" in q or "average" in q:
            return "mean"
        if "median" in q:
            return "median"
        if "standard deviation" in q or "std" in q:
            return "standard-deviation"
        if "variance" in q:
            return "variance"
        if "iqr" in q or "interquartile" in q:
            return "interquartile-range"
        if "outlier" in q or "outliers" in q or "anomaly" in q:
            return "outliers"

        # Default: best general-purpose summary concept
        return "detailed-descriptives"

    async def run(
        self,
        user_id: str,
        dataset_id: str,
        request: StatsRequest,
    ) -> StatsResult:
        analysis = self.select_analysis(request)

        # Guardrail: if the agent chose a slug that doesn't exist, fall back safely.
        # (This should rarely happen if your concepts folder is complete.)
        if not _is_supported_slug(analysis):
            analysis = "detailed-descriptives"

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
