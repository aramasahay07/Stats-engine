from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.agents.models import ImprovementPlanResponse, ImprovementRequest
from app.agents.qa_agent import QAAgent
from app.analytics.concepts.registry import meta_by_slug
from app.db import registry
from app.engine.duckdb_engine import DuckDBEngine
from app.engine.pipeline import compile_pipeline_sql
from app.services.spc_service import run_spc
from app.services.stats_service import _get_parquet_local
from app.transformers.registry import transformer_registry


_NEGATIVE_GOAL_WORDS = {
    "reduce", "decrease", "lower", "cut", "eliminate", "minimize",
    "defect", "scrap", "rework", "downtime", "delay", "cycle", "lead",
    "wait", "variation", "waste", "complaint", "cost",
}
_POSITIVE_GOAL_WORDS = {
    "increase", "improve", "raise", "boost", "grow", "maximize",
    "yield", "throughput", "output", "uptime", "efficiency", "oee",
    "quality", "service", "delivery", "fill rate",
}
_STABILITY_WORDS = {
    "stabilize", "stable", "control", "variation", "drift", "consistency",
    "special cause", "common cause", "noise", "spread",
}
_TIME_HINTS = ("date", "time", "timestamp", "day", "week", "month", "year", "shift_date")
_DIMENSION_HINTS = (
    "line", "machine", "shift", "product", "sku", "part", "operator", "supplier",
    "cell", "team", "area", "site", "family", "reason", "defect", "category",
    "department", "workcenter", "work_center", "station",
)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


class ImprovementGuidanceAgent:
    async def run(self, user_id: str, dataset_id: str, request: ImprovementRequest, profile: Dict[str, Any]) -> ImprovementPlanResponse:
        schema = profile.get("schema") or []
        sample_rows = profile.get("sample_rows") or []
        numeric_summary = profile.get("numeric_summary") or {}

        time_column = request.time_column or self._infer_time_column(schema)
        focus_metric = request.focus_metric or self._infer_focus_metric(request.question, schema, numeric_summary)
        direction = request.target_direction or self._infer_direction(request.question, focus_metric)
        group_columns = self._infer_group_columns(request.group_columns, schema, sample_rows, exclude={focus_metric, time_column, request.subgroup_column})

        if not focus_metric and not request.defectives_column and not request.defects_column:
            raise ValueError("Could not identify a primary metric. Provide focus_metric or attribute chart columns.")

        evidence = await self._build_evidence(
            user_id=user_id,
            dataset_id=dataset_id,
            request=request,
            profile=profile,
            focus_metric=focus_metric,
            direction=direction,
            time_column=time_column,
            group_columns=group_columns,
        )

        problem_definition = self._build_problem_definition(request, evidence, direction, time_column)
        smart_aim = self._build_smart_aim(request, evidence, direction)
        selected_metrics = self._build_selected_metrics(evidence)
        insights = self._build_insights(evidence, direction)
        recommended_analyses = self._build_recommended_analyses(evidence)
        root_cause_hypotheses = self._build_hypotheses(evidence, direction)
        experiments = self._build_experiments(evidence, direction)
        action_plan = self._build_action_plan(evidence, direction)
        sustainment_plan = self._build_sustainment_plan(evidence, direction, time_column)
        workbooks = self._build_workbooks(request, evidence, problem_definition, smart_aim, root_cause_hypotheses, experiments, sustainment_plan)
        summary = self._build_summary(problem_definition, smart_aim, insights)

        payload = {
            "dataset_id": dataset_id,
            "question": request.question,
            "process_name": request.process_name,
            "focus_metric": evidence.get("focus_metric"),
            "summary": summary,
            "problem_definition": problem_definition,
            "smart_aim": smart_aim,
            "baseline": evidence.get("baseline", {}),
            "selected_metrics": selected_metrics,
            "insights": insights,
            "recommended_analyses": recommended_analyses,
            "charts": evidence.get("charts", []),
            "root_cause_hypotheses": root_cause_hypotheses,
            "experiments": experiments,
            "action_plan": action_plan,
            "sustainment_plan": sustainment_plan,
            "workbooks": workbooks,
        }

        qa = QAAgent().validate_response(
            payload,
            required_fields=[
                "dataset_id",
                "summary",
                "problem_definition",
                "smart_aim",
                "baseline",
                "insights",
                "experiments",
                "sustainment_plan",
                "workbooks",
            ],
        )
        if not qa.valid:
            raise RuntimeError(f"Improvement plan QA failed: {[issue.message for issue in qa.issues]}")

        return ImprovementPlanResponse(**payload, qa=qa)

    async def _build_evidence(
        self,
        user_id: str,
        dataset_id: str,
        request: ImprovementRequest,
        profile: Dict[str, Any],
        focus_metric: Optional[str],
        direction: str,
        time_column: Optional[str],
        group_columns: List[str],
    ) -> Dict[str, Any]:
        schema = profile.get("schema") or []
        charts: List[Dict[str, Any]] = []
        insights: List[Dict[str, Any]] = []
        correlations: List[Dict[str, Any]] = []
        stratification: List[Dict[str, Any]] = []
        baseline: Dict[str, Any] = {}
        metric_label = focus_metric or request.defectives_column or request.defects_column

        if focus_metric:
            columns = [focus_metric]
            if time_column:
                columns.append(time_column)
            if request.subgroup_column and request.subgroup_column not in columns:
                columns.append(request.subgroup_column)
            for col in group_columns[:3]:
                if col and col not in columns:
                    columns.append(col)
            for col in self._select_numeric_driver_columns(schema, focus_metric):
                if col not in columns:
                    columns.append(col)

            df = await self._load_dataset_frame(
                user_id=user_id,
                dataset_id=dataset_id,
                columns=columns,
                where=request.where,
                pipeline_id=request.pipeline_id,
                limit=request.limit,
                order_by=time_column,
            )
            if df.empty:
                raise ValueError("No rows available after filters for improvement analysis")

            metric_series = pd.to_numeric(df[focus_metric], errors="coerce")
            if time_column:
                ordered_df = df.assign(__metric=metric_series).dropna(subset=["__metric"]).reset_index(drop=True)
                labels = ordered_df[time_column].astype(str).tolist()
                values = ordered_df["__metric"].to_numpy(dtype=float)
            else:
                ordered_df = df.assign(__metric=metric_series).dropna(subset=["__metric"]).reset_index(drop=True)
                labels = [str(i + 1) for i in range(len(ordered_df))]
                values = ordered_df["__metric"].to_numpy(dtype=float)

            if len(values) < 2:
                raise ValueError(f'Focus metric "{focus_metric}" has fewer than 2 numeric values after filtering')

            baseline = self._compute_numeric_baseline(values, labels, focus_metric, direction)
            charts.append(
                {
                    "chart_type": "run-chart",
                    "title": f"Run Chart for {focus_metric}",
                    "purpose": "Confirm baseline trend, shifts, and recent performance.",
                    "data": {
                        "points": [{"index": i + 1, "label": labels[i], "value": float(values[i])} for i in range(len(values))],
                        "centerLine": baseline["mean"],
                    },
                }
            )

            if time_column and len(values) >= 8:
                spc = await run_spc(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    chart_type="i-mr",
                    value_column=focus_metric,
                    time_column=time_column,
                    subgroup_column=request.subgroup_column,
                    limit=request.limit,
                    where=request.where,
                    pipeline_id=request.pipeline_id,
                )
                charts.append(
                    {
                        "chart_type": "i-mr",
                        "title": f"I-MR Control Chart for {focus_metric}",
                        "purpose": "Distinguish common-cause variation from special-cause signals.",
                        "data": spc,
                    }
                )
                rule_hits = sum((spc.get("meta", {}).get("rules_summary") or {}).values())
                if rule_hits:
                    insights.append(
                        {
                            "type": "stability",
                            "severity": "high",
                            "title": "Process is showing special-cause signals",
                            "evidence": f"{rule_hits} Nelson rule hits were detected on the I-MR chart.",
                            "recommended_action": "Investigate the time windows around flagged points before changing standard settings.",
                        }
                    )
                else:
                    insights.append(
                        {
                            "type": "stability",
                            "severity": "medium",
                            "title": "Process appears statistically stable",
                            "evidence": "No Nelson rule violations were found on the I-MR chart.",
                            "recommended_action": "Focus next on capability, centering, and stratified losses rather than special causes.",
                        }
                    )

            if len(values) >= 8:
                ewma = await run_spc(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    chart_type="ewma",
                    value_column=focus_metric,
                    time_column=time_column,
                    limit=request.limit,
                    where=request.where,
                    pipeline_id=request.pipeline_id,
                    lambda_param=request.lambda_param,
                    target=request.target,
                    sigma=request.sigma,
                )
                charts.append(
                    {
                        "chart_type": "ewma",
                        "title": f"EWMA for {focus_metric}",
                        "purpose": "Detect smaller sustained shifts that may not be obvious on a standard run chart.",
                        "data": ewma,
                    }
                )

                cusum = await run_spc(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    chart_type="cusum",
                    value_column=focus_metric,
                    time_column=time_column,
                    limit=request.limit,
                    where=request.where,
                    pipeline_id=request.pipeline_id,
                    target=request.target,
                    sigma=request.sigma,
                    k=request.k,
                    h=request.h,
                )
                charts.append(
                    {
                        "chart_type": "cusum",
                        "title": f"CUSUM for {focus_metric}",
                        "purpose": "Detect sustained directional drift against the target faster than periodic averages.",
                        "data": cusum,
                    }
                )

            stratification = self._compute_stratification(ordered_df, focus_metric, group_columns[:3], direction)
            for item in stratification[:2]:
                charts.append(
                    {
                        "chart_type": "pareto",
                        "title": f"Pareto by {item['dimension']}",
                        "purpose": f"Prioritize the biggest contributors by {item['dimension']}.",
                        "data": item["chart"],
                    }
                )

            correlations = self._compute_correlations(ordered_df, focus_metric)

        attribute_chart = await self._maybe_build_attribute_chart(
            user_id=user_id,
            dataset_id=dataset_id,
            request=request,
        )
        if attribute_chart:
            charts.append(attribute_chart)
            insights.extend(attribute_chart.get("auto_insights", []))
            if not baseline:
                primary = (attribute_chart.get("data") or {}).get("primary") or {}
                point_count = len(primary.get("points") or [])
                baseline = {
                    "metric": metric_label,
                    "direction": direction,
                    "n_observations": point_count,
                    "mean": _safe_float(primary.get("centerLine")),
                    "latest": _safe_float((primary.get("points") or [{}])[-1].get("value")) if point_count else None,
                    "std_dev": None,
                }

        baseline.setdefault("n_rows_profiled", profile.get("n_rows"))
        baseline.setdefault("n_columns_profiled", profile.get("n_cols"))

        return {
            "focus_metric": metric_label,
            "direction": direction,
            "time_column": time_column,
            "group_columns": group_columns,
            "baseline": baseline,
            "charts": charts,
            "insights_seed": insights,
            "stratification": stratification,
            "correlations": correlations,
            "question": request.question,
            "profile": profile,
        }

    def _infer_time_column(self, schema: Sequence[Dict[str, Any]]) -> Optional[str]:
        datetime_cols = [col["name"] for col in schema if str(col.get("role")).lower() == "datetime" and col.get("name")]
        if datetime_cols:
            return datetime_cols[0]
        for col in schema:
            name = str(col.get("name") or "").lower()
            if any(hint in name for hint in _TIME_HINTS):
                return col.get("name")
        return None

    def _infer_focus_metric(self, question: str, schema: Sequence[Dict[str, Any]], numeric_summary: Dict[str, Any]) -> Optional[str]:
        numeric_cols = [col.get("name") for col in schema if str(col.get("role")).lower() == "numeric" and col.get("name")]
        if not numeric_cols:
            return None

        q = (question or "").lower()
        tokens = [t for t in re.split(r"[^a-z0-9_]+", q) if t]
        scored: List[Tuple[int, str]] = []
        for name in numeric_cols:
            lname = str(name).lower()
            score = 0
            if lname in q:
                score += 5
            for token in tokens:
                if token and token in lname:
                    score += 2
            if any(k in lname for k in ("defect", "scrap", "yield", "oee", "cycle", "lead", "downtime", "throughput", "cost")):
                score += 1
            summary = numeric_summary.get(name) or {}
            if summary.get("n_non_null"):
                score += 1
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        return scored[0][1] if scored else numeric_cols[0]

    def _infer_direction(self, question: str, metric: Optional[str]) -> str:
        q = (question or "").lower()
        name = (metric or "").lower()
        if any(word in q for word in _STABILITY_WORDS):
            return "stabilize"
        if any(word in q for word in _NEGATIVE_GOAL_WORDS) or any(word in name for word in ("defect", "scrap", "rework", "downtime", "cycle", "lead", "cost")):
            return "reduce"
        if any(word in q for word in _POSITIVE_GOAL_WORDS) or any(word in name for word in ("yield", "throughput", "oee", "uptime", "quality")):
            return "increase"
        return "improve"

    def _infer_group_columns(
        self,
        requested: Sequence[str],
        schema: Sequence[Dict[str, Any]],
        sample_rows: Sequence[Dict[str, Any]],
        exclude: set[Any],
    ) -> List[str]:
        if requested:
            return [col for col in requested if col and col not in exclude]

        sample_cardinality: Dict[str, int] = {}
        for row in sample_rows[:25]:
            for key, value in row.items():
                sample_cardinality.setdefault(key, set()).add(str(value))
        sample_cardinality_counts = {k: len(v) for k, v in sample_cardinality.items()}

        candidates: List[Tuple[int, str]] = []
        for col in schema:
            name = col.get("name")
            if not name or name in exclude:
                continue
            role = str(col.get("role") or "").lower()
            if role != "categorical":
                continue
            lname = str(name).lower()
            score = 0
            if any(hint in lname for hint in _DIMENSION_HINTS):
                score += 4
            cardinality = sample_cardinality_counts.get(name, 0)
            if 1 < cardinality <= 12:
                score += 2
            elif 12 < cardinality <= 25:
                score += 1
            candidates.append((score, name))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [name for score, name in candidates[:3] if score > 0]

    def _select_numeric_driver_columns(self, schema: Sequence[Dict[str, Any]], focus_metric: str) -> List[str]:
        cols = []
        for col in schema:
            name = col.get("name")
            if not name or name == focus_metric:
                continue
            if str(col.get("role")).lower() == "numeric":
                cols.append(name)
        return cols[:6]

    async def _load_pipeline_steps(self, user_id: str, dataset_id: str, pipeline_id: str) -> List[Dict[str, Any]]:
        row = await registry.fetchrow(
            """
            SELECT steps_json
            FROM pipelines
            WHERE id = $1::uuid
              AND user_id = $2
              AND dataset_id = $3::uuid
            """,
            pipeline_id,
            user_id,
            dataset_id,
        )
        if not row:
            raise ValueError("Pipeline not found (or not owned by user / wrong dataset)")

        steps = row["steps_json"]
        if isinstance(steps, str):
            try:
                steps = json.loads(steps)
            except Exception:
                steps = []
        if not isinstance(steps, list):
            raise ValueError("Pipeline steps are invalid")
        for step in steps:
            op = (step or {}).get("op")
            if not op:
                raise ValueError("Pipeline step missing 'op'")
            transformer_registry.get(op)
        return steps

    async def _load_dataset_frame(
        self,
        user_id: str,
        dataset_id: str,
        columns: List[str],
        where: Optional[str],
        pipeline_id: Optional[str],
        limit: int,
        order_by: Optional[str] = None,
    ) -> pd.DataFrame:
        parquet = await _get_parquet_local(user_id, dataset_id)
        eng = DuckDBEngine(user_id)
        con = eng.connect()
        try:
            view = eng.register_parquet(con, dataset_id, parquet)
            con.execute(f"CREATE OR REPLACE VIEW dataset_base AS SELECT * FROM {view}")
            current_view = "dataset_base"
            if where and str(where).strip():
                con.execute(
                    f"CREATE OR REPLACE VIEW dataset_filtered AS "
                    f"SELECT * FROM {current_view} WHERE {where}"
                )
                current_view = "dataset_filtered"
            if pipeline_id:
                steps = await self._load_pipeline_steps(user_id, dataset_id, pipeline_id)
                pipeline_sql = compile_pipeline_sql(base_view=current_view, steps=steps)
                con.execute(f"CREATE OR REPLACE VIEW dataset_piped AS {pipeline_sql}")
                current_view = "dataset_piped"
            con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM {current_view}")

            unique_columns = []
            seen = set()
            for col in columns:
                if col and col not in seen:
                    unique_columns.append(col)
                    seen.add(col)
            select_sql = ", ".join(_quote_ident(col) for col in unique_columns)
            order_clause = f"ORDER BY {_quote_ident(order_by)}" if order_by else ""
            return con.execute(
                f"""
                SELECT {select_sql}
                FROM dataset
                {order_clause}
                LIMIT {int(limit)}
                """
            ).fetchdf()
        finally:
            con.close()

    def _compute_numeric_baseline(self, values: np.ndarray, labels: List[str], focus_metric: str, direction: str) -> Dict[str, Any]:
        arr = np.asarray(values, dtype=float)
        slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) >= 3 else 0.0
        first_window = arr[: max(2, len(arr) // 5)]
        last_window = arr[-max(2, len(arr) // 5):]
        return {
            "metric": focus_metric,
            "direction": direction,
            "n_observations": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "latest": float(arr[-1]),
            "trend_slope_per_row": slope,
            "first_window_mean": float(np.mean(first_window)),
            "last_window_mean": float(np.mean(last_window)),
            "change_last_vs_first_pct": float(((np.mean(last_window) - np.mean(first_window)) / np.mean(first_window)) * 100.0) if np.mean(first_window) else 0.0,
            "time_start": labels[0] if labels else None,
            "time_end": labels[-1] if labels else None,
        }

    def _compute_stratification(self, df: pd.DataFrame, focus_metric: str, group_columns: Sequence[str], direction: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if df.empty:
            return out
        metric = pd.to_numeric(df[focus_metric], errors="coerce")
        temp = df.copy()
        temp["__metric"] = metric
        temp = temp.dropna(subset=["__metric"])
        if temp.empty:
            return out

        for dim in group_columns:
            if dim not in temp.columns:
                continue
            grouped = (
                temp.groupby(dim, dropna=False)["__metric"]
                .agg(["count", "mean", "median", "std", "sum"])
                .reset_index()
            )
            if grouped.empty or len(grouped) < 2:
                continue
            grouped = grouped.sort_values("sum" if direction == "reduce" else "mean", ascending=False).reset_index(drop=True)
            top = grouped.iloc[0]
            bottom = grouped.iloc[-1]
            spread_ratio = float(top["mean"] / bottom["mean"]) if bottom["mean"] not in (0, None) else None
            out.append(
                {
                    "dimension": dim,
                    "top_group": str(top[dim]),
                    "top_mean": float(top["mean"]),
                    "top_total": float(top["sum"]),
                    "bottom_group": str(bottom[dim]),
                    "bottom_mean": float(bottom["mean"]),
                    "spread_ratio": spread_ratio,
                    "chart": {
                        "bars": [
                            {
                                "label": str(row[dim]),
                                "mean": float(row["mean"]),
                                "count": int(row["count"]),
                                "total": float(row["sum"]),
                            }
                            for _, row in grouped.head(8).iterrows()
                        ]
                    },
                }
            )
        return out

    def _compute_correlations(self, df: pd.DataFrame, focus_metric: str) -> List[Dict[str, Any]]:
        numeric_df = df.copy()
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")
        if focus_metric not in numeric_df.columns:
            return []
        corr = numeric_df.corr(numeric_only=True).get(focus_metric)
        if corr is None:
            return []
        results: List[Dict[str, Any]] = []
        for name, value in corr.items():
            if name == focus_metric or pd.isna(value):
                continue
            if abs(float(value)) < 0.2:
                continue
            results.append(
                {
                    "column": name,
                    "correlation": float(value),
                    "strength": self._describe_correlation(float(value)),
                }
            )
        results.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        return results[:5]

    def _describe_correlation(self, value: float) -> str:
        abs_v = abs(value)
        if abs_v >= 0.7:
            return "strong"
        if abs_v >= 0.4:
            return "moderate"
        return "weak"

    async def _maybe_build_attribute_chart(self, user_id: str, dataset_id: str, request: ImprovementRequest) -> Optional[Dict[str, Any]]:
        try:
            if request.defectives_column and request.sample_size_column:
                result = await run_spc(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    chart_type="p",
                    subgroup_column=request.subgroup_column,
                    time_column=request.time_column,
                    limit=request.limit,
                    where=request.where,
                    pipeline_id=request.pipeline_id,
                    defectives_column=request.defectives_column,
                    sample_size_column=request.sample_size_column,
                )
                return {
                    "chart_type": "p",
                    "title": f"P Chart for {request.defectives_column}",
                    "purpose": "Track fraction defective over time or subgroup.",
                    "data": result,
                    "auto_insights": self._attribute_chart_insights(result, "proportion defective"),
                }
            if request.defectives_column and request.sample_size:
                result = await run_spc(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    chart_type="np",
                    subgroup_column=request.subgroup_column,
                    time_column=request.time_column,
                    limit=request.limit,
                    where=request.where,
                    pipeline_id=request.pipeline_id,
                    defectives_column=request.defectives_column,
                    sample_size=request.sample_size,
                )
                return {
                    "chart_type": "np",
                    "title": f"NP Chart for {request.defectives_column}",
                    "purpose": "Track count defective when each subgroup has the same sample size.",
                    "data": result,
                    "auto_insights": self._attribute_chart_insights(result, "number defective"),
                }
            if request.defects_column and request.area_column:
                result = await run_spc(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    chart_type="u",
                    subgroup_column=request.subgroup_column,
                    time_column=request.time_column,
                    limit=request.limit,
                    where=request.where,
                    pipeline_id=request.pipeline_id,
                    defects_column=request.defects_column,
                    area_column=request.area_column,
                )
                return {
                    "chart_type": "u",
                    "title": f"U Chart for {request.defects_column}",
                    "purpose": "Track defects per unit when opportunity count varies.",
                    "data": result,
                    "auto_insights": self._attribute_chart_insights(result, "defects per unit"),
                }
            if request.defects_column:
                result = await run_spc(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    chart_type="c",
                    subgroup_column=request.subgroup_column,
                    time_column=request.time_column,
                    limit=request.limit,
                    where=request.where,
                    pipeline_id=request.pipeline_id,
                    defects_column=request.defects_column,
                )
                return {
                    "chart_type": "c",
                    "title": f"C Chart for {request.defects_column}",
                    "purpose": "Track count of defects per constant inspection unit.",
                    "data": result,
                    "auto_insights": self._attribute_chart_insights(result, "defect count"),
                }
        except Exception:
            return None
        return None

    def _attribute_chart_insights(self, chart: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
        primary = chart.get("primary") or {}
        points = primary.get("points") or []
        flagged = sum(1 for point in points if point.get("violations"))
        if not flagged:
            return [
                {
                    "type": "attribute_stability",
                    "severity": "medium",
                    "title": f"{label.capitalize()} appears stable on the current control chart",
                    "evidence": "No special-cause points were flagged on the attribute chart.",
                    "recommended_action": "Focus on average level reduction through designed changes rather than firefighting individual dates.",
                }
            ]
        return [
            {
                "type": "attribute_stability",
                "severity": "high",
                "title": f"{label.capitalize()} is unstable",
                "evidence": f"{flagged} plotted points show special-cause signals on the attribute chart.",
                "recommended_action": "Lock down when the spikes happened, stratify by line/shift/product, and confirm the trigger before scaling fixes.",
            }
        ]

    def _build_problem_definition(self, request: ImprovementRequest, evidence: Dict[str, Any], direction: str, time_column: Optional[str]) -> Dict[str, Any]:
        baseline = evidence.get("baseline", {})
        metric = evidence.get("focus_metric")
        current = baseline.get("mean") or baseline.get("center_line") or baseline.get("latest")
        variation = baseline.get("std_dev")
        scope = f"Filtered dataset scope for {metric}" if request.where else f"All available rows for {metric}"
        statement = (
            f"The current process performance for {metric} is centered around {current:.3f} "
            f"with variation of {variation:.3f} across {baseline.get('n_observations', baseline.get('n_rows_profiled', 'n/a'))} observations."
            if current is not None and variation is not None
            else f"The current process issue is framed around {metric} using the uploaded dataset."
        )
        if direction == "reduce":
            impact = f"High {metric} is creating waste, delay, or quality loss until the process is brought down and stabilized."
        elif direction == "increase":
            impact = f"Low {metric} is limiting throughput, yield, or customer value until the process is raised and stabilized."
        elif direction == "improve":
            impact = f"The current {metric} level is not meeting the desired business outcome and should be improved with verified countermeasures."
        else:
            impact = f"Instability in {metric} is making the process unpredictable and hard to sustain."
        return {
            "statement": statement,
            "business_impact": impact,
            "scope": scope,
            "time_basis": time_column,
            "voice_of_customer": request.business_context or "Use business context, customer impact, and cost/risk to refine this statement before kickoff.",
        }

    def _build_smart_aim(self, request: ImprovementRequest, evidence: Dict[str, Any], direction: str) -> Dict[str, Any]:
        baseline = evidence.get("baseline", {})
        metric = evidence.get("focus_metric")
        current = _safe_float(baseline.get("mean") or baseline.get("latest"))
        due = request.target_date or (datetime.now(timezone.utc) + timedelta(days=90)).date().isoformat()

        if request.target_value is not None:
            target_value = request.target_value
        elif current is not None and request.target_improvement_pct is not None:
            pct = request.target_improvement_pct / 100.0
            target_value = current * (1 - pct) if direction == "reduce" else current * (1 + pct)
        elif current is not None:
            default_pct = 0.20 if direction == "reduce" else 0.15
            if direction == "stabilize":
                target_value = current
            else:
                target_value = current * (1 - default_pct) if direction == "reduce" else current * (1 + default_pct)
        else:
            target_value = None

        if direction == "reduce":
            verb = "reduce"
        elif direction == "increase":
            verb = "increase"
        elif direction == "improve":
            verb = "improve"
        else:
            verb = "stabilize"

        statement = (
            f"By {due}, {verb} {metric} from {current:.3f} to {target_value:.3f} while maintaining process stability."
            if current is not None and target_value is not None
            else f"By {due}, {verb} the primary process metric using a verified baseline and control plan."
        )
        return {
            "statement": statement,
            "specific": f"Primary focus metric: {metric}",
            "measurable": {
                "current": current,
                "target": target_value,
                "due_date": due,
            },
            "achievable": "Start with the biggest stratified losses and validate causes with short experiments before scaling.",
            "relevant": request.business_context or "Tie the aim to customer quality, delivery, safety, or cost.",
            "time_bound": due,
        }

    def _build_selected_metrics(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        baseline = evidence.get("baseline", {})
        metrics = [
            {"name": evidence.get("focus_metric"), "role": "primary", "current": baseline.get("mean"), "latest": baseline.get("latest")},
            {"name": "variation", "role": "variation", "current": baseline.get("std_dev"), "latest": None},
            {"name": "observation_count", "role": "coverage", "current": baseline.get("n_observations"), "latest": None},
        ]
        for item in evidence.get("correlations", [])[:2]:
            metrics.append(
                {
                    "name": item["column"],
                    "role": "potential_driver",
                    "current": item["correlation"],
                    "latest": None,
                }
            )
        return metrics

    def _build_insights(self, evidence: Dict[str, Any], direction: str) -> List[Dict[str, Any]]:
        insights = list(evidence.get("insights_seed", []))
        baseline = evidence.get("baseline", {})
        change_pct = _safe_float(baseline.get("change_last_vs_first_pct"))
        if change_pct is not None:
            if direction == "reduce" and change_pct > 5:
                insights.append(
                    {
                        "type": "trend",
                        "severity": "high",
                        "title": "Recent performance is trending worse than the starting baseline",
                        "evidence": f"Last-window average is {change_pct:.1f}% higher than the first-window average.",
                        "recommended_action": "Treat the deterioration as a priority containment issue and verify what changed in the recent period.",
                    }
                )
            elif direction == "increase" and change_pct < -5:
                insights.append(
                    {
                        "type": "trend",
                        "severity": "high",
                        "title": "Recent performance is trending below the starting baseline",
                        "evidence": f"Last-window average is {change_pct:.1f}% lower than the first-window average.",
                        "recommended_action": "Check recent operating conditions, staffing, material, or setup changes that reduced output.",
                    }
                )

        for item in evidence.get("stratification", [])[:3]:
            ratio = _safe_float(item.get("spread_ratio"))
            if ratio and ratio >= 1.2:
                insights.append(
                    {
                        "type": "stratification",
                        "severity": "high",
                        "title": f"{item['dimension']} is a likely loss driver",
                        "evidence": f"{item['top_group']} averages {item['top_mean']:.3f} vs {item['bottom_group']} at {item['bottom_mean']:.3f}.",
                        "recommended_action": f"Compare standard work, inputs, and settings across {item['dimension']} before brainstorming broad fixes.",
                    }
                )

        for item in evidence.get("correlations", [])[:3]:
            insights.append(
                {
                    "type": "correlation",
                    "severity": "medium",
                    "title": f"{item['column']} has a {item['strength']} relationship with the focus metric",
                    "evidence": f"Correlation with {evidence.get('focus_metric')} is {item['correlation']:.2f}.",
                    "recommended_action": f"Use {item['column']} as a likely experimental factor or stratification variable; verify causality before changing the process.",
                }
            )

        if not insights:
            insights.append(
                {
                    "type": "baseline",
                    "severity": "medium",
                    "title": "Baseline established; next step is focused stratification",
                    "evidence": "The dataset supports a measured improvement plan even if one dominant signal is not yet obvious.",
                    "recommended_action": "Use the top business dimension, a control chart, and a short PDSA cycle to narrow the likely cause set.",
                }
            )
        return insights

    def _build_recommended_analyses(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        concepts = meta_by_slug()

        def pack(slug: str, why: str) -> Dict[str, Any]:
            meta = concepts.get(slug)
            return {
                "slug": slug,
                "title": getattr(meta, "title", slug),
                "why": why,
            }

        recs = [pack("detailed-descriptives", "Establish the baseline mean, spread, and tails before improvement work starts.")]
        if evidence.get("time_column"):
            recs.append(pack("imr-chart", "Check whether the process is stable or being distorted by special causes."))
            recs.append(pack("ewma-chart", "Detect smaller sustained shifts in the metric earlier."))
            recs.append(pack("cusum-chart", "Detect directional drift against the target or baseline."))
        if evidence.get("stratification"):
            recs.append(pack("pareto-analysis", "Prioritize which categories or sources contribute most to the loss."))
        if evidence.get("correlations"):
            recs.append(pack("pearson-correlation", "Screen likely numeric drivers before running experiments."))
            recs.append(pack("regression-ols", "Quantify directional relationships once the likely drivers are narrowed."))
        return recs

    def _build_hypotheses(self, evidence: Dict[str, Any], direction: str) -> List[Dict[str, Any]]:
        hypotheses: List[Dict[str, Any]] = []
        for item in evidence.get("stratification", [])[:3]:
            hypotheses.append(
                {
                    "hypothesis": f"Performance differs materially by {item['dimension']}, especially in {item['top_group']}.",
                    "because": f"{item['top_group']} shows the highest average performance loss in the current baseline.",
                    "evidence_needed": [
                        f"Confirm process method differences between {item['top_group']} and {item['bottom_group']}.",
                        f"Check setup, staffing, material, and scheduling factors associated with {item['dimension']}.",
                    ],
                }
            )
        for item in evidence.get("correlations", [])[:2]:
            hypotheses.append(
                {
                    "hypothesis": f"Changes in {item['column']} are associated with changes in {evidence.get('focus_metric')}.",
                    "because": f"Observed correlation is {item['correlation']:.2f}.",
                    "evidence_needed": [
                        f"Validate whether {item['column']} changes before the metric changes.",
                        "Run a bounded experiment or before/after test to separate cause from coincidence.",
                    ],
                }
            )
        if not hypotheses:
            hypotheses.append(
                {
                    "hypothesis": "The current loss is driven by a mix of process instability, inconsistent standard work, and hidden stratification.",
                    "because": "That is the most common pattern when one clear factor has not yet emerged.",
                    "evidence_needed": [
                        "Review recent events around the worst points on the control chart.",
                        "Stratify by the most plausible operational dimension and compare method differences.",
                    ],
                }
            )
        return hypotheses

    def _build_experiments(self, evidence: Dict[str, Any], direction: str) -> List[Dict[str, Any]]:
        focus_metric = evidence.get("focus_metric")
        experiments: List[Dict[str, Any]] = []

        if evidence.get("stratification"):
            top = evidence["stratification"][0]
            experiments.append(
                {
                    "title": f"Best-vs-worst comparison on {top['dimension']}",
                    "objective": f"Confirm what operating conditions make {top['top_group']} worse than {top['bottom_group']}.",
                    "plan": [
                        f"Document the exact method, setup, staffing, material, and schedule for {top['top_group']} and {top['bottom_group']}.",
                        "Hold all other factors constant where possible.",
                        f"Run a short comparison and plot {focus_metric} before and after the change.",
                    ],
                    "success_metric": focus_metric,
                    "timebox": "1-2 weeks",
                }
            )

        if evidence.get("correlations"):
            driver = evidence["correlations"][0]
            experiments.append(
                {
                    "title": f"Factor trial for {driver['column']}",
                    "objective": f"Test whether changing {driver['column']} moves {focus_metric} in the desired direction.",
                    "plan": [
                        f"Choose 2-3 realistic settings or bands for {driver['column']}.",
                        "Run a controlled trial with the same product / line / crew where possible.",
                        "Use the same collection plan and plot the effect on the target metric.",
                    ],
                    "success_metric": focus_metric,
                    "timebox": "1 week pilot",
                }
            )

        experiments.append(
            {
                "title": "Containment and signal check",
                "objective": "Stabilize the process enough to make learning visible.",
                "plan": [
                    "Freeze non-essential process changes during the test window.",
                    "Track the main metric daily on a run chart or control chart.",
                    "Escalate any special-cause point immediately using a standard reaction plan.",
                ],
                "success_metric": focus_metric,
                "timebox": "Immediate / daily management",
            }
        )
        return experiments

    def _build_action_plan(self, evidence: Dict[str, Any], direction: str) -> List[Dict[str, Any]]:
        focus_metric = evidence.get("focus_metric")
        actions = [
            {
                "horizon": "0-7 days",
                "action": f"Lock the baseline for {focus_metric} and review the worst points or groups with process owners.",
                "owner": "Process owner / industrial engineer",
                "output": "Baseline pack, issue list, and confirmed problem statement",
            },
            {
                "horizon": "1-3 weeks",
                "action": "Run the top two experiments and document the tested factor, result, and learning.",
                "owner": "Improvement lead",
                "output": "PDSA results with evidence of gain or no-gain",
            },
            {
                "horizon": "3-8 weeks",
                "action": "Standardize the successful change across the affected area and add a control plan.",
                "owner": "Operations leader",
                "output": "Updated standard work, training sign-off, and active control chart review",
            },
        ]
        return actions

    def _build_sustainment_plan(self, evidence: Dict[str, Any], direction: str, time_column: Optional[str]) -> Dict[str, Any]:
        focus_metric = evidence.get("focus_metric")
        cadence = "daily" if time_column else "per shift / per batch"
        return {
            "control_plan": {
                "metric": focus_metric,
                "review_cadence": cadence,
                "chart_to_use": "i-mr" if time_column else "run-chart",
                "owner": "Area owner",
                "reaction_plan": [
                    "If a special-cause signal appears, stop and identify what changed before adjusting limits.",
                    "If the metric misses the target for 3 consecutive review periods, escalate to the improvement lead.",
                ],
            },
            "standard_work": [
                "Update SOP / standard work with the new best-known method.",
                "Train affected operators, leaders, and support roles on the changed method and trigger points.",
                "Audit adherence weekly for the first month, then monthly.",
            ],
            "layered_process_audits": [
                "Supervisor audit: daily confirmation of the key process checks.",
                "Manager audit: weekly review of chart signals, missed checks, and corrective actions.",
                "Engineer audit: monthly confirmation that the metric still reflects the real process risk.",
            ],
            "visual_management": {
                "board_fields": ["target", "actual", "gap", "special-cause flags", "countermeasure owner", "due date"],
                "meeting_structure": "Review yesterday, today’s risks, open experiments, and sustainment misses.",
            },
        }

    def _build_workbooks(
        self,
        request: ImprovementRequest,
        evidence: Dict[str, Any],
        problem_definition: Dict[str, Any],
        smart_aim: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        sustainment_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "dmaic": self._dmaic_template(request, evidence, problem_definition, smart_aim, hypotheses, experiments, sustainment_plan),
            "a3": self._a3_template(request, evidence, problem_definition, smart_aim, hypotheses, experiments, sustainment_plan),
            "pdsa": self._pdsa_template(request, evidence, smart_aim, experiments),
        }

    def _dmaic_template(
        self,
        request: ImprovementRequest,
        evidence: Dict[str, Any],
        problem_definition: Dict[str, Any],
        smart_aim: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        sustainment_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "instructions": "Use this as the working DMAIC charter. Fill the owner, due date, and evidence fields as the team learns.",
            "define": {
                "problem_statement": problem_definition["statement"],
                "business_impact": problem_definition["business_impact"],
                "project_scope": problem_definition["scope"],
                "customer_need_prompt": "What customer, safety, cost, or delivery pain does this metric create today?",
            },
            "measure": {
                "baseline": evidence.get("baseline", {}),
                "data_collection_checklist": [
                    "Confirm the metric definition and unit of measure.",
                    "Confirm time stamp / subgroup logic.",
                    "Confirm all data filters and pipeline steps used for the baseline.",
                    "Check whether missing data or unstable subgrouping could bias the chart.",
                ],
            },
            "analyze": {
                "top_hypotheses": hypotheses,
                "required_evidence_prompt": "What observation, stratification, or experiment would disprove each hypothesis?",
                "analysis_priority": [item["title"] for item in self._build_recommended_analyses(evidence)],
            },
            "improve": {
                "smart_aim": smart_aim,
                "experiments": experiments,
                "selection_rule": "Scale only the changes that improve the target metric without creating a counterproductive side effect.",
            },
            "control": {
                "sustainment_plan": sustainment_plan,
                "handoff_prompt": "What standard work, audit, and ownership must be in place before the project is considered complete?",
            },
        }

    def _a3_template(
        self,
        request: ImprovementRequest,
        evidence: Dict[str, Any],
        problem_definition: Dict[str, Any],
        smart_aim: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        sustainment_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "instructions": "Use this as a one-page A3 draft. Keep each section short, evidence-based, and updated as the team learns.",
            "background": request.business_context or "Describe the process, customer need, and why the current gap matters now.",
            "current_condition": evidence.get("baseline", {}),
            "problem_statement": problem_definition["statement"],
            "target_condition": smart_aim,
            "analysis": hypotheses,
            "countermeasures": experiments,
            "implementation_plan": self._build_action_plan(evidence, evidence.get("direction", "improve")),
            "follow_up": sustainment_plan,
        }

    def _pdsa_template(self, request: ImprovementRequest, evidence: Dict[str, Any], smart_aim: Dict[str, Any], experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "instructions": "Run one change at a time when possible. Predict the result before the test and capture what you learned even if the result is negative.",
            "plan": {
                "aim": smart_aim["statement"],
                "change_to_test": experiments[0]["title"] if experiments else "Define the first testable change",
                "prediction_prompt": "What do you predict will happen to the primary metric and why?",
                "data_to_collect": [
                    evidence.get("focus_metric"),
                    "time / subgroup stamp",
                    "context notes on setup, staffing, material, and abnormal events",
                ],
            },
            "do": {
                "execution_prompt": "Who ran the test, when, on what scope, and what deviations occurred?",
            },
            "study": {
                "compare_prompt": "Compare predicted vs actual change. Did the signal move in the desired direction and is it likely to be real?",
            },
            "act": {
                "decision_prompt": "Adopt, adapt, or abandon? What is the next test or standardization step?",
            },
        }

    def _build_summary(self, problem_definition: Dict[str, Any], smart_aim: Dict[str, Any], insights: List[Dict[str, Any]]) -> str:
        top = insights[0]["title"] if insights else "Baseline established"
        return f"{problem_definition['statement']} Top signal: {top}. Recommended aim: {smart_aim['statement']}"
