from __future__ import annotations

from app.models.process_mining import AnalyzeProcessRequest, ProcessMiningResult
from app.services.process_mining.conformance import analyze_conformance
from app.services.process_mining.initiatives import build_initiatives
from app.services.process_mining.loader import load_process_mining_context
from app.services.process_mining.metrics import (
    build_edge_duration_map,
    compute_case_records,
    compute_bottlenecks,
    compute_process_map_edges,
    compute_process_map_nodes,
    compute_rework_loops,
    compute_summary,
    compute_variants,
    create_case_summary_view,
    create_direct_follows_view,
)
from app.services.process_mining.narrator import build_ai_insights
from app.services.process_mining.root_causes import analyze_root_causes
from app.services.process_mining.shaper import build_canonical_event_log_view
from app.services.process_mining.target_state import build_target_state
from app.services.process_mining.toc import build_toc_analysis
from app.services.process_mining.validator import validate_canonical_event_log, validate_pre_shape


async def analyze_process_mining(user_id: str, request: AnalyzeProcessRequest) -> ProcessMiningResult:
    ctx = await load_process_mining_context(user_id, request.dataset_id)
    try:
        validate_pre_shape(ctx.con, ctx.base_view, request.mapping, request.shape)
        event_log_view = build_canonical_event_log_view(ctx.con, ctx.base_view, request.mapping, request.shape)
        validate_canonical_event_log(ctx.con, event_log_view)

        direct_follows_view = create_direct_follows_view(ctx.con, event_log_view)
        case_summary_view = create_case_summary_view(
            ctx.con,
            event_log_view,
            request.mapping.attribute_columns,
            request.goals.sla_hours if request.goals is not None else None,
        )

        summary = compute_summary(ctx.con, case_summary_view)
        if request.goals is None or request.goals.sla_hours is None:
            summary = summary.model_copy(update={"sla_breach_rate": None})

        nodes = compute_process_map_nodes(ctx.con, direct_follows_view)
        edges = compute_process_map_edges(ctx.con, direct_follows_view)
        variants = compute_variants(ctx.con, case_summary_view)
        cases = compute_case_records(ctx.con, case_summary_view, variants)
        bottlenecks = compute_bottlenecks(ctx.con, direct_follows_view)
        rework_loops = compute_rework_loops(ctx.con)

        expected_path = list(request.expected_path or [])
        if not expected_path and variants:
            expected_path = variants[0].path

        ai_insights = build_ai_insights(summary, variants, bottlenecks, rework_loops)
        target_state, edge_annotations, constraint_edge, constraint_rationale = build_target_state(
            summary=summary,
            nodes=nodes,
            edges=edges,
            rework_loops=rework_loops,
            protected_activities=request.goals.protected_activities if request.goals is not None else [],
            total_cases=summary.total_cases,
        )
        conformance = analyze_conformance(expected_path, cases)
        root_causes = analyze_root_causes(ctx.con, case_summary_view, request.mapping.attribute_columns)
        initiatives = build_initiatives(
            edge_annotations,
            request.cost_inputs.cost_per_hour if request.cost_inputs is not None else None,
        )
        toc_analysis = build_toc_analysis(
            constraint_edge,
            constraint_rationale,
            target_state.projected_impact.throughput_uplift_pct,
        )

        return ProcessMiningResult(
            summary=summary,
            process_map={"nodes": nodes, "edges": edges},
            variants=variants,
            bottlenecks=bottlenecks,
            rework_loops=rework_loops,
            ai_insights=ai_insights,
            target_state=target_state,
            toc_analysis=toc_analysis,
            cases=cases,
            conformance=conformance,
            root_causes=root_causes,
            initiatives=initiatives,
            edge_durations=build_edge_duration_map(edges),
            expected_path=expected_path,
            goals=request.goals,
            cost_inputs=request.cost_inputs,
        )
    finally:
        ctx.close()
