from __future__ import annotations

from app.models.process_mining import AnalyzeProcessRequest, ProcessMiningResult
from app.services.process_mining.conformance import analyze_conformance
from app.services.process_mining.initiatives import build_initiatives
from app.services.process_mining.loader import load_process_mining_context
from app.services.process_mining.metrics import (
    build_edge_duration_map,
    compute_bottlenecks,
    compute_process_map_edges,
    compute_process_map_nodes,
    compute_rework_loops,
    compute_summary,
    compute_variants,
    create_direct_follows_view,
)
from app.services.process_mining.narrator import build_ai_insights
from app.services.process_mining.root_causes import analyze_root_causes
from app.services.process_mining.shaper import build_canonical_event_log_view
from app.services.process_mining.target_state import build_target_state
from app.services.process_mining.toc import build_toc_analysis
from app.services.process_mining.validator import validate_canonical_event_log, validate_pre_shape


async def analyze_process_mining(user_id: str, dataset_id: str, request: AnalyzeProcessRequest) -> ProcessMiningResult:
    ctx = await load_process_mining_context(user_id, dataset_id)
    try:
        validate_pre_shape(ctx.con, ctx.base_view, request.mapping, request.shape)
        event_log_view = build_canonical_event_log_view(ctx.con, ctx.base_view, request.mapping, request.shape)
        validate_canonical_event_log(ctx.con, event_log_view)

        direct_follows_view = create_direct_follows_view(ctx.con, event_log_view)

        goals_dict = request.goals.model_dump() if request.goals is not None else {}
        summary = compute_summary(ctx.con, event_log_view, goals=goals_dict)
        nodes = compute_process_map_nodes(ctx.con, event_log_view, direct_follows_view)
        edges = compute_process_map_edges(ctx.con, direct_follows_view)
        variants = compute_variants(ctx.con, event_log_view)
        bottlenecks = compute_bottlenecks(ctx.con, direct_follows_view)
        rework_loops = compute_rework_loops(ctx.con, event_log_view)

        expected_path = list(request.expected_path or [])
        if not expected_path and variants:
            expected_path = variants[0].activities

        ai_insights = build_ai_insights(summary, variants, bottlenecks, rework_loops)
        target_state = build_target_state()
        toc_analysis = build_toc_analysis()
        conformance = analyze_conformance(expected_path)
        root_causes = analyze_root_causes()
        initiatives = build_initiatives()

        return ProcessMiningResult(
            summary=summary,
            process_map={"nodes": nodes, "edges": edges},
            variants=variants,
            bottlenecks=bottlenecks,
            rework_loops=rework_loops,
            ai_insights=ai_insights,
            target_state=target_state,
            toc_analysis=toc_analysis,
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
