from __future__ import annotations

from fastapi.testclient import TestClient

from app.auth.supabase_jwt import get_current_user
from app.main import create_app


def test_process_mining_route_uses_authenticated_user(monkeypatch):
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "user-123"}

    called = {}

    async def fake_analyze_process_mining(user_id: str, dataset_id: str, request):
        called["user_id"] = user_id
        called["dataset_id"] = dataset_id
        return {
            "summary": {
                "total_cases": 1,
                "total_events": 2,
                "unique_activities": 2,
                "avg_cycle_time_hours": 1.0,
                "median_cycle_time_hours": 1.0,
                "rework_rate": 0.0,
                "variant_count": 1,
                "sla_breach_rate": None,
            },
            "process_map": {"nodes": [], "edges": []},
            "variants": [],
            "bottlenecks": [],
            "rework_loops": [],
            "ai_insights": {
                "executive_summary": "",
                "key_findings": [],
                "recommended_actions": [],
            },
            "target_state": {},
            "toc_analysis": {},
            "conformance": {},
            "root_causes": [],
            "initiatives": [],
            "edge_durations": {},
            "expected_path": [],
            "goals": None,
            "cost_inputs": None,
        }

    monkeypatch.setattr(
        "app.routers.process_mining.analyze_process_mining",
        fake_analyze_process_mining,
    )

    client = TestClient(app)
    response = client.post(
        "/datasets/dataset-1/process-mining/analyze",
        json={
            "mapping": {
                "case_id_column": "case_id",
                "activity_column": "activity",
                "timestamp_column": "event_ts",
            }
        },
    )

    assert response.status_code == 200
    assert called == {"user_id": "user-123", "dataset_id": "dataset-1"}
