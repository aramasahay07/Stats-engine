"""Startup and health-check utilities for required external services."""

import os
from typing import Any, Dict, List

from knowledge.client import rest_get

try:  # Optional; psycopg may not be installed in minimal environments
    import psycopg
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore

from app.services.storage_client import SupabaseStorageClient

REQUIRED_ENV_VARS = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY"]
OPTIONAL_ENV_VARS = ["SUPABASE_ANON_KEY", "SUPABASE_STORAGE_BUCKET", "DATABASE_URL"]


def required_env_status() -> List[Dict[str, Any]]:
    """Report presence of each required variable without exposing values."""

    entries = [
        {"name": "SUPABASE_URL", "present": bool(os.getenv("SUPABASE_URL"))},
        {
            "name": "SUPABASE_SERVICE_ROLE_KEY",
            "present": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
        },
        {"name": "SUPABASE_SERVICE_KEY", "present": bool(os.getenv("SUPABASE_SERVICE_KEY"))},
    ]
    return entries


def missing_env_vars() -> List[str]:
    """Return the list of required env vars that are absent."""

    missing: List[str] = []
    required = required_env_status()

    url_present = next((r for r in required if r["name"] == "SUPABASE_URL"), None)
    key_present = any(
        r["name"] in {"SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_KEY"} and r["present"]
        for r in required
    )

    if url_present and not url_present["present"]:
        missing.append("SUPABASE_URL")

    if not key_present:
        missing.append("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY")

    return missing


def require_env_vars() -> List[str]:
    """Collect missing env vars without raising during import/startup."""

    missing = missing_env_vars()
    if missing:
        print(f"⚠️  Missing required environment variables: {', '.join(missing)}")
    if not os.getenv("SUPABASE_ANON_KEY"):
        # Non-fatal: KB reads will fall back to service role.
        print(
            "⚠️  SUPABASE_ANON_KEY not set; knowledge base endpoints will use the service role key."
        )
    return missing


def check_database_connectivity() -> Dict[str, Any]:
    """Probe Supabase Postgres via DATABASE_URL if available."""

    database_url = os.getenv("DATABASE_URL") or ""
    if not database_url:
        return {"ok": False, "error": "DATABASE_URL not configured"}

    if psycopg is None:  # pragma: no cover - optional dependency
        return {"ok": False, "error": "psycopg not installed; cannot run SELECT 1"}

    try:
        with psycopg.connect(database_url, connect_timeout=5) as conn:  # type: ignore[arg-type]
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return {"ok": True, "database_url": "configured"}
    except Exception as exc:  # pragma: no cover - defensive health reporting
        return {"ok": False, "error": str(exc)}


def check_storage_connectivity() -> Dict[str, Any]:
    """Probe Supabase storage bucket reachability."""
    try:
        storage = SupabaseStorageClient()
        storage.client.storage.from_(storage.bucket).list(path="")
        return {"ok": True, "bucket": storage.bucket}
    except Exception as exc:  # pragma: no cover - defensive health reporting
        return {"ok": False, "error": str(exc)}


def check_knowledge_base() -> Dict[str, Any]:
    """Lightweight KB reachability check using the stat_topics table."""
    try:
        resp = rest_get("stat_topics", params={"select": "id", "limit": "1"})
        if resp.status_code < 300:
            return {"ok": True}
        return {"ok": False, "error": f"HTTP {resp.status_code}"}
    except Exception as exc:  # pragma: no cover - defensive health reporting
        return {"ok": False, "error": str(exc)}


def build_health_status(app_version: str | None = None) -> Dict[str, Any]:
    """Aggregate application health details without raising."""

    env_missing = missing_env_vars()
    env_required = required_env_status()
    db = check_database_connectivity()
    storage = check_storage_connectivity()
    kb = check_knowledge_base()

    checks = {
        "app": {"ok": True},
        "env": {"ok": not env_missing, "missing": env_missing, "required": env_required},
        "database": db,
        "storage": storage,
        "knowledge_base": kb,
    }

    errors = []
    for name, result in checks.items():
        if not result.get("ok"):
            msg = result.get("error") or "missing configuration"
            errors.append({"check": name, "error": msg})

    overall_ok = all(check.get("ok") for check in checks.values())

    return {
        "status": "ok" if overall_ok else "degraded",
        "version": app_version,
        "checks": checks,
        "errors": errors,
    }
