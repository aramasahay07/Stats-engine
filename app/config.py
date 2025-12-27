from __future__ import annotations

"""Lightweight configuration helpers for startup logging and health checks."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    version: str
    data_dir: Path
    bucket: str
    auth_disabled: bool
    env_mode: str


def load_settings(app_version: str | None = None) -> Settings:
    return Settings(
        version=app_version or os.getenv("APP_VERSION") or "unknown",
        data_dir=Path(os.getenv("DATA_DIR", "./cache")).resolve(),
        bucket=os.getenv("SUPABASE_STORAGE_BUCKET", "datasets"),
        auth_disabled=(os.getenv("AUTH_DISABLED", "false").lower() == "true"),
        env_mode=os.getenv("ENV") or os.getenv("ENVIRONMENT") or "development",
    )


def print_startup_banner(settings: Settings, missing_env: list[str] | None = None) -> None:
    missing_env = missing_env or []

    port = os.getenv("PORT", "8000")

    print("=" * 70)
    print(f"🚀 AI Data Lab v{settings.version} - Complete Analytics Platform")
    print("=" * 70)
    print("Environment:")
    print(f"  Mode: {settings.env_mode}")
    print(f"  Auth disabled: {settings.auth_disabled}")
    print(f"  Data directory: {settings.data_dir}")
    print(f"  Storage bucket: {settings.bucket}")
    if missing_env:
        print(f"  ⚠️  Missing env vars: {', '.join(missing_env)}")
    print(f"📚 API Docs: http://0.0.0.0:{port}/docs")
    print("=" * 70)
