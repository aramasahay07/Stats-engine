"""
knowledge/client.py
Centralized Supabase (PostgREST) client helpers.

Uses SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY (server-side) or SUPABASE_ANON_KEY (read-only).
Prefer SERVICE ROLE for importer/admin tasks; for public KB reads, ANON is OK if RLS allows SELECT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import requests


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str
    timeout: int = 30


def get_supabase_config(read_only: bool = True) -> SupabaseConfig:
    """
    read_only=True  -> uses SUPABASE_ANON_KEY (recommended for public KB endpoints)
    read_only=False -> uses SUPABASE_SERVICE_ROLE_KEY (admin/import scripts ONLY)
    """
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    if not url:
        raise RuntimeError("Missing SUPABASE_URL")

    if read_only:
        key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
        if not key:
            raise RuntimeError("Missing SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY as fallback)")
    else:
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
        if not key:
            raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY")

    timeout = int(os.getenv("KB_HTTP_TIMEOUT", "30"))
    return SupabaseConfig(url=url, key=key, timeout=timeout)


def rest_headers(cfg: SupabaseConfig, prefer: str | None = None) -> Dict[str, str]:
    h = {
        "apikey": cfg.key,
        "Authorization": f"Bearer {cfg.key}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def rest_get(path: str, params: Dict[str, str] | None = None, read_only: bool = True) -> requests.Response:
    cfg = get_supabase_config(read_only=read_only)
    url = f"{cfg.url}/rest/v1/{path.lstrip('/')}"
    return requests.get(url, headers=rest_headers(cfg), params=params, timeout=cfg.timeout)


def rest_post(path: str, payload, params: Dict[str, str] | None = None, read_only: bool = True, prefer: str | None = None) -> requests.Response:
    cfg = get_supabase_config(read_only=read_only)
    url = f"{cfg.url}/rest/v1/{path.lstrip('/')}"
    return requests.post(url, headers=rest_headers(cfg, prefer=prefer), params=params, json=payload, timeout=cfg.timeout)
