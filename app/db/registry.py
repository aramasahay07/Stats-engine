"""
app/db/registry.py

Asyncpg pool + DB helper functions.

Backwards compatible:
- Supports calling: `from app.db import registry` then `await registry.execute(...)`
- Also provides DBRegistry class + `registry_client` singleton if needed.

Supabase/Railway hardened:
- Enforces SSL
- Clear errors when DATABASE_URL missing/bad
"""

from __future__ import annotations

import asyncpg
from typing import Any, Optional, Sequence

from app.config import settings

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool

    if _pool is not None:
        return _pool

    dsn = (settings.database_url or "").strip()
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is empty or not set. "
            "In Railway Variables, set DATABASE_URL to your Supabase Postgres connection string."
        )

    try:
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=10,
            timeout=30,
            command_timeout=60,
            ssl="require",  # ✅ Supabase often requires SSL externally
        )
        return _pool

    except asyncpg.InvalidPasswordError as e:
        raise RuntimeError(
            "DB authentication failed (InvalidPasswordError). "
            "Check Supabase DATABASE_URL user/password."
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"Failed to create database pool: {type(e).__name__}: {e}. "
            "Check Supabase host/port and use '?sslmode=require' (recommended)."
        ) from e


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# -------------------------------------------------------------------
# Backwards-compatible module-level helpers
# These fix: AttributeError: module 'app.db.registry' has no attribute 'execute'
# -------------------------------------------------------------------

async def execute(query: str, *args: Any) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args: Any) -> Sequence[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args: Any) -> Any:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


# -------------------------------------------------------------------
# Optional OO wrapper (safe to keep for future refactors)
# -------------------------------------------------------------------

class DBRegistry:
    async def execute(self, query: str, *args: Any) -> str:
        return await execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> Sequence[asyncpg.Record]:
        return await fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        return await fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await fetchval(query, *args)


# If any code imports `registry_client`, it's available
registry_client = DBRegistry()
