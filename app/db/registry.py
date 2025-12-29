"""
app/db/registry.py

Central DB registry for asyncpg pool + helper methods.

Railway + Supabase fixes included:
- Enforce SSL (Supabase commonly requires sslmode=require from external hosts)
- Clear error messages when DATABASE_URL is missing or credentials are wrong
"""

from __future__ import annotations

import asyncpg
from typing import Any, Optional, Sequence

from app.config import settings

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """
    Create (once) and return the asyncpg pool.

    Fixes repeated Railway 500s:
    - Fails fast if DATABASE_URL is missing
    - Enforces SSL for Supabase
    - Raises clear RuntimeErrors instead of vague stack traces
    """
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
            # ✅ Critical for many Supabase external connections
            ssl="require",
        )
        return _pool

    except asyncpg.InvalidPasswordError as e:
        raise RuntimeError(
            "DB authentication failed (InvalidPasswordError). "
            "Check DATABASE_URL user/password from Supabase."
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"Failed to create database pool: {type(e).__name__}: {e}. "
            "Check DATABASE_URL host/port and ensure it includes '?sslmode=require' (recommended)."
        ) from e


async def close_pool() -> None:
    """Close the pool gracefully (optional)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


class DBRegistry:
    """
    Lightweight wrapper providing common DB methods.
    """

    async def execute(self, query: str, *args: Any) -> str:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> Sequence[asyncpg.Record]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)


# Singleton used across the app
registry = DBRegistry()

