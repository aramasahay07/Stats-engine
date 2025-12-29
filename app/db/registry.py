from __future__ import annotations

import json
from typing import Optional

import asyncpg
from app.config import settings

# ============================================================
# AsyncPG connection registry
# - Centralized pool management
# - Safe for Supabase / Railway / local Postgres
# - JSON / JSONB codec support
# ============================================================

_pool: Optional[asyncpg.Pool] = None


# ------------------------------------------------------------
# Connection initializer
# Ensures Python dict/list work with json & jsonb columns
# ------------------------------------------------------------
async def _init_conn(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "json",
        encoder=lambda v: json.dumps(v, default=str),
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )

    await conn.set_type_codec(
        "jsonb",
        encoder=lambda v: json.dumps(v, default=str),
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )


# ------------------------------------------------------------
# Pool access
# ------------------------------------------------------------
async def get_pool() -> asyncpg.Pool:
    """
    Lazily create and return the global asyncpg pool.
    """
    global _pool

    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=5,
        ssl="require",        # Required for Supabase / Railway
        init=_init_conn,      # JSON / JSONB handling
    )

    return _pool


# ------------------------------------------------------------
# Graceful shutdown hook (USED BY main.py)
# ------------------------------------------------------------
async def close_pool() -> None:
    """
    Close the asyncpg pool on application shutdown.
    Safe to call even if the pool was never created.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ------------------------------------------------------------
# Convenience query helpers (used across services)
# ------------------------------------------------------------
async def fetchval(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def fetchrow(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def execute(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)
