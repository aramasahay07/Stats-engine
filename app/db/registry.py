from __future__ import annotations

import json
from typing import Optional

import asyncpg
from app.config import settings

# ------------------------------------------------------------
# Database access wrapper with asyncpg JSON / JSONB support
# ------------------------------------------------------------

_pool: Optional[asyncpg.Pool] = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    """
    Ensure asyncpg can send/receive Python dict/list for json/jsonb columns.

    Without this, asyncpg may treat json/jsonb parameters as text and throw:
      TypeError: expected str, got list
    """
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


async def get_pool() -> asyncpg.Pool:
    global _pool

    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=5,
        ssl="require",   # Supabase / Railway typically require SSL
        init=_init_conn, # JSON / JSONB auto encode/decode
    )

    return _pool


async def close_pool() -> None:
    """Gracefully close the global asyncpg pool (used on FastAPI shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ------------------------------------------------------------
# Convenience helpers (optional but common in your codebase)
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
