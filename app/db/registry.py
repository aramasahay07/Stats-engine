"""
Database connection pool manager.

Manages asyncpg connection pool for efficient database access.
Connections are reused instead of creating new ones for each query.
"""

import asyncpg
from typing import List, Optional, Any
from app.config import settings

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """
    Get or create the database connection pool.
    
    Pool is created on first call and reused for subsequent calls.
    This is thread-safe due to Python's GIL.
    
    Returns:
        asyncpg.Pool: Database connection pool
        
    Raises:
        Exception: If pool creation fails
    """
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=1,
                max_size=10,
                # Optional: Add connection timeout
                timeout=30,
                # Optional: Command timeout (prevent hanging queries)
                command_timeout=60,
            )
        except Exception as e:
            # Log the error (you might want to use proper logging)
            print(f"Failed to create database pool: {e}")
            raise
    return _pool


async def close_pool() -> None:
    """
    Close the database connection pool.
    
    Should be called on application shutdown to gracefully close connections.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def fetchrow(query: str, *args) -> Optional[asyncpg.Record]:
    """
    Execute a query and return a single row.
    
    Args:
        query: SQL query string (use $1, $2, etc. for parameters)
        *args: Query parameters
        
    Returns:
        asyncpg.Record or None: Single database row or None if no results
        
    Example:
        row = await fetchrow("SELECT * FROM users WHERE id=$1", user_id)
        if row:
            print(row['name'])
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch(query: str, *args) -> List[asyncpg.Record]:
    """
    Execute a query and return all rows.
    
    Args:
        query: SQL query string (use $1, $2, etc. for parameters)
        *args: Query parameters
        
    Returns:
        List[asyncpg.Record]: List of database rows (empty list if no results)
        
    Example:
        rows = await fetch("SELECT * FROM users WHERE active=$1", True)
        for row in rows:
            print(row['name'])
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def execute(query: str, *args) -> str:
    """
    Execute a query without returning data (INSERT, UPDATE, DELETE, etc.).
    
    Args:
        query: SQL query string (use $1, $2, etc. for parameters)
        *args: Query parameters
        
    Returns:
        str: Status string (e.g., "INSERT 0 1", "UPDATE 1", "DELETE 1")
        
    Example:
        result = await execute(
            "UPDATE users SET active=$1 WHERE id=$2",
            False, user_id
        )
        # result = "UPDATE 1"
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetchval(query: str, *args, column: int = 0) -> Any:
    """
    Execute a query and return a single value.
    
    Useful for COUNT, SUM, MAX, etc. queries that return a single value.
    
    Args:
        query: SQL query string (use $1, $2, etc. for parameters)
        *args: Query parameters
        column: Column index to return (default: 0 = first column)
        
    Returns:
        Any: Single value from the query result
        
    Example:
        count = await fetchval("SELECT COUNT(*) FROM users")
        print(f"Total users: {count}")
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args, column=column)
