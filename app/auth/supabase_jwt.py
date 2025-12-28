import time
import jwt
import httpx

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings, AUTH_DISABLED

bearer = HTTPBearer(auto_error=False)

_JWKS_CACHE: dict = {"keys": None, "fetched_at": 0}
_JWKS_TTL_SECONDS = 60 * 60  # 1 hour


async def _get_jwks() -> dict:
    now = int(time.time())
    if _JWKS_CACHE["keys"] and (now - _JWKS_CACHE["fetched_at"] < _JWKS_TTL_SECONDS):
        return _JWKS_CACHE["keys"]

    url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Unable to fetch Supabase JWKS")
    jwks = resp.json()
    _JWKS_CACHE["keys"] = jwks
    _JWKS_CACHE["fetched_at"] = now
    return jwks


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = creds.credentials

    jwks = await _get_jwks()
    try:
        unverified_header = jwt.get_unverified_header(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid JWT header")

    kid = unverified_header.get("kid")
    key = None
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            key = k
            break
    if not key:
        raise HTTPException(status_code=401, detail="JWT key not found")

    try:
        payload = jwt.decode(
            token,
            jwt.PyJWK(key).key,
            algorithms=[unverified_header.get("alg", "RS256")],
            audience=settings.supabase_jwt_audience,
            options={"verify_exp": True},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="JWT expired")
    except Exception:
        raise HTTPException(status_code=401, detail="JWT verification failed")

    # Supabase: user id is in 'sub'
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="JWT missing user id")
    return {"user_id": user_id, "claims": payload}

