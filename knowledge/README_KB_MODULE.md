# KB Module (All 6 files)

## What you get
- knowledge/client.py
- knowledge/queries.py
- knowledge/enrichment.py
- knowledge/interpreter.py
- knowledge/models.py
- knowledge/__init__.py
- routers/kb.py (FastAPI router)

## Env vars
- SUPABASE_URL
- SUPABASE_ANON_KEY (recommended for KB read endpoints)
- (optional) SUPABASE_SERVICE_ROLE_KEY (admin/import scripts only)

## Mount router
In `main.py`:
```python
from routers.kb import router as kb_router
app.include_router(kb_router, prefix="/kb", tags=["Knowledge Base"])
```

## Test
- GET /kb/topics
- GET /kb/search?q=regression
- GET /kb/concepts/p-value
- POST /kb/enrich with JSON body: {"p_value": 0.03, "t_stat": 2.45}
