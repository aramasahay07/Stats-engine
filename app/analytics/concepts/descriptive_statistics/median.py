from __future__ import annotations

from typing import Any, Dict

from .._base import ConceptMeta

META = ConceptMeta(
    id='efd3f87f-357b-483b-9999-e1165429ff7a',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='median',
    title='Median',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['median'],
    tags=['center', 'robust'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Median (median).

    DuckDB is the primary analytics engine.
    - ctx.con: duckdb connection
    - dataset is mounted as view/table named `dataset`

    Return a JSON-serializable dict. Prefer keys in META.output_keys.

    This module is auto-generated scaffold; implement as needed.
    """
    raise NotImplementedError('Concept implementation not yet added for slug: median')

