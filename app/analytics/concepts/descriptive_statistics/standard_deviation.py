from __future__ import annotations

from typing import Any, Dict

from .._base import ConceptMeta

META = ConceptMeta(
    id='6dbde885-e2e2-4ef9-a45f-99544058bcb7',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='standard-deviation',
    title='Standard Deviation',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['std_dev', 'standard_deviation', 'sd'],
    tags=['spread', 'variation'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Standard Deviation (standard-deviation).

    DuckDB is the primary analytics engine.
    - ctx.con: duckdb connection
    - dataset is mounted as view/table named `dataset`

    Return a JSON-serializable dict. Prefer keys in META.output_keys.

    This module is auto-generated scaffold; implement as needed.
    """
    raise NotImplementedError('Concept implementation not yet added for slug: standard-deviation')

