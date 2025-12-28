from __future__ import annotations

from typing import Any, Dict

from .._base import ConceptMeta

META = ConceptMeta(
    id='5f0260dd-52e4-4ddc-8ab0-51870579fc68',
    topic_id='8b2247d1-7415-41e7-b0c3-d5a81878ba3f',
    topic_slug='predictive-ml',
    slug='recall',
    title='Recall (Sensitivity)',
    concept_type='metric',
    level='intro',
    status='published',
    output_keys=['recall', 'sensitivity'],
    tags=['metrics'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Recall (Sensitivity) (recall).

    DuckDB is the primary analytics engine.
    - ctx.con: duckdb connection
    - dataset is mounted as view/table named `dataset`

    Return a JSON-serializable dict. Prefer keys in META.output_keys.

    This module is auto-generated scaffold; implement as needed.
    """
    raise NotImplementedError('Concept implementation not yet added for slug: recall')

