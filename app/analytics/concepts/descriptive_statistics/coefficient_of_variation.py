from __future__ import annotations

from typing import Any, Dict

from .._base import ConceptMeta

META = ConceptMeta(
    id='ef0a896d-3a00-4b3e-b300-03b70b8abe62',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='coefficient-of-variation',
    title='Coefficient of Variation (CV)',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['cv', 'coef_var'],
    tags=['spread'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Coefficient of Variation (CV) (coefficient-of-variation).

    DuckDB is the primary analytics engine.
    - ctx.con: duckdb connection
    - dataset is mounted as view/table named `dataset`

    Return a JSON-serializable dict. Prefer keys in META.output_keys.

    This module is auto-generated scaffold; implement as needed.
    """
    raise NotImplementedError('Concept implementation not yet added for slug: coefficient-of-variation')

