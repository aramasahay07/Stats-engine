from __future__ import annotations

from typing import Any, Dict

from .._base import ConceptMeta

META = ConceptMeta(
    id='cfbc5ff5-a1ce-4ed4-a966-47c15bcbd3fc',
    topic_id='e5b8a289-d663-4317-a4cf-1e90ca3f6e64',
    topic_slug='descriptive-statistics',
    slug='outliers',
    title='Outliers',
    concept_type='procedure',
    level='intro',
    status='published',
    output_keys=['outliers', 'outlier'],
    tags=['quality', 'variation'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Outliers (outliers).

    DuckDB is the primary analytics engine.
    - ctx.con: duckdb connection
    - dataset is mounted as view/table named `dataset`

    Return a JSON-serializable dict. Prefer keys in META.output_keys.

    This module is auto-generated scaffold; implement as needed.
    """
    raise NotImplementedError('Concept implementation not yet added for slug: outliers')

