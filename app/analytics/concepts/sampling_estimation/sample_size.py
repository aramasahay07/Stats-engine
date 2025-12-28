from __future__ import annotations

from typing import Any, Dict

from .._base import ConceptMeta

META = ConceptMeta(
    id='6eb192d7-c063-4472-86bd-157b3b989c1b',
    topic_id='db0cd6cf-0baf-4ef9-819f-295b6668c581',
    topic_slug='sampling-estimation',
    slug='sample-size',
    title='Sample Size Planning',
    concept_type='procedure',
    level='intermediate',
    status='published',
    output_keys=['sample_size', 'n_required'],
    tags=['planning', 'power'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Sample Size Planning (sample-size).

    DuckDB is the primary analytics engine.
    - ctx.con: duckdb connection
    - dataset is mounted as view/table named `dataset`

    Return a JSON-serializable dict. Prefer keys in META.output_keys.

    This module is auto-generated scaffold; implement as needed.
    """
    raise NotImplementedError('Concept implementation not yet added for slug: sample-size')

