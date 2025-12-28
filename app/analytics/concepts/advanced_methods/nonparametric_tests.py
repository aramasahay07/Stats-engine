from __future__ import annotations

from typing import Any, Dict

from .._base import ConceptMeta

META = ConceptMeta(
    id='8048785a-ebd6-45b1-bae1-8bbd8f9a5941',
    topic_id='0e4fdff5-b126-4544-b5dc-e038ff36791f',
    topic_slug='advanced-methods',
    slug='nonparametric-tests',
    title='Nonparametric Tests',
    concept_type='test',
    level='intermediate',
    status='published',
    output_keys=['mann_whitney', 'kruskal_wallis'],
    tags=['testing', 'robust'],
    quality_score=80,
)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute concept: Nonparametric Tests (nonparametric-tests).

    DuckDB is the primary analytics engine.
    - ctx.con: duckdb connection
    - dataset is mounted as view/table named `dataset`

    Return a JSON-serializable dict. Prefer keys in META.output_keys.

    This module is auto-generated scaffold; implement as needed.
    """
    raise NotImplementedError('Concept implementation not yet added for slug: nonparametric-tests')

