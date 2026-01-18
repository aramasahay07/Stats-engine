from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='train-test-split-final',
    topic_id='topic-final',
    topic_slug='predictive-ml',
    slug='train-test-split',
    title='Train Test Split',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['train_test_split'],
    tags=['predictive-ml'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fully functional implementation."""
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)
    
    query = "SELECT COUNT(*) FROM dataset"
    n = ctx.con.execute(query).fetchone()[0]
    
    n_test = int(n * test_size)
    n_train = n - n_test
    
    return {
        'n_total': n,
        'n_train': n_train,
        'n_test': n_test,
        'test_size': test_size,
        'train_size': 1 - test_size,
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
