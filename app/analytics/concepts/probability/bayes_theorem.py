from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='bayes-theorem-func',
    topic_id='topic-id',
    topic_slug='probability',
    slug='bayes-theorem',
    title='Bayes' Theorem',
    concept_type='metric',
    level='intermediate',
    status='published',
    output_keys=['bayes_theorem'],
    tags=['probability'],
    quality_score=80,
)

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Bayes' Theorem - fully functional implementation."""
    p_a = params.get('p_a')
    p_b_given_a = params.get('p_b_given_a')
    p_b = params.get('p_b')
    
    if None in [p_a, p_b_given_a, p_b]:
        raise ValueError('p_a, p_b_given_a, and p_b required')
    
    # P(A|B) = P(B|A) * P(A) / P(B)
    p_a_given_b = (p_b_given_a * p_a) / p_b
    
    return {
        'posterior': float(p_a_given_b),
        'p_a_given_b': float(p_a_given_b),
        'prior': float(p_a),
        'likelihood': float(p_b_given_a),
        'marginal': float(p_b),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
