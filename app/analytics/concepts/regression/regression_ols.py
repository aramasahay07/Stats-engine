from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict, List, Union

META = ConceptMeta(
    id='151cc576-a07a-4bac-8e96-70d82cd6d27a',
    topic_id='c35a2689-8dc3-4d54-9b26-470e8057bee2',
    topic_slug='regression',
    slug='regression-ols',
    title='OLS Regression (Auto: Simple or Multiple)',
    concept_type='analysis',
    level='intro',
    status='published',
    output_keys=['regression'],
    tags=['regression', 'ols'],
    quality_score=80,
)


def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i]
    return [str(x)]

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to simple-linear-regression or multiple-linear-regression depending on predictors.

    Params:
      - y: target column (or target, dependent_column)
      - x: predictor column(s) (string or list) (or feature, features)
    """
    # NOTE: registry provides concept lookup by slug
    from ..registry import get_concept_module  # type: ignore

    y = params.get("y") or params.get("target") or params.get("dependent_column")
    x = params.get("x") or params.get("feature") or params.get("features")

    if not y:
        raise ValueError("Provide y (or target/dependent_column)")
    xs = _as_list(x)
    if not xs:
        raise ValueError("Provide x (or feature/features)")

    pass_params = dict(params)
    pass_params["y"] = y
    pass_params["x"] = xs if len(xs) > 1 else xs[0]

    slug = "multiple-linear-regression" if len(xs) > 1 else "simple-linear-regression"
    mod = get_concept_module(slug)
    if mod is None:
        raise ValueError(f"Required concept missing: chi-square-test")

    return await mod.run(ctx, pass_params)

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
