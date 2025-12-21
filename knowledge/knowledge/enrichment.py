"""
knowledge/enrichment.py
Attach KB concepts + interpretations to raw stats results.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .queries import get_concepts_by_output_keys
from .interpreter import interpret_p_value, interpret_r_squared, interpret_correlation, next_steps_from_outputs


DEFAULT_KEY_WHITELIST = {
    "p_value", "t_stat", "z_stat", "f_stat", "chi2", "df",
    "r_squared", "adjusted_r_squared",
    "slope", "intercept", "coefficients",
    "pearson_r", "spearman_r", "kendall_tau",
    "mae", "mse", "rmse", "accuracy", "precision", "recall", "f1", "auc", "roc_auc",
    "mape",
}


def extract_output_keys(result: Dict[str, Any]) -> List[str]:
    keys: set[str] = set()
    for k, v in (result or {}).items():
        if k in DEFAULT_KEY_WHITELIST:
            keys.add(k)
        if isinstance(v, dict):
            for kk in v.keys():
                if kk in DEFAULT_KEY_WHITELIST:
                    keys.add(kk)
    return sorted(keys)


def enrich_stats_result(result: Dict[str, Any], max_concepts: int = 10) -> Dict[str, Any]:
    result = dict(result or {})
    keys = extract_output_keys(result)

    kb = get_concepts_by_output_keys(keys, limit=max_concepts) if keys else []
    kb_matches = []
    for c in kb:
        kb_matches.append({
            "slug": c.get("slug"),
            "title": c.get("title"),
            "plain_english": c.get("plain_english"),
            "when_to_use": c.get("when_to_use"),
            "concept_type": c.get("concept_type"),
            "output_keys": c.get("output_keys"),
            "quality_score": c.get("quality_score"),
        })

    interpretations: Dict[str, Any] = {}

    p = result.get("p_value")
    if isinstance(p, (int, float)):
        interpretations["p_value"] = interpret_p_value(float(p))

    r2 = result.get("r_squared") or result.get("r2")
    if isinstance(r2, (int, float)):
        interpretations["r_squared"] = interpret_r_squared(float(r2))

    r = result.get("pearson_r") or result.get("spearman_r") or result.get("kendall_tau")
    if isinstance(r, (int, float)):
        interpretations["correlation"] = interpret_correlation(float(r))

    result["kb_matches"] = kb_matches
    if interpretations:
        result["interpretations"] = interpretations

    result["suggested_next_steps"] = next_steps_from_outputs(result)
    return result
