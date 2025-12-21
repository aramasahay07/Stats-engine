"""
knowledge/interpreter.py
Lightweight, rule-based interpretation helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def interpret_p_value(p: Optional[float], alpha: float = 0.05) -> str:
    if p is None:
        return "No p-value provided."
    if p < alpha:
        return f"Statistically significant at α={alpha:g} (p < {alpha:g})."
    return f"Not statistically significant at α={alpha:g} (p ≥ {alpha:g})."


def interpret_r_squared(r2: Optional[float]) -> str:
    if r2 is None:
        return "No R² provided."
    if r2 >= 0.9:
        return "Very strong fit (explains most variation)."
    if r2 >= 0.7:
        return "Strong fit."
    if r2 >= 0.5:
        return "Moderate fit."
    return "Weak fit (explains limited variation)."


def interpret_correlation(r: Optional[float]) -> str:
    if r is None:
        return "No correlation provided."
    ar = abs(r)
    if ar >= 0.8:
        strength = "very strong"
    elif ar >= 0.6:
        strength = "strong"
    elif ar >= 0.4:
        strength = "moderate"
    elif ar >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    direction = "positive" if r >= 0 else "negative"
    return f"{strength.title()} {direction} association."


def next_steps_from_outputs(outputs: Dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if "p_value" in outputs:
        steps.append("Check effect size and confidence interval to assess practical impact.")
    if "r_squared" in outputs or "r2" in outputs:
        steps.append("Review residual plots to confirm assumptions and check for nonlinearity.")
    if "roc_auc" in outputs or "auc" in outputs:
        steps.append("Pick an operating threshold based on false-positive vs false-negative costs.")
    return steps
