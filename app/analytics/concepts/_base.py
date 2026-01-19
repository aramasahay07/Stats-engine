from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import time


@dataclass(frozen=True)
class ConceptMeta:
    id: str
    topic_id: str
    topic_slug: str
    slug: str
    title: str
    concept_type: str
    level: str
    status: str
    output_keys: List[str]
    tags: List[str]
    quality_score: int


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the data."""
    severity: str  # 'error', 'warning', 'info'
    message: str
    field: Optional[str] = None


def _json_safe(value: Any) -> Any:
    """
    Recursively convert common non-JSON-serializable scientific Python types
    (NumPy/Pandas scalars, arrays, etc.) into native Python types.
    """
    # Fast path for already-JSON-safe primitives
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Dict-like
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # List / tuple-like
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    # Try NumPy/Pandas scalar -> Python scalar
    # (np.bool_, np.float64, pd.Timestamp scalars, etc.)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    # Try arrays/Series -> Python lists
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    # Fallback: leave as-is (may still fail if truly non-serializable,
    # but this covers the common NumPy/Pandas cases safely).
    return value


async def run_concept(
    meta: ConceptMeta,
    ctx: Any,
    params: Dict[str, Any],
    execute_analysis: Callable,
) -> Dict[str, Any]:
    """
    Universal wrapper for all concept execution.

    Provides:
    - Validation phase
    - Assumptions checking
    - Timing metrics
    - Error handling
    - Standardized output format

    Args:
        meta: Concept metadata
        ctx: Analysis context (contains DuckDB connection, etc.)
        params: User-provided parameters
        execute_analysis: The actual analysis function

    Returns:
        Standardized result dictionary
    """
    start_time = time.time()

    result: Dict[str, Any] = {
        "meta": {
            "slug": meta.slug,
            "title": meta.title,
            "concept_type": meta.concept_type,
        },
        "status": "ok",
        "validation_issues": [],
        "assumptions": {},
        "results": {},
        "timing_ms": 0,
    }

    try:
        # Phase 1: Basic validation
        validation_issues = await validate_inputs(ctx, params)
        result["validation_issues"] = [
            {"severity": v.severity, "message": v.message, "field": v.field}
            for v in validation_issues
        ]

        # Stop if there are validation errors
        if any(v.severity == "error" for v in validation_issues):
            result["status"] = "validation_failed"
            return _json_safe(result)

        # Phase 2: Check statistical assumptions (if applicable)
        assumptions = await check_assumptions(ctx, params, meta)
        result["assumptions"] = assumptions

        # Phase 3: Execute the actual analysis
        analysis_results = await execute_analysis(ctx, params)
        result["results"] = analysis_results

        # Phase 4: Generate visualizations (optional)
        # visuals = await generate_visuals(ctx, params, analysis_results, meta)
        # result['visuals'] = visuals

    except ValueError as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["error_type"] = "ValueError"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["error_type"] = type(e).__name__

    finally:
        # Always record timing
        elapsed_ms = (time.time() - start_time) * 1000
        result["timing_ms"] = round(elapsed_ms, 2)

    # Ensure the full payload is JSON-serializable (NumPy/Pandas safe)
    return _json_safe(result)


async def validate_inputs(ctx: Any, params: Dict[str, Any]) -> List[ValidationIssue]:
    """Validate input parameters."""
    issues: List[ValidationIssue] = []

    # Check for required columns exist in dataset
    for key, value in params.items():
        if key.endswith("_column") or key == "column":
            if value:
                # Check if column exists (simplified - adapt to your needs)
                try:
                    ctx.con.execute(f"SELECT {value} FROM dataset LIMIT 1")
                except Exception:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=f'Column "{value}" not found in dataset',
                            field=key,
                        )
                    )

    return issues


async def check_assumptions(ctx: Any, params: Dict[str, Any], meta: ConceptMeta) -> Dict[str, Any]:
    """Check statistical assumptions (normality, equal variance, etc.)."""
    assumptions: Dict[str, Any] = {}

    # Only check assumptions for tests that require them
    if "t_test" in meta.tags or "anova" in meta.tags:
        # Check normality
        column = params.get("measure_column") or params.get("column")
        if column:
            try:
                from scipy import stats  # noqa: F401
                import numpy as np  # noqa: F401

                query = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL LIMIT 5000"
                data = [r[0] for r in ctx.con.execute(query).fetchall()]

                if len(data) >= 3:
                    # Shapiro-Wilk test for normality
                    if len(data) <= 5000:
                        stat, p_value = stats.shapiro(data)
                    else:
                        stat, p_value = stats.normaltest(data)

                    assumptions["normality"] = {
                        "test": "shapiro_wilk" if len(data) <= 5000 else "dagostino_pearson",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "normal": p_value > 0.05,
                        "interpretation": (
                            "Data appears normally distributed"
                            if p_value > 0.05
                            else "Data may not be normally distributed"
                        ),
                    }
            except Exception:
                # Assumptions are best-effort; don't fail the concept because of them.
                pass

    return assumptions


async def generate_visuals(
    ctx: Any,
    params: Dict[str, Any],
    results: Dict[str, Any],
    meta: ConceptMeta,
) -> Dict[str, Any]:
    """Generate visualization specifications (Vega-Lite format)."""
    visuals: Dict[str, Any] = {}

    # This is a placeholder - you can implement actual chart generation
    # based on the concept type and results

    return visuals

