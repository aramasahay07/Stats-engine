"""
QAAgent - Quality Assurance Validation Agent.

Validates combined AI Analyst responses to ensure:
- JSON serializable outputs
- Narrative matches actual test and numbers
- Charts match test type
- No invented values
"""

import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from .models import AnalystResponse, ChartSpec
from .utils import json_safe


class ValidationError:
    """A validation error found by QAAgent."""

    def __init__(self, category: str, message: str, severity: str = "error"):
        self.category = category
        self.message = message
        self.severity = severity  # "error", "warning", "info"

    def to_dict(self) -> Dict[str, str]:
        return {
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
        }


class QAAgent:
    """
    Agent for validating AI Analyst responses.

    Ensures responses are consistent, accurate, and properly formatted.
    Blocks final output if critical validation errors are found.
    """

    # Expected chart types for each analysis category
    EXPECTED_CHARTS: Dict[str, Set[str]] = {
        "t-test": {"boxplot", "histogram", "violin", "strip"},
        "anova": {"boxplot", "histogram", "violin", "strip", "error_bar"},
        "chi-square": {"stacked_bar", "normalized_bar", "heatmap", "mosaic"},
        "correlation": {"scatter", "hexbin", "heatmap"},
        "regression": {"scatter", "regression", "residual", "histogram", "qq"},
        "time_series": {"line", "line_ma", "run_chart", "area"},
        "descriptive": {"histogram", "boxplot", "density"},
    }

    # Analysis slug to category mapping
    ANALYSIS_CATEGORIES: Dict[str, str] = {
        "two-sample-t-test": "t-test",
        "ttest_2samp": "t-test",
        "welch-t-test": "t-test",
        "paired-t-test": "t-test",
        "one-sample-t-test": "t-test",
        "anova-one-way": "anova",
        "anova_oneway": "anova",
        "kruskal-wallis": "anova",
        "chi-square-test": "chi-square",
        "chi_square": "chi-square",
        "fisher-exact-test": "chi-square",
        "pearson-correlation": "correlation",
        "spearman-correlation": "correlation",
        "kendall-tau": "correlation",
        "correlation": "correlation",
        "simple-linear-regression": "regression",
        "linear_regression": "regression",
        "multiple-linear-regression": "regression",
        "polynomial-regression": "regression",
        "moving-average": "time_series",
        "time-series": "time_series",
        "arima": "time_series",
        "mean": "descriptive",
        "median": "descriptive",
        "variance": "descriptive",
        "descriptives": "descriptive",
        "normality-test": "descriptive",
    }

    def __init__(self):
        """Initialize the QAAgent."""
        pass

    async def validate(
        self,
        response: AnalystResponse,
        raw_stats_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Validate an analyst response.

        Args:
            response: The AnalystResponse to validate
            raw_stats_result: The raw result from run_stats for cross-checking

        Returns:
            (is_valid, errors) tuple where is_valid is False if blocking errors found
        """
        errors: List[ValidationError] = []

        # 1. Check JSON serializability
        json_errors = self._validate_json_serializable(response)
        errors.extend(json_errors)

        # 2. Validate narrative consistency
        if response.chosen_method and response.results:
            narrative_errors = self._validate_narrative_consistency(
                response.chosen_method.analysis_slug,
                response.results.raw,
                response.results.interpretation,
            )
            errors.extend(narrative_errors)

        # 3. Validate charts match test type
        if response.chosen_method and response.visuals.charts:
            chart_errors = self._validate_charts_match_test(
                response.chosen_method.analysis_slug,
                response.visuals.charts,
            )
            errors.extend(chart_errors)

        # 4. Cross-check with raw stats result
        if raw_stats_result:
            crosscheck_errors = self._cross_check_numbers(
                response.results.key_numbers,
                raw_stats_result,
            )
            errors.extend(crosscheck_errors)

        # 5. Validate no invented values
        invention_errors = self._check_for_invented_values(response, raw_stats_result)
        errors.extend(invention_errors)

        # 6. Validate required fields
        required_errors = self._validate_required_fields(response)
        errors.extend(required_errors)

        # Determine if valid (no blocking errors)
        blocking_errors = [e for e in errors if e.severity == "error"]
        is_valid = len(blocking_errors) == 0

        return is_valid, errors

    def _validate_json_serializable(self, response: AnalystResponse) -> List[ValidationError]:
        """Check that the response is JSON serializable."""
        errors = []

        try:
            # Try to serialize the full response
            response_dict = response.model_dump()
            json.dumps(response_dict, default=str)
        except (TypeError, ValueError, OverflowError) as e:
            errors.append(ValidationError(
                category="json_serialization",
                message=f"Response is not JSON serializable: {str(e)}",
                severity="error"
            ))

        # Check individual components for numpy/pandas types
        if response.results and response.results.raw:
            try:
                safe_raw = json_safe(response.results.raw)
                json.dumps(safe_raw)
            except Exception as e:
                errors.append(ValidationError(
                    category="json_serialization",
                    message=f"Raw results contain non-serializable types: {str(e)}",
                    severity="error"
                ))

        # Check chart specs
        for chart in response.visuals.charts:
            try:
                json.dumps(chart.spec)
            except Exception as e:
                errors.append(ValidationError(
                    category="json_serialization",
                    message=f"Chart '{chart.title}' spec not serializable: {str(e)}",
                    severity="error"
                ))

        return errors

    def _validate_narrative_consistency(
        self,
        analysis_slug: str,
        raw_results: Dict[str, Any],
        interpretation: Any,
    ) -> List[ValidationError]:
        """Check that narrative matches actual test and numbers."""
        errors = []

        if not interpretation:
            return errors

        # Get interpretation texts
        plain = getattr(interpretation, 'plain_english', '') or ''
        statistical = getattr(interpretation, 'statistical', '') or ''

        # Check that test name is mentioned appropriately
        test_keywords = {
            "t-test": ["t-test", "t test", "mean difference", "means"],
            "anova": ["anova", "analysis of variance", "groups differ", "f-test"],
            "chi-square": ["chi-square", "chi square", "χ²", "independence", "association"],
            "correlation": ["correlation", "relationship", "associated", "r ="],
            "regression": ["regression", "predict", "coefficient", "r²", "r-squared"],
        }

        category = self.ANALYSIS_CATEGORIES.get(analysis_slug, "")
        if category and category in test_keywords:
            keywords = test_keywords[category]
            full_text = (plain + " " + statistical).lower()
            if not any(kw.lower() in full_text for kw in keywords):
                errors.append(ValidationError(
                    category="narrative_consistency",
                    message=f"Narrative doesn't mention expected keywords for {category} analysis",
                    severity="warning"
                ))

        # Check p-value consistency
        if 'p_value' in raw_results or 'p-value' in raw_results:
            p_val = raw_results.get('p_value') or raw_results.get('p-value')
            if p_val is not None and isinstance(p_val, (int, float)):
                # Check if significance interpretation matches p-value
                is_significant = p_val < 0.05
                if is_significant and "not significant" in plain.lower():
                    errors.append(ValidationError(
                        category="narrative_consistency",
                        message=f"p-value ({p_val:.4f}) indicates significance but narrative says 'not significant'",
                        severity="error"
                    ))
                elif not is_significant and "significant" in plain.lower() and "not significant" not in plain.lower():
                    errors.append(ValidationError(
                        category="narrative_consistency",
                        message=f"p-value ({p_val:.4f}) indicates non-significance but narrative says 'significant'",
                        severity="warning"
                    ))

        return errors

    def _validate_charts_match_test(
        self,
        analysis_slug: str,
        charts: List[ChartSpec],
    ) -> List[ValidationError]:
        """Check that chart types are appropriate for the analysis."""
        errors = []

        category = self.ANALYSIS_CATEGORIES.get(analysis_slug)
        if not category:
            return errors

        expected_types = self.EXPECTED_CHARTS.get(category, set())
        if not expected_types:
            return errors

        chart_types = {c.type.lower() for c in charts}

        # Check if at least one expected chart type is present
        matching = chart_types & expected_types
        if not matching:
            errors.append(ValidationError(
                category="chart_mismatch",
                message=f"No appropriate charts for {category} analysis. Expected: {expected_types}, got: {chart_types}",
                severity="warning"
            ))

        # Check for obviously wrong chart types
        inappropriate_mappings = {
            "t-test": {"line", "line_ma", "run_chart"},
            "chi-square": {"scatter", "regression", "residual"},
            "regression": {"stacked_bar", "normalized_bar"},
            "time_series": {"boxplot", "violin"},
        }

        inappropriate = inappropriate_mappings.get(category, set())
        wrong_charts = chart_types & inappropriate
        if wrong_charts:
            errors.append(ValidationError(
                category="chart_mismatch",
                message=f"Inappropriate chart types for {category}: {wrong_charts}",
                severity="warning"
            ))

        return errors

    def _cross_check_numbers(
        self,
        key_numbers: Any,
        raw_result: Dict[str, Any],
    ) -> List[ValidationError]:
        """Cross-check key numbers against raw stats result."""
        errors = []

        if not key_numbers:
            return errors

        # Extract nested results if present
        result_data = raw_result.get('results', raw_result)

        # Check p-value
        if key_numbers.p_value is not None:
            raw_p = result_data.get('p_value') or result_data.get('p-value') or result_data.get('pvalue')
            if raw_p is not None and isinstance(raw_p, (int, float)):
                if not math.isclose(key_numbers.p_value, raw_p, rel_tol=1e-6):
                    errors.append(ValidationError(
                        category="number_mismatch",
                        message=f"p-value mismatch: key_numbers={key_numbers.p_value}, raw={raw_p}",
                        severity="error"
                    ))

        # Check n (sample size)
        if key_numbers.n is not None:
            raw_n = result_data.get('n') or result_data.get('sample_size') or result_data.get('n_total')
            if raw_n is not None and key_numbers.n != raw_n:
                errors.append(ValidationError(
                    category="number_mismatch",
                    message=f"Sample size mismatch: key_numbers={key_numbers.n}, raw={raw_n}",
                    severity="error"
                ))

        # Check effect size
        if key_numbers.effect_size is not None:
            raw_effect = (
                result_data.get('effect_size') or
                result_data.get('cohens_d') or
                result_data.get('eta_squared')
            )
            if raw_effect is not None and isinstance(raw_effect, (int, float)):
                if not math.isclose(key_numbers.effect_size, raw_effect, rel_tol=1e-4):
                    errors.append(ValidationError(
                        category="number_mismatch",
                        message=f"Effect size mismatch: key_numbers={key_numbers.effect_size}, raw={raw_effect}",
                        severity="warning"
                    ))

        return errors

    def _check_for_invented_values(
        self,
        response: AnalystResponse,
        raw_result: Optional[Dict[str, Any]],
    ) -> List[ValidationError]:
        """Check for potentially invented values not in raw results."""
        errors = []

        if not raw_result:
            return errors

        # Extract all numeric values from raw result
        raw_numbers = self._extract_numbers(raw_result)

        # Check key numbers
        if response.results and response.results.key_numbers:
            kn = response.results.key_numbers
            key_vals = [
                kn.p_value, kn.effect_size, kn.n, kn.statistic,
                kn.r_squared, kn.correlation, kn.mean_diff, kn.chi_square
            ]
            for val in key_vals:
                if val is not None and isinstance(val, (int, float)):
                    # Allow n to be computed
                    if val == kn.n:
                        continue
                    # Check if this value exists in raw (with tolerance)
                    found = any(
                        math.isclose(val, raw_val, rel_tol=1e-4)
                        for raw_val in raw_numbers
                        if isinstance(raw_val, (int, float)) and not math.isnan(raw_val)
                    )
                    if not found and raw_numbers:
                        errors.append(ValidationError(
                            category="invented_value",
                            message=f"Value {val} not found in raw stats results",
                            severity="warning"
                        ))

        return errors

    def _extract_numbers(self, obj: Any, depth: int = 0) -> List[float]:
        """Recursively extract numeric values from a nested structure."""
        if depth > 10:
            return []

        numbers = []

        if isinstance(obj, (int, float)):
            if not math.isnan(obj) and not math.isinf(obj):
                numbers.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                numbers.extend(self._extract_numbers(v, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                numbers.extend(self._extract_numbers(item, depth + 1))

        return numbers

    def _validate_required_fields(self, response: AnalystResponse) -> List[ValidationError]:
        """Validate that required fields are present."""
        errors = []

        if not response.analysis_id:
            errors.append(ValidationError(
                category="missing_field",
                message="Missing analysis_id",
                severity="error"
            ))

        if response.status == "ok":
            if not response.chosen_method:
                errors.append(ValidationError(
                    category="missing_field",
                    message="Status is 'ok' but chosen_method is missing",
                    severity="error"
                ))

            if not response.results or not response.results.raw:
                errors.append(ValidationError(
                    category="missing_field",
                    message="Status is 'ok' but results.raw is empty",
                    severity="error"
                ))

        if response.status == "needs_info" and not response.missing_info:
            errors.append(ValidationError(
                category="missing_field",
                message="Status is 'needs_info' but missing_info is empty",
                severity="warning"
            ))

        return errors

    def get_validation_summary(self, errors: List[ValidationError]) -> Dict[str, Any]:
        """Get a summary of validation results."""
        error_count = sum(1 for e in errors if e.severity == "error")
        warning_count = sum(1 for e in errors if e.severity == "warning")
        info_count = sum(1 for e in errors if e.severity == "info")

        categories = {}
        for e in errors:
            if e.category not in categories:
                categories[e.category] = 0
            categories[e.category] += 1

        return {
            "total_issues": len(errors),
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count,
            "is_valid": error_count == 0,
            "categories": categories,
            "details": [e.to_dict() for e in errors],
        }
