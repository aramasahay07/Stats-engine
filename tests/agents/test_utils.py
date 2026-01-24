"""Tests for agent utilities."""

import json
import math
import pytest
import numpy as np

from app.agents.utils import (
    json_safe,
    detect_column_role,
    infer_analysis_from_question,
    get_test_alternatives,
    format_p_value,
    interpret_p_value,
    interpret_effect_size,
    get_required_columns_for_analysis,
)


class TestJsonSafe:
    """Tests for json_safe serialization function."""

    def test_none(self):
        assert json_safe(None) is None

    def test_native_types(self):
        assert json_safe(42) == 42
        assert json_safe(3.14) == 3.14
        assert json_safe("hello") == "hello"
        assert json_safe(True) is True

    def test_numpy_int(self):
        result = json_safe(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self):
        result = json_safe(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_bool(self):
        result = json_safe(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = json_safe(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nan_becomes_none(self):
        assert json_safe(float('nan')) is None
        assert json_safe(np.nan) is None

    def test_inf_becomes_none(self):
        assert json_safe(float('inf')) is None
        assert json_safe(float('-inf')) is None

    def test_nested_dict(self):
        data = {
            "int": np.int64(1),
            "float": np.float64(2.5),
            "nested": {"array": np.array([1, 2])}
        }
        result = json_safe(data)
        assert result == {"int": 1, "float": 2.5, "nested": {"array": [1, 2]}}
        # Verify it's JSON serializable
        json.dumps(result)

    def test_list_with_numpy(self):
        data = [np.int64(1), np.float64(2.5), "string"]
        result = json_safe(data)
        assert result == [1, 2.5, "string"]

    def test_set_conversion(self):
        result = json_safe({1, 2, 3})
        assert sorted(result) == [1, 2, 3]

    def test_complex_nested_structure(self):
        data = {
            "results": {
                "p_value": np.float64(0.03),
                "ci": np.array([1.5, 2.5]),
                "counts": [np.int64(10), np.int64(20)],
            },
            "meta": {"valid": np.bool_(True)}
        }
        result = json_safe(data)

        # Verify structure
        assert result["results"]["p_value"] == 0.03
        assert result["results"]["ci"] == [1.5, 2.5]
        assert result["results"]["counts"] == [10, 20]
        assert result["meta"]["valid"] is True

        # Verify JSON serializable
        json.dumps(result)


class TestDetectColumnRole:
    """Tests for column role detection."""

    def test_numeric_types(self):
        assert detect_column_role("INTEGER") == "numeric"
        assert detect_column_role("DOUBLE") == "numeric"
        assert detect_column_role("FLOAT") == "numeric"
        assert detect_column_role("DECIMAL(10,2)") == "numeric"
        assert detect_column_role("BIGINT") == "numeric"

    def test_datetime_types(self):
        assert detect_column_role("TIMESTAMP") == "datetime"
        assert detect_column_role("DATE") == "datetime"
        assert detect_column_role("DATETIME") == "datetime"
        assert detect_column_role("TIME") == "datetime"

    def test_categorical_from_unique_ratio(self):
        # Low unique ratio -> categorical
        result = detect_column_role(
            "VARCHAR",
            sample_values=["A", "B", "C"],
            unique_count=10,
            n_rows=1000,
        )
        assert result == "categorical"

    def test_text_from_high_unique_ratio(self):
        # High unique ratio -> text
        result = detect_column_role(
            "VARCHAR",
            sample_values=["long text " * 20] * 5,
            unique_count=950,
            n_rows=1000,
        )
        assert result == "text"

    def test_boolean_is_categorical(self):
        assert detect_column_role("BOOLEAN") == "categorical"


class TestInferAnalysisFromQuestion:
    """Tests for analysis inference from natural language questions."""

    @pytest.fixture
    def sample_columns(self):
        return [
            {"name": "age", "role": "numeric"},
            {"name": "income", "role": "numeric"},
            {"name": "gender", "role": "categorical"},
            {"name": "region", "role": "categorical"},
            {"name": "date", "role": "datetime"},
        ]

    def test_correlation_question(self, sample_columns):
        slug, params, reasoning = infer_analysis_from_question(
            "Is there a correlation between age and income?",
            sample_columns,
        )
        assert slug == "pearson-correlation"
        assert "x" in params
        assert "y" in params

    def test_comparison_question(self, sample_columns):
        slug, params, reasoning = infer_analysis_from_question(
            "Compare income between genders",
            sample_columns,
            selected_columns={"y": "income", "group": "gender"},
        )
        assert slug in ["two-sample-t-test", "anova-one-way"]
        assert "measure_column" in params or "y" in params

    def test_regression_question(self, sample_columns):
        slug, params, reasoning = infer_analysis_from_question(
            "Predict income from age",
            sample_columns,
        )
        assert slug == "simple-linear-regression"

    def test_chi_square_question(self, sample_columns):
        slug, params, reasoning = infer_analysis_from_question(
            "Test independence between gender and region",
            sample_columns,
        )
        # Should recognize chi-square or categorical analysis
        assert "chi" in slug or params.get("x") or params.get("y")

    def test_trend_question(self, sample_columns):
        slug, params, reasoning = infer_analysis_from_question(
            "Show the trend of income over time",
            sample_columns,
            selected_columns={"time": "date", "y": "income"},
        )
        assert "time" in slug or "moving" in slug or "time_column" in params

    def test_descriptive_question(self, sample_columns):
        slug, params, reasoning = infer_analysis_from_question(
            "What is the average income?",
            sample_columns,
        )
        assert slug in ["mean", "descriptives"]


class TestGetTestAlternatives:
    """Tests for getting alternative tests."""

    def test_t_test_alternatives(self):
        alts = get_test_alternatives("two-sample-t-test")
        test_names = [a[0] for a in alts]
        assert "mann-whitney-u" in test_names
        assert "welch-t-test" in test_names

    def test_correlation_alternatives(self):
        alts = get_test_alternatives("pearson-correlation")
        test_names = [a[0] for a in alts]
        assert "spearman-correlation" in test_names

    def test_unknown_test(self):
        alts = get_test_alternatives("unknown-test")
        assert alts == []


class TestFormatPValue:
    """Tests for p-value formatting."""

    def test_none(self):
        assert format_p_value(None) == "N/A"

    def test_very_small(self):
        assert format_p_value(0.0001) == "< 0.001"

    def test_small(self):
        result = format_p_value(0.005)
        assert "0.005" in result

    def test_normal(self):
        result = format_p_value(0.05)
        assert "0.05" in result


class TestInterpretPValue:
    """Tests for p-value interpretation."""

    def test_significant(self):
        result = interpret_p_value(0.01)
        assert "significant" in result.lower()

    def test_not_significant(self):
        result = interpret_p_value(0.10)
        assert "not" in result.lower()

    def test_none(self):
        result = interpret_p_value(None)
        assert "unable" in result.lower()


class TestInterpretEffectSize:
    """Tests for effect size interpretation."""

    def test_cohens_d_small(self):
        result = interpret_effect_size(0.15, "cohens_d")
        assert "negligible" in result.lower()

    def test_cohens_d_medium(self):
        result = interpret_effect_size(0.5, "cohens_d")
        assert "medium" in result.lower()

    def test_cohens_d_large(self):
        result = interpret_effect_size(1.0, "cohens_d")
        assert "large" in result.lower()

    def test_correlation(self):
        result = interpret_effect_size(0.6, "r")
        assert "strong" in result.lower()


class TestGetRequiredColumns:
    """Tests for required column parameters."""

    def test_t_test_requirements(self):
        reqs = get_required_columns_for_analysis("two-sample-t-test")
        assert "measure_column" in reqs
        assert "group_column" in reqs

    def test_correlation_requirements(self):
        reqs = get_required_columns_for_analysis("pearson-correlation")
        assert "x" in reqs
        assert "y" in reqs

    def test_unknown_analysis(self):
        reqs = get_required_columns_for_analysis("unknown")
        assert reqs == {}
