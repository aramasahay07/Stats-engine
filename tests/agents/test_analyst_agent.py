"""Tests for AIAnalystAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.analyst_agent import AIAnalystAgent
from app.agents.models import (
    AnalystRequest,
    AnalystResponse,
    AnalystContext,
    SelectedColumns,
    DatasetInfo,
)


class TestAIAnalystAgent:
    """Tests for AIAnalystAgent orchestration."""

    @pytest.fixture
    def agent(self):
        return AIAnalystAgent(openai_api_key=None)

    @pytest.fixture
    def dataset_info(self):
        return DatasetInfo(
            dataset_id="test-dataset-123",
            user_id="test-user-456",
            file_name="test.csv",
            n_rows=1000,
            n_cols=5,
            schema=[
                {"name": "age", "dtype": "DOUBLE", "role": "numeric", "missing_pct": 0.02},
                {"name": "income", "dtype": "DOUBLE", "role": "numeric", "missing_pct": 0.05},
                {"name": "gender", "dtype": "VARCHAR", "role": "categorical", "missing_pct": 0, "unique_count": 2},
                {"name": "region", "dtype": "VARCHAR", "role": "categorical", "missing_pct": 0, "unique_count": 4},
                {"name": "date", "dtype": "TIMESTAMP", "role": "datetime", "missing_pct": 0},
            ],
            profile={
                "n_rows": 1000,
                "numeric_summary": {
                    "age": {"mean": 35, "std": 10, "min": 18, "max": 70},
                    "income": {"mean": 50000, "std": 20000, "min": 20000, "max": 150000},
                },
            },
        )

    @pytest.fixture
    def mock_run_stats(self):
        async def _run_stats(user_id, dataset_id, analysis, params):
            # Return mock results based on analysis type
            if "t-test" in analysis or "ttest" in analysis:
                return {
                    "p_value": 0.03,
                    "t_statistic": 2.5,
                    "n": 100,
                    "effect_size": 0.5,
                    "results": {
                        "p_value": 0.03,
                        "statistic": 2.5,
                    }
                }, False
            elif "anova" in analysis:
                return {
                    "p_value": 0.01,
                    "f_statistic": 5.5,
                    "results": {
                        "p_value": 0.01,
                        "f_statistic": 5.5,
                    }
                }, False
            elif "correlation" in analysis:
                return {
                    "correlation": 0.75,
                    "p_value": 0.001,
                    "results": {
                        "correlation": 0.75,
                        "p_value": 0.001,
                    }
                }, False
            elif "chi" in analysis:
                return {
                    "chi2": 10.5,
                    "p_value": 0.005,
                    "results": {
                        "chi2": 10.5,
                        "p_value": 0.005,
                    }
                }, False
            elif "regression" in analysis:
                return {
                    "r_squared": 0.65,
                    "p_value": 0.001,
                    "results": {
                        "r_squared": 0.65,
                        "p_value": 0.001,
                    }
                }, False
            else:
                return {
                    "mean": 42.5,
                    "std": 10.2,
                    "results": {
                        "mean": 42.5,
                        "std": 10.2,
                    }
                }, False

        return _run_stats

    @pytest.fixture
    def sample_data(self):
        return [
            {"age": 25, "income": 40000, "gender": "M", "region": "North"},
            {"age": 35, "income": 60000, "gender": "F", "region": "South"},
            {"age": 45, "income": 80000, "gender": "M", "region": "East"},
            {"age": 30, "income": 50000, "gender": "F", "region": "West"},
        ]

    @pytest.mark.asyncio
    async def test_analyze_two_group_comparison(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test analysis for two-group comparison."""
        request = AnalystRequest(
            question="Compare income between genders",
            context=AnalystContext(
                selected_columns=SelectedColumns(y="income", group="gender"),
                visuals=True,
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert isinstance(response, AnalystResponse)
        assert response.status == "ok"
        assert response.chosen_method is not None
        assert "t-test" in response.chosen_method.analysis_slug.lower() or "anova" in response.chosen_method.analysis_slug.lower()
        assert response.results.key_numbers.p_value is not None

    @pytest.mark.asyncio
    async def test_analyze_correlation(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test analysis for correlation."""
        request = AnalystRequest(
            question="Is there a correlation between age and income?",
            context=AnalystContext(
                selected_columns=SelectedColumns(x="age", y="income"),
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.status == "ok"
        assert "correlation" in response.chosen_method.analysis_slug.lower()
        assert response.results.key_numbers.correlation is not None

    @pytest.mark.asyncio
    async def test_analyze_preferred_test(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test using preferred test from context."""
        request = AnalystRequest(
            question="Analyze this data",
            context=AnalystContext(
                preferred_test="chi-square-test",
                selected_columns=SelectedColumns(x="gender", y="region"),
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.status == "ok"
        assert response.chosen_method.analysis_slug == "chi-square-test"

    @pytest.mark.asyncio
    async def test_needs_info_response(self, agent, mock_run_stats):
        """Test needs_info status when required columns missing."""
        # Dataset with only one numeric column
        limited_dataset = DatasetInfo(
            dataset_id="test-123",
            user_id="user-456",
            file_name="test.csv",
            n_rows=100,
            n_cols=1,
            schema=[
                {"name": "value", "dtype": "DOUBLE", "role": "numeric"},
            ],
            profile={"n_rows": 100},
        )

        request = AnalystRequest(
            question="Compare groups",  # Needs group column
            context=AnalystContext(),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=limited_dataset,
            run_stats_func=mock_run_stats,
        )

        # Should indicate needs_info if group column required but not found
        # Or fall back to descriptive stats
        assert response.status in ["ok", "needs_info"]

    @pytest.mark.asyncio
    async def test_generates_visualizations(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that visualizations are generated when requested."""
        request = AnalystRequest(
            question="Compare income between genders",
            context=AnalystContext(
                selected_columns=SelectedColumns(y="income", group="gender"),
                visuals=True,
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.status == "ok"
        assert len(response.visuals.charts) > 0

    @pytest.mark.asyncio
    async def test_skips_visualizations_when_disabled(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that visualizations are skipped when disabled."""
        request = AnalystRequest(
            question="Compare income between genders",
            context=AnalystContext(
                selected_columns=SelectedColumns(y="income", group="gender"),
                visuals=False,
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.status == "ok"
        assert len(response.visuals.charts) == 0

    @pytest.mark.asyncio
    async def test_generates_interpretation(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that interpretation is generated."""
        request = AnalystRequest(
            question="Is there a difference in income?",
            context=AnalystContext(
                selected_columns=SelectedColumns(y="income", group="gender"),
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.results.interpretation.plain_english
        assert response.results.interpretation.statistical
        assert response.results.interpretation.business_meaning

    @pytest.mark.asyncio
    async def test_generates_next_steps(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that next steps are generated."""
        request = AnalystRequest(
            question="Compare income",
            context=AnalystContext(
                selected_columns=SelectedColumns(y="income", group="gender"),
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert len(response.next_steps) > 0

    @pytest.mark.asyncio
    async def test_includes_data_prep_analysis(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that data prep analysis is included."""
        request = AnalystRequest(
            question="Analyze income",
            context=AnalystContext(),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.data_prep is not None
        # May or may not have issues depending on schema

    @pytest.mark.asyncio
    async def test_includes_transform_plan(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that transform plan is included when allowed."""
        request = AnalystRequest(
            question="Analyze income",
            context=AnalystContext(allow_transform_plan=True),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.transform_plan is not None

    @pytest.mark.asyncio
    async def test_handles_run_stats_error(self, agent, dataset_info, sample_data):
        """Test handling of errors from run_stats."""
        async def error_run_stats(user_id, dataset_id, analysis, params):
            raise ValueError("Analysis failed")

        request = AnalystRequest(
            question="Analyze data",
            context=AnalystContext(),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=error_run_stats,
            data_sample=sample_data,
        )

        assert response.status == "error"
        assert len(response.errors) > 0

    @pytest.mark.asyncio
    async def test_method_selection_reasoning(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that method selection includes reasoning."""
        request = AnalystRequest(
            question="Is there a relationship between age and income?",
            context=AnalystContext(),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        assert response.chosen_method is not None
        assert len(response.chosen_method.why_this_test) > 0

    @pytest.mark.asyncio
    async def test_alternatives_considered(self, agent, dataset_info, mock_run_stats, sample_data):
        """Test that alternatives are considered."""
        request = AnalystRequest(
            question="Compare income between genders",
            context=AnalystContext(
                selected_columns=SelectedColumns(y="income", group="gender"),
            ),
        )

        response = await agent.analyze(
            request=request,
            dataset_info=dataset_info,
            run_stats_func=mock_run_stats,
            data_sample=sample_data,
        )

        # Should have alternatives (or empty list if none applicable)
        assert response.chosen_method.alternatives_considered is not None
