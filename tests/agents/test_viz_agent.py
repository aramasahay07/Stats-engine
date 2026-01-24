"""Tests for VizAgent chart generation."""

import json
import pytest

from app.agents.viz_agent import VizAgent
from app.agents.models import VisualsResult, ChartSpec


class TestVizAgent:
    """Tests for VizAgent Vega-Lite chart generation."""

    @pytest.fixture
    def agent(self):
        return VizAgent()

    @pytest.fixture
    def sample_data(self):
        return [
            {"group": "A", "value": 10, "x": 1, "y": 2},
            {"group": "A", "value": 12, "x": 2, "y": 4},
            {"group": "B", "value": 15, "x": 3, "y": 5},
            {"group": "B", "value": 18, "x": 4, "y": 8},
        ]

    @pytest.mark.asyncio
    async def test_two_group_charts(self, agent, sample_data):
        """Test chart generation for t-test."""
        result = await agent.generate(
            analysis_slug="two-sample-t-test",
            params={"measure_column": "value", "group_column": "group"},
            results={"p_value": 0.03, "effect_size": 0.5},
            data_sample=sample_data,
        )

        assert isinstance(result, VisualsResult)
        assert len(result.charts) >= 1

        # Should have boxplot and/or histogram
        chart_types = {c.type for c in result.charts}
        assert chart_types & {"boxplot", "histogram"}

    @pytest.mark.asyncio
    async def test_anova_charts(self, agent, sample_data):
        """Test chart generation for ANOVA."""
        result = await agent.generate(
            analysis_slug="anova-one-way",
            params={"measure_column": "value", "group_column": "group"},
            results={"p_value": 0.01, "f_statistic": 5.5},
            data_sample=sample_data,
        )

        assert len(result.charts) >= 1
        chart_types = {c.type for c in result.charts}
        assert chart_types & {"boxplot", "strip", "error_bar"}

    @pytest.mark.asyncio
    async def test_chi_square_charts(self, agent):
        """Test chart generation for chi-square."""
        data = [
            {"cat1": "A", "cat2": "X"},
            {"cat1": "A", "cat2": "Y"},
            {"cat1": "B", "cat2": "X"},
            {"cat1": "B", "cat2": "Y"},
        ]

        result = await agent.generate(
            analysis_slug="chi-square-test",
            params={"x": "cat1", "y": "cat2"},
            results={"chi2": 5.5, "p_value": 0.02},
            data_sample=data,
        )

        chart_types = {c.type for c in result.charts}
        assert chart_types & {"stacked_bar", "normalized_bar", "heatmap"}

    @pytest.mark.asyncio
    async def test_correlation_charts(self, agent, sample_data):
        """Test chart generation for correlation."""
        result = await agent.generate(
            analysis_slug="pearson-correlation",
            params={"x": "x", "y": "y"},
            results={"correlation": 0.95, "p_value": 0.01},
            data_sample=sample_data,
        )

        chart_types = {c.type for c in result.charts}
        assert "scatter" in chart_types

        # Scatter should have trend line (check spec)
        scatter_chart = next(c for c in result.charts if c.type == "scatter")
        spec = scatter_chart.spec
        assert "layer" in spec  # Layered for scatter + trend

    @pytest.mark.asyncio
    async def test_regression_charts(self, agent, sample_data):
        """Test chart generation for regression."""
        result = await agent.generate(
            analysis_slug="simple-linear-regression",
            params={"x": "x", "y": "y"},
            results={"r_squared": 0.85, "p_value": 0.001},
            data_sample=sample_data,
        )

        chart_types = {c.type for c in result.charts}
        assert "regression" in chart_types or "scatter" in chart_types

    @pytest.mark.asyncio
    async def test_regression_with_residuals(self, agent, sample_data):
        """Test regression charts include residual plot when data available."""
        result = await agent.generate(
            analysis_slug="simple-linear-regression",
            params={"x": "x", "y": "y"},
            results={
                "r_squared": 0.85,
                "residuals": [0.1, -0.2, 0.3, -0.1],
                "fitted_values": [2.1, 3.9, 5.2, 7.9],
            },
            data_sample=sample_data,
        )

        chart_types = {c.type for c in result.charts}
        assert "residual" in chart_types

    @pytest.mark.asyncio
    async def test_time_series_charts(self, agent):
        """Test chart generation for time series."""
        data = [
            {"date": "2024-01-01", "value": 100},
            {"date": "2024-01-02", "value": 105},
            {"date": "2024-01-03", "value": 103},
            {"date": "2024-01-04", "value": 110},
        ]

        result = await agent.generate(
            analysis_slug="moving-average",
            params={"time_column": "date", "measure_column": "value"},
            results={"moving_average": [None, 102.5, 104, 106.5]},
            data_sample=data,
        )

        chart_types = {c.type for c in result.charts}
        assert chart_types & {"line", "line_ma", "run_chart"}

    @pytest.mark.asyncio
    async def test_descriptive_charts(self, agent, sample_data):
        """Test chart generation for descriptive statistics."""
        result = await agent.generate(
            analysis_slug="mean",
            params={"column": "value"},
            results={"mean": 13.75, "std": 3.5},
            data_sample=sample_data,
        )

        chart_types = {c.type for c in result.charts}
        assert chart_types & {"histogram", "boxplot"}

    @pytest.mark.asyncio
    async def test_chart_spec_is_valid_vegalite(self, agent, sample_data):
        """Test that generated specs are valid Vega-Lite."""
        result = await agent.generate(
            analysis_slug="two-sample-t-test",
            params={"measure_column": "value", "group_column": "group"},
            results={"p_value": 0.03},
            data_sample=sample_data,
        )

        for chart in result.charts:
            # Should have required Vega-Lite fields
            assert "$schema" in chart.spec
            assert "vega-lite" in chart.spec["$schema"]

            # Should be JSON serializable
            json.dumps(chart.spec)

            # Should have encoding or layer
            assert "encoding" in chart.spec or "layer" in chart.spec

    @pytest.mark.asyncio
    async def test_chart_includes_insight(self, agent, sample_data):
        """Test that charts include insights."""
        result = await agent.generate(
            analysis_slug="pearson-correlation",
            params={"x": "x", "y": "y"},
            results={"correlation": 0.95},
            data_sample=sample_data,
        )

        for chart in result.charts:
            assert chart.insight  # Should have non-empty insight
            assert isinstance(chart.insight, str)

    @pytest.mark.asyncio
    async def test_custom_chart_creation(self, agent):
        """Test custom chart creation."""
        chart = agent.create_custom_chart(
            chart_type="custom_bar",
            title="Custom Chart",
            spec_overrides={
                "mark": "bar",
                "encoding": {
                    "x": {"field": "category", "type": "nominal"},
                    "y": {"field": "count", "type": "quantitative"}
                }
            },
            data=[{"category": "A", "count": 10}],
            insight="Test insight",
        )

        assert isinstance(chart, ChartSpec)
        assert chart.type == "custom_bar"
        assert chart.title == "Custom Chart"
        assert chart.insight == "Test insight"
        assert "data" in chart.spec
