"""
VizAgent - Visualization Generation Agent.

Generates Vega-Lite specifications aligned to statistical test types:
- t-test/ANOVA: boxplot/violin + histogram by group
- chi-square: stacked bar + counts table chart
- correlation: scatter + trend line
- regression: residual plot + predicted vs actual
- time trend: line/run chart with median line
"""

from typing import Any, Dict, List, Optional
import uuid
from .models import ChartSpec, VisualsResult
from .utils import json_safe


class VizAgent:
    """
    Agent for generating Vega-Lite chart specifications.

    Creates visualizations that align with the statistical test performed,
    helping users understand and interpret results.
    """

    # Vega-Lite schema version
    VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"

    # Default chart dimensions
    DEFAULT_WIDTH = 400
    DEFAULT_HEIGHT = 300

    def __init__(self):
        """Initialize the VizAgent."""
        pass

    async def generate(
        self,
        analysis_slug: str,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]] = None,
    ) -> VisualsResult:
        """
        Generate Vega-Lite charts for the given analysis.

        Args:
            analysis_slug: The statistical analysis performed
            params: Parameters used in the analysis
            results: Results from the analysis
            data_sample: Optional sample data for inline charts

        Returns:
            VisualsResult containing chart specifications
        """
        charts: List[ChartSpec] = []

        # Route to appropriate chart generator
        if analysis_slug in ['two-sample-t-test', 'ttest_2samp', 'welch-t-test']:
            charts = self._generate_two_group_charts(params, results, data_sample)

        elif analysis_slug in ['anova-one-way', 'anova_oneway', 'kruskal-wallis']:
            charts = self._generate_multi_group_charts(params, results, data_sample)

        elif analysis_slug in ['chi-square-test', 'chi_square', 'fisher-exact-test']:
            charts = self._generate_chi_square_charts(params, results, data_sample)

        elif analysis_slug in ['pearson-correlation', 'spearman-correlation', 'kendall-tau', 'correlation']:
            charts = self._generate_correlation_charts(params, results, data_sample)

        elif analysis_slug in ['simple-linear-regression', 'linear_regression', 'multiple-linear-regression']:
            charts = self._generate_regression_charts(params, results, data_sample)

        elif analysis_slug in ['moving-average', 'time-series', 'arima', 'exponential-smoothing']:
            charts = self._generate_time_series_charts(params, results, data_sample)

        elif analysis_slug in ['mean', 'median', 'variance', 'descriptives', 'normality-test']:
            charts = self._generate_descriptive_charts(params, results, data_sample)

        else:
            # Generic fallback
            charts = self._generate_generic_charts(params, results, data_sample)

        return VisualsResult(charts=charts)

    def _generate_two_group_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate charts for two-group comparison (t-test)."""
        charts = []
        measure_col = params.get('measure_column', params.get('y', 'value'))
        group_col = params.get('group_column', params.get('group', 'group'))

        # 1. Boxplot by group
        boxplot_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Distribution of {measure_col} by {group_col}",
            "mark": {"type": "boxplot", "extent": 1.5},
            "encoding": {
                "x": {"field": group_col, "type": "nominal", "title": group_col},
                "y": {"field": measure_col, "type": "quantitative", "title": measure_col},
                "color": {"field": group_col, "type": "nominal", "legend": None}
            }
        }

        if data_sample:
            boxplot_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Boxplot: {measure_col} by {group_col}",
            type="boxplot",
            spec=boxplot_spec,
            insight="Compare the medians and spread between groups. Non-overlapping boxes suggest a meaningful difference."
        ))

        # 2. Histogram by group (overlaid)
        histogram_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Distribution of {measure_col} by {group_col}",
            "mark": {"type": "bar", "opacity": 0.7},
            "encoding": {
                "x": {
                    "field": measure_col,
                    "type": "quantitative",
                    "bin": {"maxbins": 20},
                    "title": measure_col
                },
                "y": {"aggregate": "count", "title": "Count"},
                "color": {"field": group_col, "type": "nominal"},
            },
            "config": {"view": {"stroke": "transparent"}}
        }

        if data_sample:
            histogram_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Histogram: {measure_col} by {group_col}",
            type="histogram",
            spec=histogram_spec,
            insight="Overlapping distributions indicate similarity; separated distributions suggest a real difference."
        ))

        return charts

    def _generate_multi_group_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate charts for multi-group comparison (ANOVA)."""
        charts = []
        measure_col = params.get('measure_column', params.get('y', 'value'))
        group_col = params.get('group_column', params.get('group', 'group'))

        # 1. Boxplot by group
        boxplot_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Distribution of {measure_col} across {group_col}",
            "mark": {"type": "boxplot", "extent": 1.5},
            "encoding": {
                "x": {"field": group_col, "type": "nominal", "title": group_col, "sort": None},
                "y": {"field": measure_col, "type": "quantitative", "title": measure_col},
                "color": {"field": group_col, "type": "nominal"}
            }
        }

        if data_sample:
            boxplot_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Boxplot: {measure_col} by {group_col}",
            type="boxplot",
            spec=boxplot_spec,
            insight="Look for groups with clearly different medians. Large variance within groups reduces statistical power."
        ))

        # 2. Strip plot (jittered points)
        strip_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Individual Values: {measure_col} by {group_col}",
            "mark": {"type": "circle", "opacity": 0.6, "size": 40},
            "encoding": {
                "x": {
                    "field": group_col,
                    "type": "nominal",
                    "title": group_col,
                    "axis": {"labelAngle": -45}
                },
                "y": {"field": measure_col, "type": "quantitative", "title": measure_col},
                "color": {"field": group_col, "type": "nominal", "legend": None},
                "xOffset": {"field": group_col, "type": "nominal"}
            },
            "transform": [{"calculate": "random()", "as": "jitter"}]
        }

        if data_sample:
            strip_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Strip Plot: {measure_col} by {group_col}",
            type="strip",
            spec=strip_spec,
            insight="Individual data points reveal outliers and the actual distribution within each group."
        ))

        # 3. Mean with error bars
        error_bar_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Mean ± SE: {measure_col} by {group_col}",
            "layer": [
                {
                    "mark": {"type": "errorbar", "extent": "stderr"},
                    "encoding": {
                        "x": {"field": group_col, "type": "nominal"},
                        "y": {"field": measure_col, "type": "quantitative"}
                    }
                },
                {
                    "mark": {"type": "point", "filled": True, "size": 100},
                    "encoding": {
                        "x": {"field": group_col, "type": "nominal"},
                        "y": {"field": measure_col, "type": "quantitative", "aggregate": "mean"},
                        "color": {"field": group_col, "type": "nominal", "legend": None}
                    }
                }
            ]
        }

        if data_sample:
            error_bar_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Mean with Error Bars",
            type="error_bar",
            spec=error_bar_spec,
            insight="Error bars show standard error. Non-overlapping bars suggest significant differences."
        ))

        return charts

    def _generate_chi_square_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate charts for chi-square test."""
        charts = []
        x_col = params.get('x', 'category1')
        y_col = params.get('y', 'category2')

        # 1. Stacked bar chart
        stacked_bar_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Distribution of {y_col} by {x_col}",
            "mark": "bar",
            "encoding": {
                "x": {"field": x_col, "type": "nominal", "title": x_col},
                "y": {"aggregate": "count", "title": "Count"},
                "color": {"field": y_col, "type": "nominal", "title": y_col}
            }
        }

        if data_sample:
            stacked_bar_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Stacked Bar: {y_col} by {x_col}",
            type="stacked_bar",
            spec=stacked_bar_spec,
            insight="Uneven proportions across categories suggest an association between variables."
        ))

        # 2. Normalized stacked bar (proportions)
        normalized_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Proportion of {y_col} by {x_col}",
            "mark": "bar",
            "encoding": {
                "x": {"field": x_col, "type": "nominal", "title": x_col},
                "y": {
                    "aggregate": "count",
                    "stack": "normalize",
                    "title": "Proportion",
                    "axis": {"format": ".0%"}
                },
                "color": {"field": y_col, "type": "nominal", "title": y_col}
            }
        }

        if data_sample:
            normalized_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Proportions: {y_col} by {x_col}",
            type="normalized_bar",
            spec=normalized_spec,
            insight="100% stacked bars make it easy to compare proportions across categories."
        ))

        # 3. Heatmap of counts
        heatmap_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Contingency Table: {x_col} × {y_col}",
            "mark": "rect",
            "encoding": {
                "x": {"field": x_col, "type": "nominal", "title": x_col},
                "y": {"field": y_col, "type": "nominal", "title": y_col},
                "color": {
                    "aggregate": "count",
                    "type": "quantitative",
                    "title": "Count",
                    "scale": {"scheme": "blues"}
                }
            },
            "config": {"axis": {"grid": True, "tickBand": "extent"}}
        }

        if data_sample:
            heatmap_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Heatmap: {x_col} × {y_col}",
            type="heatmap",
            spec=heatmap_spec,
            insight="Darker cells indicate higher frequencies. Uneven distribution suggests association."
        ))

        return charts

    def _generate_correlation_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate charts for correlation analysis."""
        charts = []
        x_col = params.get('x', 'x')
        y_col = params.get('y', 'y')

        # Extract correlation value from results
        r_value = results.get('results', {}).get('correlation', results.get('correlation', None))

        # 1. Scatter plot with trend line
        scatter_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Relationship: {x_col} vs {y_col}",
            "layer": [
                {
                    "mark": {"type": "circle", "opacity": 0.6},
                    "encoding": {
                        "x": {"field": x_col, "type": "quantitative", "title": x_col},
                        "y": {"field": y_col, "type": "quantitative", "title": y_col}
                    }
                },
                {
                    "mark": {"type": "line", "color": "firebrick"},
                    "transform": [
                        {"regression": y_col, "on": x_col}
                    ],
                    "encoding": {
                        "x": {"field": x_col, "type": "quantitative"},
                        "y": {"field": y_col, "type": "quantitative"}
                    }
                }
            ]
        }

        if data_sample:
            scatter_spec["data"] = {"values": json_safe(data_sample)}

        insight = "Points clustered around the trend line indicate a strong relationship."
        if r_value is not None:
            direction = "positive" if r_value > 0 else "negative"
            insight = f"r = {r_value:.3f} indicates a {direction} relationship."

        charts.append(ChartSpec(
            title=f"Scatter: {x_col} vs {y_col}",
            type="scatter",
            spec=scatter_spec,
            insight=insight
        ))

        # 2. Hexbin for large datasets
        hexbin_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Density: {x_col} vs {y_col}",
            "mark": "rect",
            "encoding": {
                "x": {"field": x_col, "type": "quantitative", "bin": {"maxbins": 20}, "title": x_col},
                "y": {"field": y_col, "type": "quantitative", "bin": {"maxbins": 20}, "title": y_col},
                "color": {
                    "aggregate": "count",
                    "type": "quantitative",
                    "scale": {"scheme": "greenblue"}
                }
            }
        }

        if data_sample:
            hexbin_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Density Heatmap: {x_col} vs {y_col}",
            type="hexbin",
            spec=hexbin_spec,
            insight="Darker regions show where most data points cluster."
        ))

        return charts

    def _generate_regression_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate charts for regression analysis."""
        charts = []
        x_col = params.get('x', 'x')
        y_col = params.get('y', 'y')

        # Extract R² from results
        r_squared = results.get('results', {}).get('r_squared', results.get('r_squared'))

        # 1. Scatter with regression line
        reg_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Regression: {y_col} ~ {x_col}",
            "layer": [
                {
                    "mark": {"type": "circle", "opacity": 0.5},
                    "encoding": {
                        "x": {"field": x_col, "type": "quantitative", "title": x_col},
                        "y": {"field": y_col, "type": "quantitative", "title": y_col}
                    }
                },
                {
                    "mark": {"type": "line", "color": "#e45756"},
                    "transform": [{"regression": y_col, "on": x_col}],
                    "encoding": {
                        "x": {"field": x_col, "type": "quantitative"},
                        "y": {"field": y_col, "type": "quantitative"}
                    }
                }
            ]
        }

        if data_sample:
            reg_spec["data"] = {"values": json_safe(data_sample)}

        insight = "The red line shows the best-fit linear relationship."
        if r_squared is not None:
            pct = r_squared * 100
            insight = f"R² = {r_squared:.3f} means {pct:.1f}% of variance in {y_col} is explained by {x_col}."

        charts.append(ChartSpec(
            title=f"Regression Line: {y_col} ~ {x_col}",
            type="regression",
            spec=reg_spec,
            insight=insight
        ))

        # 2. Residuals plot (if residuals available in results)
        residuals = results.get('results', {}).get('residuals')
        fitted = results.get('results', {}).get('fitted_values')

        if residuals and fitted and len(residuals) == len(fitted):
            residual_data = [{"fitted": f, "residual": r} for f, r in zip(fitted, residuals)]

            residual_spec = {
                "$schema": self.VEGA_LITE_SCHEMA,
                "width": self.DEFAULT_WIDTH,
                "height": self.DEFAULT_HEIGHT,
                "title": "Residuals vs Fitted Values",
                "data": {"values": json_safe(residual_data)},
                "layer": [
                    {
                        "mark": {"type": "circle", "opacity": 0.5},
                        "encoding": {
                            "x": {"field": "fitted", "type": "quantitative", "title": "Fitted Values"},
                            "y": {"field": "residual", "type": "quantitative", "title": "Residuals"}
                        }
                    },
                    {
                        "mark": {"type": "rule", "color": "gray", "strokeDash": [4, 4]},
                        "encoding": {"y": {"datum": 0}}
                    }
                ]
            }

            charts.append(ChartSpec(
                title="Residual Plot",
                type="residual",
                spec=residual_spec,
                insight="Points should scatter randomly around zero. Patterns suggest model issues."
            ))

        # 3. Q-Q plot for normality of residuals (simplified)
        if residuals:
            # Note: Full Q-Q plot would need quantile calculations
            hist_residual_spec = {
                "$schema": self.VEGA_LITE_SCHEMA,
                "width": self.DEFAULT_WIDTH,
                "height": self.DEFAULT_HEIGHT // 2,
                "title": "Distribution of Residuals",
                "data": {"values": [{"residual": r} for r in residuals]},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "residual", "type": "quantitative", "bin": {"maxbins": 30}},
                    "y": {"aggregate": "count"}
                }
            }

            charts.append(ChartSpec(
                title="Residual Distribution",
                type="histogram",
                spec=hist_residual_spec,
                insight="Residuals should be approximately normally distributed (bell-shaped)."
            ))

        return charts

    def _generate_time_series_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate charts for time series analysis."""
        charts = []
        time_col = params.get('time_column', params.get('date', 'date'))
        measure_col = params.get('measure_column', params.get('y', 'value'))

        # 1. Line chart
        line_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH * 1.5,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Time Series: {measure_col}",
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": time_col, "type": "temporal", "title": "Time"},
                "y": {"field": measure_col, "type": "quantitative", "title": measure_col}
            }
        }

        if data_sample:
            line_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Time Series: {measure_col}",
            type="line",
            spec=line_spec,
            insight="Look for trends (upward/downward), seasonality (repeating patterns), and anomalies."
        ))

        # 2. Line with moving average (if computed)
        ma_values = results.get('results', {}).get('moving_average') or results.get('moving_average')

        if ma_values and data_sample:
            # Combine original with MA
            ma_data = []
            for i, row in enumerate(data_sample):
                entry = dict(row)
                entry['moving_avg'] = ma_values[i] if i < len(ma_values) else None
                ma_data.append(entry)

            ma_spec = {
                "$schema": self.VEGA_LITE_SCHEMA,
                "width": self.DEFAULT_WIDTH * 1.5,
                "height": self.DEFAULT_HEIGHT,
                "title": f"{measure_col} with Moving Average",
                "data": {"values": json_safe(ma_data)},
                "layer": [
                    {
                        "mark": {"type": "line", "opacity": 0.4},
                        "encoding": {
                            "x": {"field": time_col, "type": "temporal"},
                            "y": {"field": measure_col, "type": "quantitative"}
                        }
                    },
                    {
                        "mark": {"type": "line", "color": "red", "strokeWidth": 2},
                        "encoding": {
                            "x": {"field": time_col, "type": "temporal"},
                            "y": {"field": "moving_avg", "type": "quantitative"}
                        }
                    }
                ]
            }

            charts.append(ChartSpec(
                title="Moving Average Trend",
                type="line_ma",
                spec=ma_spec,
                insight="The red line (moving average) smooths out noise to reveal the underlying trend."
            ))

        # 3. Run chart with median line
        run_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH * 1.5,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Run Chart: {measure_col}",
            "layer": [
                {
                    "mark": {"type": "line", "point": True},
                    "encoding": {
                        "x": {"field": time_col, "type": "temporal"},
                        "y": {"field": measure_col, "type": "quantitative"}
                    }
                },
                {
                    "mark": {"type": "rule", "color": "orange", "strokeDash": [4, 4]},
                    "encoding": {
                        "y": {"field": measure_col, "aggregate": "median"}
                    }
                }
            ]
        }

        if data_sample:
            run_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title="Run Chart with Median",
            type="run_chart",
            spec=run_spec,
            insight="The dashed line shows the median. Long runs above/below suggest systematic shifts."
        ))

        return charts

    def _generate_descriptive_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate charts for descriptive statistics."""
        charts = []
        column = params.get('column', params.get('x', 'value'))

        # 1. Histogram
        hist_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Distribution of {column}",
            "mark": "bar",
            "encoding": {
                "x": {"field": column, "type": "quantitative", "bin": {"maxbins": 30}, "title": column},
                "y": {"aggregate": "count", "title": "Frequency"}
            }
        }

        if data_sample:
            hist_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Histogram: {column}",
            type="histogram",
            spec=hist_spec,
            insight="The shape reveals if data is symmetric, skewed, or has multiple modes."
        ))

        # 2. Boxplot
        box_spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": 150,
            "height": self.DEFAULT_HEIGHT,
            "title": f"Boxplot: {column}",
            "mark": {"type": "boxplot", "extent": 1.5},
            "encoding": {
                "y": {"field": column, "type": "quantitative", "title": column}
            }
        }

        if data_sample:
            box_spec["data"] = {"values": json_safe(data_sample)}

        charts.append(ChartSpec(
            title=f"Boxplot: {column}",
            type="boxplot",
            spec=box_spec,
            insight="Box shows IQR (25th-75th percentile), line is median. Points outside whiskers are outliers."
        ))

        return charts

    def _generate_generic_charts(
        self,
        params: Dict[str, Any],
        results: Dict[str, Any],
        data_sample: Optional[List[Dict[str, Any]]],
    ) -> List[ChartSpec]:
        """Generate generic charts when specific type is unknown."""
        charts = []

        # Try to infer columns from params
        x_col = params.get('x') or params.get('column') or params.get('measure_column')
        y_col = params.get('y')
        group_col = params.get('group') or params.get('group_column')

        if x_col and y_col:
            # Scatter plot
            scatter_spec = {
                "$schema": self.VEGA_LITE_SCHEMA,
                "width": self.DEFAULT_WIDTH,
                "height": self.DEFAULT_HEIGHT,
                "title": f"{x_col} vs {y_col}",
                "mark": "circle",
                "encoding": {
                    "x": {"field": x_col, "type": "quantitative"},
                    "y": {"field": y_col, "type": "quantitative"}
                }
            }

            if group_col:
                scatter_spec["encoding"]["color"] = {"field": group_col, "type": "nominal"}

            if data_sample:
                scatter_spec["data"] = {"values": json_safe(data_sample)}

            charts.append(ChartSpec(
                title=f"Scatter: {x_col} vs {y_col}",
                type="scatter",
                spec=scatter_spec,
                insight="Explore the relationship between these two variables."
            ))

        elif x_col:
            # Single variable histogram
            hist_spec = {
                "$schema": self.VEGA_LITE_SCHEMA,
                "width": self.DEFAULT_WIDTH,
                "height": self.DEFAULT_HEIGHT,
                "title": f"Distribution of {x_col}",
                "mark": "bar",
                "encoding": {
                    "x": {"field": x_col, "type": "quantitative", "bin": True},
                    "y": {"aggregate": "count"}
                }
            }

            if data_sample:
                hist_spec["data"] = {"values": json_safe(data_sample)}

            charts.append(ChartSpec(
                title=f"Histogram: {x_col}",
                type="histogram",
                spec=hist_spec,
                insight="View the distribution of values."
            ))

        return charts

    def create_custom_chart(
        self,
        chart_type: str,
        title: str,
        spec_overrides: Dict[str, Any],
        data: Optional[List[Dict[str, Any]]] = None,
        insight: str = "",
    ) -> ChartSpec:
        """Create a custom Vega-Lite chart with provided specifications."""
        spec = {
            "$schema": self.VEGA_LITE_SCHEMA,
            "width": self.DEFAULT_WIDTH,
            "height": self.DEFAULT_HEIGHT,
            "title": title,
            **spec_overrides
        }

        if data:
            spec["data"] = {"values": json_safe(data)}

        return ChartSpec(
            title=title,
            type=chart_type,
            spec=spec,
            insight=insight
        )
