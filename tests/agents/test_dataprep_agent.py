"""Tests for DataPrepAgent."""

import pytest

from app.agents.dataprep_agent import DataPrepAgent
from app.agents.models import DataPrepResult


class TestDataPrepAgent:
    """Tests for DataPrepAgent data quality analysis."""

    @pytest.fixture
    def agent(self):
        return DataPrepAgent()

    @pytest.fixture
    def sample_schema(self):
        return [
            {"name": "id", "dtype": "INTEGER", "role": "numeric", "missing_pct": 0},
            {"name": "age", "dtype": "DOUBLE", "role": "numeric", "missing_pct": 0.05},
            {"name": "income", "dtype": "DOUBLE", "role": "numeric", "missing_pct": 0.35},
            {"name": "category", "dtype": "VARCHAR", "role": "categorical", "missing_pct": 0, "unique_count": 5},
            {"name": "date_col", "dtype": "VARCHAR", "role": "datetime", "missing_pct": 0},
        ]

    @pytest.fixture
    def sample_profile(self):
        return {
            "n_rows": 1000,
            "n_cols": 5,
            "numeric_summary": {
                "age": {
                    "mean": 35.0,
                    "std": 10.0,
                    "min": 18,
                    "max": 200,  # Outlier
                    "p25": 28,
                    "p75": 42,
                },
                "income": {
                    "mean": 50000,
                    "std": 20000,
                    "min": 10000,
                    "max": 100000,
                    "p25": 35000,
                    "p75": 65000,
                },
            },
        }

    @pytest.mark.asyncio
    async def test_detect_missing_values(self, agent, sample_schema, sample_profile):
        """Test detection of missing value issues."""
        result = await agent.analyze(sample_profile, sample_schema)

        assert isinstance(result, DataPrepResult)
        assert len(result.issues) > 0

        # Should detect high missing rate for income column
        income_issues = [i for i in result.issues if i.column == "income"]
        assert len(income_issues) > 0
        assert any(i.severity == "high" for i in income_issues)

    @pytest.mark.asyncio
    async def test_suggest_fixes_for_missing(self, agent, sample_schema, sample_profile):
        """Test that fixes are suggested for missing values."""
        result = await agent.analyze(sample_profile, sample_schema)

        # Should suggest fill_nulls or drop_columns
        fix_ops = [f.op for f in result.suggested_fixes]
        assert any(op in fix_ops for op in ["fill_nulls", "drop_columns"])

    @pytest.mark.asyncio
    async def test_detect_outliers(self, agent, sample_schema, sample_profile):
        """Test detection of outliers using IQR method."""
        result = await agent.analyze(sample_profile, sample_schema)

        # Age column has outlier (max=200, well above 3*IQR)
        age_issues = [i for i in result.issues if i.column == "age" and "outlier" in i.description.lower()]
        assert len(age_issues) > 0

    @pytest.mark.asyncio
    async def test_detect_date_parsing_issues(self, agent, sample_schema, sample_profile):
        """Test detection of date columns stored as strings."""
        result = await agent.analyze(sample_profile, sample_schema)

        # date_col is VARCHAR but role is datetime
        date_issues = [i for i in result.issues if i.column == "date_col"]
        # Should detect it needs parsing
        assert any("datetime" in i.description.lower() or "text" in i.description.lower()
                   for i in date_issues) or any(f.op == "date_from_text" and f.args.get("column") == "date_col"
                                                 for f in result.suggested_fixes)

    @pytest.mark.asyncio
    async def test_target_specific_columns(self, agent, sample_schema, sample_profile):
        """Test analyzing only specific columns."""
        result = await agent.analyze(
            sample_profile,
            sample_schema,
            target_columns=["age"]
        )

        # Only age issues should be reported
        for issue in result.issues:
            if issue.column:
                assert issue.column == "age"

    @pytest.mark.asyncio
    async def test_empty_profile(self, agent):
        """Test handling empty profile."""
        result = await agent.analyze({}, [])
        assert isinstance(result, DataPrepResult)
        assert len(result.issues) == 0

    def test_quality_score_calculation(self, agent):
        """Test data quality score calculation."""
        # No issues = perfect score
        result = DataPrepResult(issues=[], suggested_fixes=[])
        summary = agent.get_summary(result)
        assert summary["data_quality_score"] == 100.0

        # With issues, score should decrease
        from app.agents.models import DataIssue
        result_with_issues = DataPrepResult(
            issues=[
                DataIssue(severity="high", column="col1", description="test"),
                DataIssue(severity="med", column="col2", description="test"),
            ],
            suggested_fixes=[]
        )
        summary = agent.get_summary(result_with_issues)
        assert summary["data_quality_score"] < 100.0
        assert summary["high_severity"] == 1
        assert summary["medium_severity"] == 1
