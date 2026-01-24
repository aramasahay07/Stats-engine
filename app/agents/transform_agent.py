"""
TransformAgent - Transformation Pipeline Planner.

Converts user intent into a transformer pipeline plan using existing transformer ops.
Outputs reproducible pipeline_steps that can be executed by the pipeline system.
"""

from typing import Any, Dict, List, Optional, Tuple
from .models import TransformStep, TransformPlan


class TransformAgent:
    """
    Agent for planning data transformation pipelines.

    Translates analysis requirements and data prep suggestions into
    executable transformer pipeline steps.
    """

    # Mapping of common intents to transformer operations
    INTENT_TO_OPS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
        "remove_missing": [("drop_nulls", {})],
        "fill_missing_numeric": [("fill_nulls", {"strategy": "median"})],
        "fill_missing_categorical": [("fill_nulls", {"strategy": "mode"})],
        "remove_duplicates": [("remove_duplicates", {})],
        "normalize": [("normalize", {"method": "min_max"})],
        "standardize": [("z_score", {})],
        "log_transform": [("add_computed_column", {"expression": "LOG({column})", "new_column": "{column}_log"})],
        "filter_outliers": [("filter_rows", {"condition": "{column} BETWEEN {lower} AND {upper}"})],
    }

    # Available transformer operations (subset for validation)
    AVAILABLE_OPS = {
        # Table shaping
        "select_columns", "drop_columns", "rename_columns", "reorder_columns",
        "add_constant_column", "duplicate_column", "move_column",
        # Row operations
        "sort_rows", "limit_rows", "offset_rows", "sample_rows", "distinct_rows",
        "remove_duplicates", "add_index_column", "top_n", "bottom_n", "random_sample",
        # Filtering
        "filter_rows", "filter_rows_safe", "filter_top_percent", "filter_by_range",
        # Computed columns
        "add_computed_column", "add_computed_safe", "add_conditional_column", "add_math_column",
        # Data cleaning
        "drop_nulls", "fill_nulls", "replace_values", "change_type", "coalesce", "clean_whitespace",
        # Text operations
        "text_trim", "text_lower", "text_upper", "text_replace", "text_split", "text_merge",
        "text_length", "text_substring", "text_pad", "text_contains", "text_starts_with", "text_ends_with",
        # Datetime operations
        "date_from_text", "date_part", "date_trunc", "format_datetime", "date_diff",
        "date_add", "date_subtract", "age_calculation", "quarter_from_date", "week_of_year",
        # Statistical operations
        "percentile", "quartiles", "z_score", "normalize", "standard_deviation", "variance",
        "correlation", "covariance", "binning", "outlier_detection", "outlier_flag",
        "moving_average", "exponential_moving_average", "cumulative_sum", "cumulative_product",
        "rank_column", "percent_rank", "mode", "rolling_std_dev",
        # Window functions
        "lag_column", "lead_column", "first_value", "last_value", "nth_value",
        "running_total", "running_min", "running_max", "running_average",
        "rolling_min", "rolling_max", "rolling_sum",
        # Aggregation
        "group_aggregate", "weighted_average", "count_distinct", "string_agg",
        "join", "union_all", "pivot", "unpivot",
        # Data quality
        "data_validation", "find_duplicates", "value_frequency",
        "missing_value_flag", "data_type_check", "row_quality_score",
    }

    def __init__(self):
        """Initialize the TransformAgent."""
        pass

    async def plan(
        self,
        schema: List[Dict[str, Any]],
        suggested_fixes: Optional[List[TransformStep]] = None,
        analysis_requirements: Optional[Dict[str, Any]] = None,
        user_intent: Optional[str] = None,
    ) -> TransformPlan:
        """
        Create a transformation pipeline plan.

        Args:
            schema: Dataset schema with column info
            suggested_fixes: Fixes suggested by DataPrepAgent
            analysis_requirements: Requirements from the analysis to be run
            user_intent: Natural language description of desired transformations

        Returns:
            TransformPlan with pipeline_steps and notes
        """
        steps: List[TransformStep] = []
        notes: List[str] = []

        # Start with suggested fixes if provided
        if suggested_fixes:
            for fix in suggested_fixes:
                if self._validate_step(fix):
                    steps.append(fix)
                    notes.append(f"Data prep: {fix.op} on {fix.args.get('column', 'dataset')}")

        # Add steps based on analysis requirements
        if analysis_requirements:
            req_steps, req_notes = self._plan_for_analysis(analysis_requirements, schema)
            steps.extend(req_steps)
            notes.extend(req_notes)

        # Parse user intent if provided
        if user_intent:
            intent_steps, intent_notes = self._parse_intent(user_intent, schema)
            steps.extend(intent_steps)
            notes.extend(intent_notes)

        # Optimize and deduplicate steps
        steps = self._optimize_pipeline(steps)

        return TransformPlan(pipeline_steps=steps, notes=notes)

    def _validate_step(self, step: TransformStep) -> bool:
        """Validate that a transform step uses a known operation."""
        return step.op in self.AVAILABLE_OPS

    def _plan_for_analysis(
        self,
        requirements: Dict[str, Any],
        schema: List[Dict[str, Any]],
    ) -> Tuple[List[TransformStep], List[str]]:
        """Plan transformations needed for a specific analysis."""
        steps: List[TransformStep] = []
        notes: List[str] = []

        analysis_type = requirements.get('analysis_type', '')
        columns = requirements.get('columns', [])

        # Get column roles
        col_roles = {c['name']: c.get('role', 'unknown') for c in schema}

        # For numeric analyses, ensure columns are properly handled
        if analysis_type in ['t-test', 'anova', 'regression', 'correlation']:
            for col in columns:
                if col in col_roles:
                    # Check if needs type conversion
                    role = col_roles[col]
                    if role not in ['numeric']:
                        notes.append(f"Warning: {col} may need conversion to numeric for {analysis_type}")

        # For time series, ensure datetime column is properly parsed
        if analysis_type == 'time_series':
            time_col = requirements.get('time_column')
            if time_col:
                dtype = next(
                    (c.get('dtype', '') for c in schema if c['name'] == time_col),
                    ''
                )
                if 'varchar' in dtype.lower() or 'text' in dtype.lower():
                    steps.append(TransformStep(
                        op="date_from_text",
                        args={"column": time_col, "format": "auto"}
                    ))
                    notes.append(f"Converting {time_col} from text to datetime")

        # Sort by time column for time series
        if analysis_type == 'time_series':
            time_col = requirements.get('time_column')
            if time_col:
                steps.append(TransformStep(
                    op="sort_rows",
                    args={"column": time_col, "ascending": True}
                ))
                notes.append(f"Sorting by {time_col} for time series analysis")

        return steps, notes

    def _parse_intent(
        self,
        intent: str,
        schema: List[Dict[str, Any]],
    ) -> Tuple[List[TransformStep], List[str]]:
        """Parse natural language intent into transform steps."""
        steps: List[TransformStep] = []
        notes: List[str] = []

        intent_lower = intent.lower()
        col_names = [c['name'] for c in schema]

        # Check for common intents
        if 'remove missing' in intent_lower or 'drop null' in intent_lower:
            steps.append(TransformStep(op="drop_nulls", args={}))
            notes.append("Removing rows with missing values")

        if 'fill missing' in intent_lower or 'impute' in intent_lower:
            # Find which columns have missing values
            for col in schema:
                if col.get('missing_pct', 0) > 0:
                    strategy = 'median' if col.get('role') == 'numeric' else 'mode'
                    steps.append(TransformStep(
                        op="fill_nulls",
                        args={"column": col['name'], "strategy": strategy}
                    ))
                    notes.append(f"Filling missing values in {col['name']} with {strategy}")

        if 'remove duplicate' in intent_lower or 'deduplicate' in intent_lower:
            steps.append(TransformStep(op="remove_duplicates", args={}))
            notes.append("Removing duplicate rows")

        if 'normalize' in intent_lower:
            numeric_cols = [c['name'] for c in schema if c.get('role') == 'numeric']
            for col in numeric_cols:
                if col.lower() in intent_lower or not any(c.lower() in intent_lower for c in col_names):
                    steps.append(TransformStep(
                        op="normalize",
                        args={"column": col, "method": "min_max"}
                    ))
                    notes.append(f"Normalizing {col} to 0-1 range")

        if 'standardize' in intent_lower or 'z-score' in intent_lower:
            numeric_cols = [c['name'] for c in schema if c.get('role') == 'numeric']
            for col in numeric_cols:
                if col.lower() in intent_lower or not any(c.lower() in intent_lower for c in col_names):
                    steps.append(TransformStep(
                        op="z_score",
                        args={"column": col}
                    ))
                    notes.append(f"Standardizing {col} (z-score)")

        if 'log transform' in intent_lower:
            numeric_cols = [c['name'] for c in schema if c.get('role') == 'numeric']
            for col in numeric_cols:
                if col.lower() in intent_lower or not any(c.lower() in intent_lower for c in col_names):
                    steps.append(TransformStep(
                        op="add_computed_column",
                        args={
                            "expression": f"LOG(\"{col}\")",
                            "new_column": f"{col}_log"
                        }
                    ))
                    notes.append(f"Adding log-transformed column for {col}")

        if 'filter' in intent_lower:
            # Look for column name and condition
            for col in col_names:
                if col.lower() in intent_lower:
                    notes.append(f"Filter intent detected for {col} - manual condition specification needed")

        return steps, notes

    def _optimize_pipeline(self, steps: List[TransformStep]) -> List[TransformStep]:
        """Optimize and deduplicate pipeline steps."""
        if not steps:
            return steps

        optimized: List[TransformStep] = []
        seen_ops: Dict[str, TransformStep] = {}

        for step in steps:
            # Create a unique key for deduplication
            key = f"{step.op}:{sorted(step.args.items())}"

            if key not in seen_ops:
                seen_ops[key] = step
                optimized.append(step)

        # Reorder for efficiency:
        # 1. Filtering operations first (reduce data size)
        # 2. Column operations
        # 3. Row operations
        # 4. Aggregations last

        filter_ops = {'filter_rows', 'filter_rows_safe', 'filter_top_percent', 'filter_by_range', 'drop_nulls'}
        column_ops = {'select_columns', 'drop_columns', 'rename_columns', 'add_computed_column'}
        agg_ops = {'group_aggregate', 'pivot', 'unpivot'}

        def sort_key(step: TransformStep) -> int:
            if step.op in filter_ops:
                return 0
            elif step.op in column_ops:
                return 1
            elif step.op in agg_ops:
                return 3
            return 2

        optimized.sort(key=sort_key)

        return optimized

    def create_step(self, op: str, **kwargs) -> Optional[TransformStep]:
        """Create a validated transform step."""
        if op not in self.AVAILABLE_OPS:
            return None
        return TransformStep(op=op, args=kwargs)

    def get_available_ops(self) -> List[str]:
        """Get list of available transformer operations."""
        return sorted(self.AVAILABLE_OPS)

    def describe_op(self, op: str) -> str:
        """Get a description of a transformer operation."""
        descriptions = {
            "select_columns": "Select specific columns to keep",
            "drop_columns": "Remove specified columns",
            "rename_columns": "Rename one or more columns",
            "filter_rows": "Filter rows based on a condition",
            "drop_nulls": "Remove rows containing null values",
            "fill_nulls": "Fill null values using a strategy (mean, median, mode, constant)",
            "sort_rows": "Sort rows by one or more columns",
            "remove_duplicates": "Remove duplicate rows",
            "normalize": "Scale values to 0-1 range",
            "z_score": "Standardize values (mean=0, std=1)",
            "add_computed_column": "Create new column from expression",
            "date_from_text": "Parse text column as datetime",
            "group_aggregate": "Group by columns and aggregate",
            "binning": "Create bins/buckets from continuous values",
            "outlier_flag": "Flag outlier values in a column",
        }
        return descriptions.get(op, f"Transformer operation: {op}")
