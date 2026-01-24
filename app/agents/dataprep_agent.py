"""
DataPrepAgent - Data Quality Analysis Agent.

Analyzes dataset profiles to detect data quality issues including:
- Missing values
- Invalid types
- Duplicates
- Outliers
- Date parsing issues

Outputs issues with severity levels and suggested fixes using existing transformers.
"""

from typing import Any, Dict, List, Optional
from .models import DataIssue, TransformStep, DataPrepResult


class DataPrepAgent:
    """
    Agent for analyzing data quality and suggesting preparation steps.

    Uses existing dataset profile to detect issues without modifying data.
    Suggests fixes using available transformer operations.
    """

    # Thresholds for issue severity
    MISSING_HIGH_THRESHOLD = 0.3  # >30% missing -> high severity
    MISSING_MED_THRESHOLD = 0.1   # >10% missing -> medium severity
    OUTLIER_IQR_MULTIPLIER = 3.0  # 3x IQR for outlier detection
    CARDINALITY_HIGH_THRESHOLD = 0.95  # >95% unique in categorical -> warning

    def __init__(self):
        """Initialize the DataPrepAgent."""
        pass

    async def analyze(
        self,
        profile: Dict[str, Any],
        schema: List[Dict[str, Any]],
        target_columns: Optional[List[str]] = None,
    ) -> DataPrepResult:
        """
        Analyze dataset profile for data quality issues.

        Args:
            profile: Dataset profile including numeric_summary, sample_rows
            schema: Column schema with dtypes and roles
            target_columns: Optional list of columns to focus on

        Returns:
            DataPrepResult with issues and suggested fixes
        """
        issues: List[DataIssue] = []
        fixes: List[TransformStep] = []

        # Build column info lookup
        col_info = {col['name']: col for col in schema}

        # Filter to target columns if specified
        columns_to_check = target_columns or list(col_info.keys())

        # Check each column
        for col_name in columns_to_check:
            if col_name not in col_info:
                continue

            col = col_info[col_name]
            col_issues, col_fixes = self._analyze_column(
                col_name=col_name,
                col_info=col,
                profile=profile,
            )
            issues.extend(col_issues)
            fixes.extend(col_fixes)

        # Check for duplicate rows (dataset-level)
        dup_issues, dup_fixes = self._check_duplicates(profile, schema)
        issues.extend(dup_issues)
        fixes.extend(dup_fixes)

        # Check for date parsing issues
        date_issues, date_fixes = self._check_date_columns(schema, profile)
        issues.extend(date_issues)
        fixes.extend(date_fixes)

        # Sort by severity
        severity_order = {'high': 0, 'med': 1, 'low': 2}
        issues.sort(key=lambda x: severity_order.get(x.severity, 3))

        return DataPrepResult(issues=issues, suggested_fixes=fixes)

    def _analyze_column(
        self,
        col_name: str,
        col_info: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> tuple[List[DataIssue], List[TransformStep]]:
        """Analyze a single column for issues."""
        issues: List[DataIssue] = []
        fixes: List[TransformStep] = []

        missing_pct = col_info.get('missing_pct', 0)
        role = col_info.get('role', 'unknown')
        dtype = col_info.get('dtype', '')

        # Check missing values
        if missing_pct > 0:
            if missing_pct >= self.MISSING_HIGH_THRESHOLD:
                severity = 'high'
                issues.append(DataIssue(
                    severity=severity,
                    column=col_name,
                    description=f"High missing rate: {missing_pct:.1%} of values are missing. Consider dropping column or careful imputation."
                ))
                # Suggest dropping if >50% missing
                if missing_pct > 0.5:
                    fixes.append(TransformStep(
                        op="drop_columns",
                        args={"columns": [col_name]}
                    ))
                else:
                    # Suggest fill for numeric, mode for categorical
                    if role == 'numeric':
                        fixes.append(TransformStep(
                            op="fill_nulls",
                            args={"column": col_name, "strategy": "median"}
                        ))
                    else:
                        fixes.append(TransformStep(
                            op="fill_nulls",
                            args={"column": col_name, "strategy": "mode"}
                        ))
            elif missing_pct >= self.MISSING_MED_THRESHOLD:
                severity = 'med'
                issues.append(DataIssue(
                    severity=severity,
                    column=col_name,
                    description=f"Moderate missing rate: {missing_pct:.1%} of values are missing."
                ))
                if role == 'numeric':
                    fixes.append(TransformStep(
                        op="fill_nulls",
                        args={"column": col_name, "strategy": "median"}
                    ))
            else:
                issues.append(DataIssue(
                    severity='low',
                    column=col_name,
                    description=f"Low missing rate: {missing_pct:.1%} of values are missing."
                ))

        # Check for outliers in numeric columns
        if role == 'numeric':
            numeric_summary = profile.get('numeric_summary', {}).get(col_name, {})
            outlier_issues, outlier_fixes = self._check_outliers(
                col_name, numeric_summary
            )
            issues.extend(outlier_issues)
            fixes.extend(outlier_fixes)

        # Check for high cardinality in categorical columns
        if role == 'categorical':
            unique_count = col_info.get('unique_count')
            n_rows = profile.get('n_rows', 0)
            if unique_count and n_rows > 0:
                cardinality = unique_count / n_rows
                if cardinality > self.CARDINALITY_HIGH_THRESHOLD:
                    issues.append(DataIssue(
                        severity='low',
                        column=col_name,
                        description=f"High cardinality: {unique_count} unique values ({cardinality:.1%}). May be ID-like or free text."
                    ))

        return issues, fixes

    def _check_outliers(
        self,
        col_name: str,
        numeric_summary: Dict[str, Any],
    ) -> tuple[List[DataIssue], List[TransformStep]]:
        """Check for outliers using IQR method."""
        issues: List[DataIssue] = []
        fixes: List[TransformStep] = []

        # Need quartile info for IQR method
        q1 = numeric_summary.get('p25') or numeric_summary.get('q1')
        q3 = numeric_summary.get('p75') or numeric_summary.get('q3')
        min_val = numeric_summary.get('min')
        max_val = numeric_summary.get('max')
        mean_val = numeric_summary.get('mean')
        std_val = numeric_summary.get('std')

        # If we have quartiles, use IQR method
        if q1 is not None and q3 is not None:
            iqr = q3 - q1
            lower_fence = q1 - (self.OUTLIER_IQR_MULTIPLIER * iqr)
            upper_fence = q3 + (self.OUTLIER_IQR_MULTIPLIER * iqr)

            has_low_outliers = min_val is not None and min_val < lower_fence
            has_high_outliers = max_val is not None and max_val > upper_fence

            if has_low_outliers or has_high_outliers:
                direction = []
                if has_low_outliers:
                    direction.append(f"low (min={min_val:.2g} < {lower_fence:.2g})")
                if has_high_outliers:
                    direction.append(f"high (max={max_val:.2g} > {upper_fence:.2g})")

                issues.append(DataIssue(
                    severity='med',
                    column=col_name,
                    description=f"Potential outliers detected: {', '.join(direction)} using 3x IQR rule."
                ))

                # Suggest outlier flagging
                fixes.append(TransformStep(
                    op="outlier_flag",
                    args={
                        "column": col_name,
                        "method": "iqr",
                        "multiplier": self.OUTLIER_IQR_MULTIPLIER
                    }
                ))

        # Alternative: check using z-score if we have mean/std
        elif mean_val is not None and std_val is not None and std_val > 0:
            if min_val is not None:
                min_z = (min_val - mean_val) / std_val
                if abs(min_z) > 3:
                    issues.append(DataIssue(
                        severity='low',
                        column=col_name,
                        description=f"Minimum value may be outlier (z-score = {min_z:.2f})"
                    ))
            if max_val is not None:
                max_z = (max_val - mean_val) / std_val
                if abs(max_z) > 3:
                    issues.append(DataIssue(
                        severity='low',
                        column=col_name,
                        description=f"Maximum value may be outlier (z-score = {max_z:.2f})"
                    ))

        return issues, fixes

    def _check_duplicates(
        self,
        profile: Dict[str, Any],
        schema: List[Dict[str, Any]],
    ) -> tuple[List[DataIssue], List[TransformStep]]:
        """Check for potential duplicate row issues."""
        issues: List[DataIssue] = []
        fixes: List[TransformStep] = []

        # If profile has duplicate info, use it
        n_rows = profile.get('n_rows', 0)
        n_distinct = profile.get('n_distinct_rows')

        if n_distinct is not None and n_rows > 0:
            dup_count = n_rows - n_distinct
            if dup_count > 0:
                dup_pct = dup_count / n_rows
                severity = 'high' if dup_pct > 0.1 else ('med' if dup_pct > 0.01 else 'low')
                issues.append(DataIssue(
                    severity=severity,
                    column=None,
                    description=f"Duplicate rows detected: {dup_count} ({dup_pct:.1%})"
                ))
                fixes.append(TransformStep(
                    op="remove_duplicates",
                    args={}
                ))

        return issues, fixes

    def _check_date_columns(
        self,
        schema: List[Dict[str, Any]],
        profile: Dict[str, Any],
    ) -> tuple[List[DataIssue], List[TransformStep]]:
        """Check datetime columns for parsing issues."""
        issues: List[DataIssue] = []
        fixes: List[TransformStep] = []

        sample_rows = profile.get('sample_rows', [])

        for col in schema:
            if col.get('role') != 'datetime':
                continue

            col_name = col['name']
            dtype = col.get('dtype', '').lower()

            # Check if stored as string but should be datetime
            if 'varchar' in dtype or 'text' in dtype or 'string' in dtype:
                issues.append(DataIssue(
                    severity='med',
                    column=col_name,
                    description=f"Column appears to be datetime but stored as text. Consider parsing to datetime type."
                ))
                fixes.append(TransformStep(
                    op="date_from_text",
                    args={"column": col_name, "format": "auto"}
                ))

            # Check for timezone issues
            if 'timezone' in dtype or 'tz' in dtype:
                issues.append(DataIssue(
                    severity='low',
                    column=col_name,
                    description=f"Column has timezone information. Ensure consistent timezone handling."
                ))

        return issues, fixes

    def get_summary(self, result: DataPrepResult) -> Dict[str, Any]:
        """Get a summary of data prep analysis."""
        high_count = sum(1 for i in result.issues if i.severity == 'high')
        med_count = sum(1 for i in result.issues if i.severity == 'med')
        low_count = sum(1 for i in result.issues if i.severity == 'low')

        return {
            "total_issues": len(result.issues),
            "high_severity": high_count,
            "medium_severity": med_count,
            "low_severity": low_count,
            "suggested_fixes_count": len(result.suggested_fixes),
            "data_quality_score": self._calculate_quality_score(result),
        }

    def _calculate_quality_score(self, result: DataPrepResult) -> float:
        """Calculate a 0-100 data quality score."""
        if not result.issues:
            return 100.0

        # Deduct points based on severity
        deductions = sum(
            30 if i.severity == 'high' else (15 if i.severity == 'med' else 5)
            for i in result.issues
        )

        return max(0.0, 100.0 - deductions)
