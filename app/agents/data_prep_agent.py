from __future__ import annotations

from typing import Any, Dict, Iterable, List

from app.agents.models import DataPrepIssue, Severity, TransformStep


class DataPrepAgent:
    """Identify common data preparation issues and recommend fixes."""

    def identify_issues(self, profile: Dict[str, Any]) -> List[DataPrepIssue]:
        issues: List[DataPrepIssue] = []
        row_count = profile.get("row_count") or profile.get("rows") or 0
        columns = profile.get("columns") or profile.get("fields") or []

        for column in columns:
            name = column.get("name") or column.get("column") or column.get("field")
            if not name:
                continue

            missing_count = (
                column.get("missing_count")
                or column.get("null_count")
                or column.get("n_missing")
                or 0
            )
            missing_pct = (
                column.get("missing_pct")
                or column.get("missing_percent")
                or column.get("null_pct")
                or 0
            )

            if missing_count or missing_pct:
                severity = Severity.high if missing_pct >= 0.2 else Severity.medium
                issues.append(
                    DataPrepIssue(
                        issue="missing_values",
                        severity=severity,
                        affected_columns=[name],
                        suggested_fix="impute_missing",
                        details={
                            "missing_count": missing_count,
                            "missing_pct": missing_pct,
                        },
                    )
                )

            invalid_count = column.get("invalid_count") or column.get("n_invalid")
            if invalid_count:
                issues.append(
                    DataPrepIssue(
                        issue="invalid_values",
                        severity=Severity.medium,
                        affected_columns=[name],
                        suggested_fix="cast_or_clean",
                        details={"invalid_count": invalid_count},
                    )
                )

            unique_count = (
                column.get("unique_count")
                or column.get("n_unique")
                or column.get("distinct")
            )
            if row_count and unique_count is not None:
                duplicate_ratio = 1 - (float(unique_count) / float(row_count))
                if duplicate_ratio >= 0.5:
                    issues.append(
                        DataPrepIssue(
                            issue="high_duplicate_ratio",
                            severity=Severity.medium,
                            affected_columns=[name],
                            suggested_fix="deduplicate",
                            details={"duplicate_ratio": duplicate_ratio},
                        )
                    )

        if not columns:
            issues.append(
                DataPrepIssue(
                    issue="missing_profile",
                    severity=Severity.high,
                    suggested_fix="refresh_profile",
                )
            )

        return issues

    def suggest_fixes(self, issues: Iterable[DataPrepIssue]) -> List[TransformStep]:
        steps: List[TransformStep] = []
        for issue in issues:
            if issue.issue == "missing_values":
                steps.append(
                    TransformStep(
                        action="impute_missing",
                        params={"columns": issue.affected_columns},
                        reason="Fill missing values before analysis.",
                    )
                )
            elif issue.issue == "invalid_values":
                steps.append(
                    TransformStep(
                        action="clean_invalid",
                        params={"columns": issue.affected_columns},
                        reason="Normalize invalid values to expected types.",
                    )
                )
            elif issue.issue == "high_duplicate_ratio":
                steps.append(
                    TransformStep(
                        action="deduplicate",
                        params={"columns": issue.affected_columns},
                        reason="Reduce duplicate records for accuracy.",
                    )
                )
            elif issue.issue == "missing_profile":
                steps.append(
                    TransformStep(
                        action="refresh_profile",
                        reason="Refresh dataset profile before planning transforms.",
                    )
                )

        return steps
