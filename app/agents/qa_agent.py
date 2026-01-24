from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from app.agents.models import Severity, ValidationIssue, ValidationResult


class QAAgent:
    """Validate responses for completeness and obvious issues."""

    def validate_response(
        self,
        response: Mapping[str, Any],
        required_fields: Iterable[str] | None = None,
    ) -> ValidationResult:
        issues: List[ValidationIssue] = []

        if not response:
            issues.append(
                ValidationIssue(
                    message="Response payload is empty.",
                    severity=Severity.high,
                )
            )
        if required_fields:
            for field in required_fields:
                if field not in response or response[field] is None:
                    issues.append(
                        ValidationIssue(
                            message=f"Missing required field: {field}.",
                            field=field,
                            severity=Severity.high,
                        )
                    )

        if isinstance(response, Mapping) and response.get("error"):
            issues.append(
                ValidationIssue(
                    message="Response contains error message.",
                    field="error",
                    severity=Severity.high,
                )
            )

        return ValidationResult(valid=not issues, issues=issues)
