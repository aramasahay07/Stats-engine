from __future__ import annotations

from typing import Iterable, List

from app.agents.models import DataPrepIssue, TransformPlan, TransformStep


class TransformAgent:
    """Plan a transformation pipeline based on data prep issues."""

    def plan(self, issues: Iterable[DataPrepIssue]) -> TransformPlan:
        steps: List[TransformStep] = []
        notes: List[str] = []

        for issue in issues:
            if issue.suggested_fix:
                steps.append(
                    TransformStep(
                        action=issue.suggested_fix,
                        params={"columns": issue.affected_columns},
                        reason=f"Address {issue.issue} before analysis.",
                    )
                )

        if not steps:
            notes.append("No transformations required based on current profile.")

        return TransformPlan(steps=steps, notes=notes)
