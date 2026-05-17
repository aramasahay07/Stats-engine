from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import List

from app.models.process_mining import CaseRecord, ConformanceResult, ConformanceViolation


def analyze_conformance(expected_path: List[str], cases: List[CaseRecord]) -> ConformanceResult:
    if not expected_path or not cases:
        return ConformanceResult(
            expected_path=expected_path,
            fitness=0.0,
            perfect_share=0.0,
            cases_compliant=0,
            cases_total=len(cases),
            violations=[],
        )

    fitness_scores: list[float] = []
    perfect = 0
    violations: Counter[tuple[str, str]] = Counter()

    for case in cases:
        matcher = SequenceMatcher(a=expected_path, b=case.path)
        score = float(matcher.ratio())
        fitness_scores.append(score)
        if score == 1.0:
            perfect += 1

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "delete":
                for activity in expected_path[i1:i2]:
                    violations[("skipped", activity)] += 1
            elif tag == "insert":
                for activity in case.path[j1:j2]:
                    violations[("extra", activity)] += 1
            else:
                for activity in set(expected_path[i1:i2] + case.path[j1:j2]):
                    violations[("out_of_order", activity)] += 1

    total_cases = len(cases)
    violation_list = [
        ConformanceViolation(
            type=kind,  # type: ignore[arg-type]
            activity=activity,
            case_count=count,
            share=float(count / total_cases),
        )
        for (kind, activity), count in sorted(violations.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]

    return ConformanceResult(
        expected_path=expected_path,
        fitness=float(sum(fitness_scores) / total_cases),
        perfect_share=float(perfect / total_cases),
        cases_compliant=perfect,
        cases_total=total_cases,
        violations=violation_list,
    )
