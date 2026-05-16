from __future__ import annotations

from typing import List


def analyze_conformance(expected_path: List[str]) -> dict:
    return {
        "expected_path": expected_path,
        "average_fitness": None,
        "perfect_share": None,
        "violations": {"skipped": 0, "extra": 0, "out_of_order": 0},
    }
