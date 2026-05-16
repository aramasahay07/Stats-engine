from __future__ import annotations

from typing import Iterable


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def qstring(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def ensure_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out

