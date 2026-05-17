from __future__ import annotations

from typing import Iterable
import re


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


def humanize_activity(value: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(value)).strip()
    if not text:
        return ""
    return " ".join(word[:1].upper() + word[1:] for word in text.split())
