from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ConceptMeta:
    id: str
    topic_id: str
    topic_slug: str
    slug: str
    title: str
    concept_type: str
    level: str
    status: str
    output_keys: List[str]
    tags: List[str]
    quality_score: int
