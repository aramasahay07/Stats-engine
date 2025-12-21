"""
knowledge/models.py
Pydantic models for consistent API responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConceptBasic(BaseModel):
    slug: str
    title: str
    definition: Optional[str] = None
    plain_english: Optional[str] = None
    when_to_use: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    level: Optional[str] = "intro"
    status: Optional[str] = "published"
    quality_score: Optional[int] = 50
    concept_type: Optional[str] = "definition"
    output_keys: List[str] = Field(default_factory=list)


class TopicBasic(BaseModel):
    slug: str
    title: str
    description: Optional[str] = None
    icon: Optional[str] = None
    sort_order: int = 0


class EnrichedStatsResult(BaseModel):
    raw: Dict[str, Any]
    kb_matches: List[ConceptBasic] = Field(default_factory=list)
    interpretations: Optional[Dict[str, Any]] = None
    suggested_next_steps: List[str] = Field(default_factory=list)
