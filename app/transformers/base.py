from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Transformer(ABC):
    """A single PowerQuery/PowerBI-like step that rewrites SQL.

    Contract: given prior_sql (a SELECT), return a new SELECT that wraps it.
    """

    op: str

    @abstractmethod
    def apply_sql(self, prior_sql: str, args: Dict[str, Any]) -> str:
        raise NotImplementedError
