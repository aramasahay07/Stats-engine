from __future__ import annotations

from typing import Dict, List

from app.transformers.base import Transformer
from app.transformers.steps import ALL_TRANSFORMERS


class TransformerRegistry:
    def __init__(self):
        self._ops: Dict[str, Transformer] = {}

    def register(self, t: Transformer):
        self._ops[t.op] = t

    def get(self, op: str) -> Transformer:
        if op not in self._ops:
            raise ValueError(f"Unknown transform op: {op}")
        return self._ops[op]

    def available_ops(self) -> List[str]:
        return sorted(self._ops.keys())


transformer_registry = TransformerRegistry()

for t in ALL_TRANSFORMERS:
    transformer_registry.register(t)
