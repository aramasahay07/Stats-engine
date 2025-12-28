from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Iterable, List

from ._base import ConceptMeta

def iter_concept_modules() -> Iterable[str]:
    pkg = __name__.rsplit(".", 1)[0]  # app.analytics.concepts
    for m in pkgutil.walk_packages(__path__, prefix=pkg + "."):
        name = m.name
        if name.endswith("._base") or name.endswith(".registry"):
            continue
        yield name

def load_all_meta() -> List[ConceptMeta]:
    metas: List[ConceptMeta] = []
    for modname in iter_concept_modules():
        mod = importlib.import_module(modname)
        meta = getattr(mod, "META", None)
        if meta is not None:
            metas.append(meta)
    metas.sort(key=lambda m: (m.topic_slug, m.slug))
    return metas

def meta_by_slug() -> Dict[str, ConceptMeta]:
    return {m.slug: m for m in load_all_meta()}
