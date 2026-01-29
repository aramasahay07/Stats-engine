from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Iterable, List, Optional

import app.analytics.concepts as concepts_pkg

from ._base import ConceptMeta


# -------------------------
# Internal caches
# -------------------------

_CONCEPT_META_BY_SLUG: Dict[str, ConceptMeta] | None = None
_CONCEPT_MODULE_BY_SLUG: Dict[str, object] | None = None


# -------------------------
# Module discovery
# -------------------------

def iter_concept_modules() -> Iterable[str]:
    """
    Yield all concept module import paths under `app.analytics.concepts`.
    """
    pkg = concepts_pkg.__name__  # "app.analytics.concepts"
    for m in pkgutil.walk_packages(concepts_pkg.__path__, prefix=pkg + "."):
        name = m.name
        if name.endswith("._base") or name.endswith(".registry"):
            continue
        yield name


# -------------------------
# META loading
# -------------------------

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
    global _CONCEPT_META_BY_SLUG

    if _CONCEPT_META_BY_SLUG is None:
        _CONCEPT_META_BY_SLUG = {m.slug: m for m in load_all_meta()}

    return _CONCEPT_META_BY_SLUG


# -------------------------
# Module registry (CRITICAL)
# -------------------------

def load_concept_modules() -> Dict[str, object]:
    """
    Load and cache all concept modules keyed by META.slug.
    """
    global _CONCEPT_MODULE_BY_SLUG

    if _CONCEPT_MODULE_BY_SLUG is not None:
        return _CONCEPT_MODULE_BY_SLUG

    modules: Dict[str, object] = {}

    for modname in iter_concept_modules():
        mod = importlib.import_module(modname)
        meta = getattr(mod, "META", None)

        if meta is None:
            continue

        slug = meta.slug

        if slug in modules:
            raise RuntimeError(
                f"Duplicate concept slug detected: '{slug}' "
                f"in module {modname}"
            )

        modules[slug] = mod

    _CONCEPT_MODULE_BY_SLUG = modules
    return modules


def get_concept_module(slug: str) -> Optional[object]:
    """
    Public API used by stats_service.py.
    Returns the concept module for a given slug, or None if not found.
    """
    modules = load_concept_modules()
    return modules.get(slug)

