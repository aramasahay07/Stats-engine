"""
knowledge/queries.py
KB query helpers tailored to your schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .client import rest_get


def _raise(resp, context: str) -> None:
    if resp.status_code >= 300:
        raise RuntimeError(f"{context} ({resp.status_code}): {resp.text}")


def get_all_topics() -> List[Dict[str, Any]]:
    resp = rest_get("stat_topics", params={"select": "id,slug,title,description,icon,sort_order", "order": "sort_order.asc"})
    _raise(resp, "get_all_topics failed")
    return resp.json()


def get_concept_by_slug(slug: str, include_children: bool = True) -> Optional[Dict[str, Any]]:
    resp = rest_get("stat_concepts", params={"select": "*", "slug": f"eq.{slug}", "limit": "1"})
    _raise(resp, "get_concept_by_slug failed")
    rows = resp.json()
    if not rows:
        return None
    concept = rows[0]

    if not include_children:
        return concept

    cid = concept["id"]
    concept["aliases"] = _fetch_aliases(cid)
    concept["links_out"] = _fetch_links_out(cid)
    concept["formulas"] = _fetch_formulas(cid)
    concept["examples"] = _fetch_examples(cid)
    concept["prerequisites"] = _fetch_prerequisites(cid)
    concept["resources"] = _fetch_resources(cid)
    return concept


def search_concepts(q: str, limit: int = 20) -> List[Dict[str, Any]]:
    q = (q or "").strip()
    if not q:
        return []

    pattern = f"*{q}*"
    resp = rest_get(
        "stat_concepts",
        params={
            "select": "id,slug,title,definition,plain_english,when_to_use,tags,level,status,quality_score,concept_type,output_keys,topic_id",
            "or": f"(title.ilike.{pattern},definition.ilike.{pattern},plain_english.ilike.{pattern})",
            "limit": str(limit),
        },
    )
    _raise(resp, "search_concepts concept search failed")
    concepts = resp.json()

    resp_a = rest_get(
        "stat_concept_aliases",
        params={"select": "concept_id,alias", "alias": f"ilike.{pattern}", "limit": str(limit)},
    )
    _raise(resp_a, "search_concepts alias search failed")
    alias_hits = resp_a.json()

    missing_ids = {a["concept_id"] for a in alias_hits} - {c["id"] for c in concepts}
    if missing_ids:
        ids_csv = ",".join(missing_ids)
        resp_c = rest_get(
            "stat_concepts",
            params={
                "select": "id,slug,title,definition,plain_english,when_to_use,tags,level,status,quality_score,concept_type,output_keys,topic_id",
                "id": f"in.({ids_csv})",
                "limit": str(limit),
            },
        )
        _raise(resp_c, "search_concepts fetch alias concepts failed")
        concepts.extend(resp_c.json())

    by_id = {c["id"]: c for c in concepts}
    out = list(by_id.values())
    out.sort(key=lambda x: (x.get("quality_score") or 0), reverse=True)
    return out[:limit]


def get_concepts_by_output_keys(keys: List[str], limit: int = 25) -> List[Dict[str, Any]]:
    keys = [k for k in (keys or []) if isinstance(k, str) and k.strip()]
    if not keys:
        return []

    results: List[Dict[str, Any]] = []
    seen = set()

    for k in keys:
        resp = rest_get(
            "stat_concepts",
            params={
                "select": "id,slug,title,definition,plain_english,when_to_use,tags,level,status,quality_score,concept_type,output_keys,topic_id",
                "output_keys": f"cs.{{{k}}}",
                "limit": str(limit),
            },
        )
        _raise(resp, f"get_concepts_by_output_keys failed for key={k}")
        for row in resp.json():
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            results.append(row)

    def score(row: Dict[str, Any]) -> int:
        ok = set(row.get("output_keys") or [])
        exact = len(ok.intersection(keys))
        return exact * 1000 + int(row.get("quality_score") or 0)

    results.sort(key=score, reverse=True)
    return results[:limit]


def _fetch_aliases(concept_id: str) -> List[str]:
    resp = rest_get("stat_concept_aliases", params={"select": "alias", "concept_id": f"eq.{concept_id}"})
    _raise(resp, "fetch aliases failed")
    return [r["alias"] for r in resp.json()]


def _fetch_links_out(concept_id: str) -> List[Dict[str, Any]]:
    resp = rest_get("stat_concept_links", params={"select": "to_concept_id,link_type,note", "from_concept_id": f"eq.{concept_id}"})
    _raise(resp, "fetch links failed")
    return resp.json()


def _fetch_formulas(concept_id: str) -> List[Dict[str, Any]]:
    resp = rest_get("stat_formulas", params={"select": "latex,description,variables,sort_order", "concept_id": f"eq.{concept_id}", "order": "sort_order.asc"})
    _raise(resp, "fetch formulas failed")
    return resp.json()


def _fetch_examples(concept_id: str) -> List[Dict[str, Any]]:
    resp = rest_get("stat_examples", params={"select": "title,scenario,solution,dataset_sample,sort_order", "concept_id": f"eq.{concept_id}", "order": "sort_order.asc"})
    _raise(resp, "fetch examples failed")
    return resp.json()


def _fetch_prerequisites(concept_id: str) -> List[str]:
    resp = rest_get("stat_prerequisites", params={"select": "prerequisite_id", "concept_id": f"eq.{concept_id}"})
    _raise(resp, "fetch prerequisites failed")
    return [r["prerequisite_id"] for r in resp.json()]


def _fetch_resources(concept_id: str) -> List[Dict[str, Any]]:
    resp = rest_get("stat_resources", params={"select": "resource_type,title,url,description,source_anchor,page_start,page_end,license", "concept_id": f"eq.{concept_id}", "order": "created_at.desc"})
    _raise(resp, "fetch resources failed")
    return resp.json()
