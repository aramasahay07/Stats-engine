#!/usr/bin/env python
"""
import_kb_all_tables_v4.py

✅ Imports ALL KB tables from your v3 seed files.
✅ Works even if stat_resources.url does NOT have a usable UNIQUE constraint.
   - We de-duplicate resources by URL in Python (skip URLs that already exist).
✅ Safe to re-run when KB_REPLACE_CHILDREN=1 (default recommended).

Environment variables:
- SUPABASE_URL                    e.g. https://xxxxx.supabase.co
- SUPABASE_SERVICE_ROLE_KEY       your service role key
- KB_SEED_PATH                    path to seed json (part file)
- KB_REPLACE_CHILDREN             "1" to delete+recreate children for concepts in this seed (default 1)

Notes:
- This script uses Supabase REST (PostgREST) via requests.
- It upserts topics and concepts by slug.
- It then inserts child rows: aliases, links, formulas, examples, prerequisites, resources.
"""
import os
import sys
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import requests
except ImportError:
    print("❌ Missing dependency: requests. Install with: python -m pip install requests")
    sys.exit(1)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # IMPORTANT exact name
SEED_PATH = os.getenv("KB_SEED_PATH")
REPLACE_CHILDREN = os.getenv("KB_REPLACE_CHILDREN", "1") == "1"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment.")
    sys.exit(1)

if not SEED_PATH:
    print("❌ Missing KB_SEED_PATH in environment.")
    sys.exit(1)

REST_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def _req(method: str, path: str, params: Optional[Dict[str, str]] = None, json_body: Any = None,
         prefer: Optional[str] = None, timeout: int = 60):
    headers = dict(HEADERS)
    if prefer:
        headers["Prefer"] = prefer
    url = REST_BASE + path
    resp = requests.request(method, url, headers=headers, params=params, json=json_body, timeout=timeout)
    return resp


def _raise_for(resp, context: str):
    if resp.status_code >= 400:
        raise RuntimeError(f"{context} ({resp.status_code}): {resp.text}")


def load_seed(path: str) -> Dict[str, Any]:
    print(f"📥 Loading seed from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def flatten_topics(seed: Dict[str, Any]) -> List[Dict[str, Any]]:
    return seed.get("topics", [])


def upsert_topics(topics: List[Dict[str, Any]]) -> Dict[str, str]:
    rows = []
    for t in topics:
        rows.append({
            "slug": t["slug"],
            "title": t.get("title"),
            "description": t.get("description"),
            "icon": t.get("icon"),
            "sort_order": t.get("sort_order", 0),
        })
    if not rows:
        return {}

    print("🧩 Upserting topics...")
    resp = _req("POST", "/stat_topics?on_conflict=slug", json_body=rows, prefer="resolution=merge-duplicates,return=representation")
    _raise_for(resp, "Upsert stat_topics failed")
    data = resp.json()
    print(f"✅ Topics upserted: {len(data)}")
    return {r["slug"]: r["id"] for r in data}


def upsert_concepts(topics: List[Dict[str, Any]], topic_id_by_slug: Dict[str, str]) -> Dict[str, str]:
    rows = []
    for t in topics:
        topic_slug = t["slug"]
        topic_id = topic_id_by_slug.get(topic_slug)
        for c in t.get("concepts", []):
            rows.append({
                "slug": c["slug"],
                "topic_id": topic_id,
                "title": c.get("title"),
                "definition": c.get("definition"),
                "plain_english": c.get("plain_english"),
                "when_to_use": c.get("when_to_use"),
                "tags": c.get("tags", []),
                "level": c.get("level", "intro"),
                "status": c.get("status", "published"),
                "quality_score": c.get("quality_score", 50),
                "concept_type": c.get("concept_type", "definition"),
                "output_keys": c.get("output_keys", []),

                # Extended fields (if present in your schema)
                "assumptions": c.get("assumptions"),
                "limitations": c.get("limitations"),
                "improvement_goal": c.get("improvement_goal"),
                "diagnostic_questions": c.get("diagnostic_questions", []),
                "improvement_playbook": c.get("improvement_playbook", {}),
            })

    if not rows:
        return {}

    print("🧠 Upserting concepts...")
    resp = _req("POST", "/stat_concepts?on_conflict=slug", json_body=rows, prefer="resolution=merge-duplicates,return=representation")
    _raise_for(resp, "Upsert stat_concepts failed")
    data = resp.json()
    print(f"✅ Concepts upserted: {len(data)}")
    return {r["slug"]: r["id"] for r in data}


def delete_children_for_concepts(concept_ids: List[str]):
    # Delete children tables for only the concept_ids in this seed
    if not concept_ids:
        return
    # PostgREST filter: concept_id=in.(...)
    in_filter = "in.(" + ",".join(concept_ids) + ")"

    def _del(path: str, col: str = "concept_id"):
        resp = _req("DELETE", path, params={col: in_filter})
        _raise_for(resp, f"Delete {path} failed")

    print("🧹 REPLACE mode ON: deleting existing child rows for concepts in this seed...")
    _del("/stat_concept_aliases", "concept_id")
    # links: two cols
    resp = _req("DELETE", "/stat_concept_links", params={"from_concept_id": in_filter})
    _raise_for(resp, "Delete stat_concept_links failed")
    _del("/stat_formulas", "concept_id")
    _del("/stat_examples", "concept_id")
    _del("/stat_prerequisites", "concept_id")
    _del("/stat_resources", "concept_id")


def insert_aliases(topics: List[Dict[str, Any]], concept_id_by_slug: Dict[str, str]) -> int:
    rows = []
    for t in topics:
        for c in t.get("concepts", []):
            cid = concept_id_by_slug.get(c["slug"])
            if not cid:
                continue
            for alias in c.get("aliases", []) or []:
                rows.append({"concept_id": cid, "alias": alias})
    if not rows:
        return 0
    print(f"🧾 Inserting aliases: {len(rows)}")
    resp = _req("POST", "/stat_concept_aliases", json_body=rows, prefer="resolution=ignore-duplicates")
    _raise_for(resp, "Insert stat_concept_aliases failed")
    return len(rows)


def insert_links(topics: List[Dict[str, Any]], concept_id_by_slug: Dict[str, str]) -> int:
    rows = []
    for t in topics:
        for c in t.get("concepts", []):
            from_id = concept_id_by_slug.get(c["slug"])
            if not from_id:
                continue
            for link in c.get("links", []) or []:
                to_slug = link.get("to_slug")
                to_id = concept_id_by_slug.get(to_slug)
                if not to_id:
                    continue
                rows.append({
                    "from_concept_id": from_id,
                    "to_concept_id": to_id,
                    "link_type": link.get("link_type", "related"),
                    "note": link.get("note"),
                })
    if not rows:
        return 0
    print(f"🔗 Upserting links: {len(rows)}")
    # link PK is (from,to,link_type) so ignore duplicates is fine for re-runs
    resp = _req("POST", "/stat_concept_links", json_body=rows, prefer="resolution=ignore-duplicates")
    _raise_for(resp, "Insert stat_concept_links failed")
    return len(rows)


def insert_formulas(topics: List[Dict[str, Any]], concept_id_by_slug: Dict[str, str]) -> int:
    rows = []
    for t in topics:
        for c in t.get("concepts", []):
            cid = concept_id_by_slug.get(c["slug"])
            if not cid:
                continue
            for f in c.get("formulas", []) or []:
                # Only send columns that exist in most schemas: concept_id, latex, description, sort_order, variables(optional)
                row = {
                    "concept_id": cid,
                    # label is NOT NULL in many schemas
                    "label": f.get("label") or f.get("name") or c.get("title") or "Formula",
                    "latex": f.get("latex") or f.get("formula") or "",
                    "description": f.get("explanation") or f.get("description"),
                    "sort_order": f.get("sort_order", 0),
                }
                # variables is optional; include if present
                if "variables" in f:
                    row["variables"] = f.get("variables") or {}
                rows.append(row)
    if not rows:
        return 0
    print(f"∑ Inserting formulas: {len(rows)}")
    resp = _req("POST", "/stat_formulas", json_body=rows, prefer="resolution=ignore-duplicates")
    _raise_for(resp, "Insert stat_formulas failed")
    return len(rows)


def insert_examples(topics: List[Dict[str, Any]], concept_id_by_slug: Dict[str, str]) -> int:
    rows = []
    for t in topics:
        for c in t.get("concepts", []):
            cid = concept_id_by_slug.get(c["slug"])
            if not cid:
                continue
            for ex in c.get("examples", []) or []:
                rows.append({
                    "concept_id": cid,
                    "title": ex.get("title") or (c.get("title","") + " example"),
                    "scenario": ex.get("scenario"),
                    "solution": ex.get("solution"),
                    "dataset_json": ex.get("dataset_json") or ex.get("dataset_sample"),
                    "sort_order": ex.get("sort_order", 0),
                })
    if not rows:
        return 0
    print(f"🧪 Inserting examples: {len(rows)}")
    resp = _req("POST", "/stat_examples", json_body=rows, prefer="resolution=ignore-duplicates")
    _raise_for(resp, "Insert stat_examples failed")
    return len(rows)


def insert_prereqs(topics: List[Dict[str, Any]], concept_id_by_slug: Dict[str, str]) -> int:
    rows = []
    for t in topics:
        for c in t.get("concepts", []):
            cid = concept_id_by_slug.get(c["slug"])
            if not cid:
                continue
            for p in c.get("prerequisites", []) or []:
                # supports {"prerequisite_slug": "...", "importance": "..."} OR "slug-string"
                if isinstance(p, str):
                    prereq_slug = p
                    importance = "recommended"
                else:
                    prereq_slug = p.get("prerequisite_slug") or p.get("slug")
                    importance = p.get("importance", "recommended")
                pid = concept_id_by_slug.get(prereq_slug)
                if not pid:
                    continue
                rows.append({
                    "concept_id": cid,
                    "prerequisite_id": pid,
                    "importance": importance,
                })
    if not rows:
        return 0
    print(f"🧷 Upserting prerequisites: {len(rows)}")
    resp = _req("POST", "/stat_prerequisites", json_body=rows, prefer="resolution=ignore-duplicates")
    _raise_for(resp, "Insert stat_prerequisites failed")
    return len(rows)


def fetch_existing_resource_urls() -> Set[str]:
    """
    Get existing URLs so we can skip duplicates even without ON CONFLICT support.
    """
    urls: Set[str] = set()
    # Pull in pages (if needed). Your KB is small so this is fine.
    offset = 0
    limit = 1000
    while True:
        params = {
            "select": "url",
            "url": "not.is.null",
            "limit": str(limit),
            "offset": str(offset),
        }
        resp = _req("GET", "/stat_resources", params=params)
        _raise_for(resp, "Fetch stat_resources urls failed")
        data = resp.json()
        if not data:
            break
        for r in data:
            u = r.get("url")
            if u:
                urls.add(u)
        if len(data) < limit:
            break
        offset += limit
    return urls


def insert_resources(topics: List[Dict[str, Any]], concept_id_by_slug: Dict[str, str]) -> int:
    # De-dupe by URL to avoid unique conflicts (works with or without unique constraint).
    existing_urls = fetch_existing_resource_urls()
    seen_urls: Set[str] = set()

    rows = []
    skipped_existing = 0
    skipped_dup_in_seed = 0

    for t in topics:
        for c in t.get("concepts", []):
            cid = concept_id_by_slug.get(c["slug"])
            if not cid:
                continue
            for r in c.get("resources", []) or []:
                url = r.get("url")
                if url:
                    if url in existing_urls:
                        skipped_existing += 1
                        continue
                    if url in seen_urls:
                        skipped_dup_in_seed += 1
                        continue
                    seen_urls.add(url)

                rows.append({
                    "concept_id": cid,
                    "resource_type": r.get("resource_type", "article"),
                    "title": r.get("title"),
                    "url": url,
                    "description": r.get("description"),
                    "source_anchor": r.get("source_anchor"),
                    "page_start": r.get("page_start"),
                    "page_end": r.get("page_end"),
                    "license": r.get("license"),
                    "sort_order": r.get("sort_order", 0),
                })

    if not rows:
        print("📚 No new resources to insert (all duplicates already exist).")
        return 0

    print(f"📚 Inserting resources: {len(rows)} (skipped existing: {skipped_existing}, skipped dup-in-seed: {skipped_dup_in_seed})")
    resp = _req("POST", "/stat_resources", json_body=rows, prefer="resolution=ignore-duplicates")
    _raise_for(resp, "Insert stat_resources failed")
    return len(rows)


def main():
    try:
        seed = load_seed(SEED_PATH)
        topics = flatten_topics(seed)
        print(f"   Topics: {len(topics)}")

        topic_id_by_slug = upsert_topics(topics)
        concept_id_by_slug = upsert_concepts(topics, topic_id_by_slug)

        concept_ids = list(concept_id_by_slug.values())
        if REPLACE_CHILDREN:
            delete_children_for_concepts(concept_ids)

        insert_aliases(topics, concept_id_by_slug)
        insert_links(topics, concept_id_by_slug)
        insert_formulas(topics, concept_id_by_slug)
        insert_examples(topics, concept_id_by_slug)
        insert_prereqs(topics, concept_id_by_slug)
        insert_resources(topics, concept_id_by_slug)

        print("🎉 Import complete.")
        return 0
    except Exception as e:
        import traceback
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
