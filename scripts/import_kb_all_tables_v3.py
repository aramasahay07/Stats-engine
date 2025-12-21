#!/usr/bin/env python3
"""
import_kb_all_tables.py — Supabase Knowledge Base importer (ALL TABLES, v3 seed compatible)

✅ Supports your schema (from your Supabase column listing):
- stat_topics(slug, title, description, icon, sort_order)
- stat_concepts(topic_id, slug, title, definition, plain_english, when_to_use, assumptions, limitations,
               level, status, quality_score, concept_type, output_keys, improvement_goal,
               diagnostic_questions, improvement_playbook, tags)
- stat_concept_aliases(concept_id, alias)             (unique is expression index on lower(alias))
- stat_concept_links(from_concept_id, to_concept_id, link_type, note)
- stat_formulas(concept_id, label, latex, explanation, variables, sort_order)
- stat_examples(concept_id, title, scenario, solution, dataset_json, sort_order)
- stat_prerequisites(concept_id, prerequisite_id, importance)
- stat_resources(concept_id, resource_type, title, url, description, source_anchor, page_start, page_end,
                 license, sort_order)

Seed formats supported:
1) Single file:
   {"version":"...", "generated_at":"...", "topics":[{...}]}

2) Split files (parted):
   {"version":"...", "generated_at":"...", "part":2, "parts_total":5, "topics":[{...}]}

Concept child objects supported (inside each concept):
- aliases: ["sd", "std dev", ...]
- links: [{"to_slug":"...", "link_type":"related", "note":"..."}]
- formulas: [{"label":"...", "latex":"...", "explanation":"...", "variables":{...}, "sort_order":0}]
- examples: [{"title":"...", "scenario":"...", "solution":"...", "dataset_json":{...}, "sort_order":0}]
- prerequisites:
      a) list of slugs: ["what-is-data", ...]
      b) list of objects: [{"prerequisite_slug":"what-is-data", "importance":"recommended"}]
- resources: [{"resource_type":"book", "title":"...", "url":"...", "sort_order":0, ...}]

Idempotency:
- Topics & concepts are UPSERTed by slug.
- Links are UPSERTed by (from_concept_id,to_concept_id,link_type).
- Prereqs are UPSERTed by (concept_id, prerequisite_id).
- For aliases/formulas/examples/resources (no simple natural key), we default to REPLACE mode:
    KB_REPLACE_CHILDREN=1 (default) => delete existing children for concepts in the seed, then insert.
    KB_REPLACE_CHILDREN=0           => only insert (may create duplicates on re-runs).

Required env vars:
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY   (service role, NOT anon)

Optional:
- KB_SEED_PATH                (default: seed/seed_kb.json)
- KB_REPLACE_CHILDREN         (default: 1)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# -----------------------------
# Env / Config
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SEED_PATH = os.getenv("KB_SEED_PATH", "seed/seed_kb.json")
REPLACE_CHILDREN = os.getenv("KB_REPLACE_CHILDREN", "1").strip() not in ("0", "false", "False", "no", "NO")

TIMEOUT_SECS = 60
TOPIC_BATCH = 200
CHILD_BATCH = 500
DELETE_CHUNK = 200

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment.")
    sys.exit(1)

def headers(prefer: Optional[str] = None) -> Dict[str, str]:
    h = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h

def http_get(url: str) -> requests.Response:
    return requests.get(url, headers=headers(), timeout=TIMEOUT_SECS)

def http_post(url: str, payload: Any, prefer: Optional[str] = None) -> requests.Response:
    return requests.post(url, headers=headers(prefer), json=payload, timeout=TIMEOUT_SECS)

def http_delete(url: str) -> requests.Response:
    return requests.delete(url, headers=headers(), timeout=TIMEOUT_SECS)

def chunked(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i+size] for i in range(0, len(items), size)]

def safe_run(label: str, fn):
    try:
        return fn()
    except Exception as e:
        raise RuntimeError(f"{label} failed: {e}")

def upsert(table: str, rows: List[Dict[str, Any]], on_conflict: str) -> None:
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}"
    # For upsert: Prefer resolution=merge-duplicates
    resp = http_post(url, rows, prefer="resolution=merge-duplicates,return=minimal")
    if resp.status_code >= 300:
        raise RuntimeError(f"Upsert {table} ({resp.status_code}): {resp.text}")

def insert(table: str, rows: List[Dict[str, Any]], ignore_duplicates: bool = False, on_conflict: Optional[str] = None) -> None:
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if on_conflict:
        url = url + f"?on_conflict={on_conflict}"
    prefer = "return=minimal"
    if ignore_duplicates:
        prefer = "resolution=ignore-duplicates,return=minimal"
    resp = http_post(url, rows, prefer=prefer)
    if resp.status_code >= 300:
        raise RuntimeError(f"Insert {table} ({resp.status_code}): {resp.text}")

def select_by_slugs(table: str, slugs: List[str], cols: str) -> Dict[str, Dict[str, Any]]:
    if not slugs:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for batch in chunked(slugs, 200):
        in_list = ",".join([f'"{s}"' for s in batch])
        url = f"{SUPABASE_URL}/rest/v1/{table}?select={cols}&slug=in.({in_list})"
        resp = http_get(url)
        if resp.status_code >= 300:
            raise RuntimeError(f"Select {table} ({resp.status_code}): {resp.text}")
        for row in resp.json():
            out[row["slug"]] = row
    return out

def delete_children_for_concepts(table: str, concept_ids: List[str]) -> None:
    if not concept_ids:
        return
    for batch in chunked(concept_ids, DELETE_CHUNK):
        in_list = ",".join([f'"{cid}"' for cid in batch])
        url = f"{SUPABASE_URL}/rest/v1/{table}?concept_id=in.({in_list})"
        resp = http_delete(url)
        if resp.status_code >= 300:
            raise RuntimeError(f"Delete {table} ({resp.status_code}): {resp.text}")

def load_seed(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_topics(seed: Dict[str, Any]) -> List[Dict[str, Any]]:
    topics = seed.get("topics")
    if not isinstance(topics, list):
        raise ValueError("Seed JSON must contain a top-level 'topics' list.")
    return topics

def main():
    print(f"📥 Loading seed from: {SEED_PATH}")
    seed = load_seed(SEED_PATH)
    topics_in = normalize_topics(seed)
    print(f"   Topics: {len(topics_in)}")

    # -----------------------------
    # 1) Upsert topics
    # -----------------------------
    topic_rows: List[Dict[str, Any]] = []
    for t in topics_in:
        topic_rows.append({
            "slug": t["slug"],
            "title": t.get("title") or t["slug"],
            "description": t.get("description"),
            "icon": t.get("icon"),
            "sort_order": int(t.get("sort_order") or 0),
        })

    print("🧩 Upserting topics...")
    for batch in chunked(topic_rows, TOPIC_BATCH):
        safe_run("upsert stat_topics", lambda b=batch: upsert("stat_topics", b, on_conflict="slug"))
    print(f"✅ Topics upserted: {len(topic_rows)}")

    topic_map = safe_run("fetch topic ids", lambda: select_by_slugs("stat_topics", [t["slug"] for t in topic_rows], cols="id,slug"))
    if not topic_map:
        raise RuntimeError("Could not fetch topics after upsert.")

    # -----------------------------
    # 2) Upsert concepts
    # -----------------------------
    concept_rows: List[Dict[str, Any]] = []
    seed_concept_by_slug: Dict[str, Dict[str, Any]] = {}

    for t in topics_in:
        topic_id = topic_map[t["slug"]]["id"]
        for c in (t.get("concepts") or []):
            slug = c["slug"]
            seed_concept_by_slug[slug] = c

            row = {
                "topic_id": topic_id,
                "slug": slug,
                "title": c.get("title") or slug,
                "definition": c.get("definition"),
                "plain_english": c.get("plain_english"),
                "when_to_use": c.get("when_to_use"),
                "assumptions": c.get("assumptions"),
                "limitations": c.get("limitations"),
                "level": c.get("level") or "intro",
                "status": c.get("status") or "published",
                "quality_score": int(c.get("quality_score") or 50),
                "concept_type": c.get("concept_type") or "definition",
                "output_keys": c.get("output_keys") or [],
                "improvement_goal": c.get("improvement_goal"),
                "diagnostic_questions": c.get("diagnostic_questions") or [],
                "improvement_playbook": c.get("improvement_playbook") or {},
                "tags": c.get("tags") or [],
            }
            concept_rows.append(row)

    print("🧠 Upserting concepts...")
    for batch in chunked(concept_rows, CHILD_BATCH):
        safe_run("upsert stat_concepts", lambda b=batch: upsert("stat_concepts", b, on_conflict="slug"))
    print(f"✅ Concepts upserted: {len(concept_rows)}")

    print("🔎 Fetching concept ids...")
    concept_map = safe_run("fetch concept ids", lambda: select_by_slugs("stat_concepts", [r["slug"] for r in concept_rows], cols="id,slug"))
    if not concept_map:
        raise RuntimeError("Could not fetch concepts after upsert.")

    concept_ids = [concept_map[s]["id"] for s in concept_map.keys()]

    # -----------------------------
    # 3) Children preparation
    # -----------------------------
    aliases_rows: List[Dict[str, Any]] = []
    links_rows: List[Dict[str, Any]] = []
    formulas_rows: List[Dict[str, Any]] = []
    examples_rows: List[Dict[str, Any]] = []
    prereq_rows: List[Dict[str, Any]] = []
    resources_rows: List[Dict[str, Any]] = []

    # Aliases
    for cslug, c in seed_concept_by_slug.items():
        cid = concept_map[cslug]["id"]
        for a in (c.get("aliases") or []):
            if a:
                aliases_rows.append({"concept_id": cid, "alias": a})

    # Links
    for cslug, c in seed_concept_by_slug.items():
        from_id = concept_map[cslug]["id"]
        for link in (c.get("links") or []):
            to_slug = link.get("to_slug")
            if not to_slug or to_slug not in concept_map or to_slug == cslug:
                continue
            links_rows.append({
                "from_concept_id": from_id,
                "to_concept_id": concept_map[to_slug]["id"],
                "link_type": link.get("link_type") or "related",
                "note": link.get("note"),
            })

    # Formulas
    for cslug, c in seed_concept_by_slug.items():
        cid = concept_map[cslug]["id"]
        for f in (c.get("formulas") or []):
            latex = f.get("latex")
            if not latex:
                continue
            formulas_rows.append({
                "concept_id": cid,
                "label": f.get("label"),
                "latex": latex,
                "explanation": f.get("explanation") or f.get("description"),
                                "sort_order": int(f.get("sort_order") or 0),
            })

    # Examples
    for cslug, c in seed_concept_by_slug.items():
        cid = concept_map[cslug]["id"]
        for ex in (c.get("examples") or []):
            title = ex.get("title")
            if not title:
                continue
            examples_rows.append({
                "concept_id": cid,
                "title": title,
                "scenario": ex.get("scenario"),
                "solution": ex.get("solution"),
                "dataset_json": ex.get("dataset_json") or ex.get("dataset_sample"),
                "sort_order": int(ex.get("sort_order") or 0),
            })

    # Prerequisites
    for cslug, c in seed_concept_by_slug.items():
        cid = concept_map[cslug]["id"]
        prereqs = c.get("prerequisites") or []
        # allow list[str] or list[dict]
        for p in prereqs:
            if isinstance(p, str):
                prereq_slug = p
                importance = "recommended"
            elif isinstance(p, dict):
                prereq_slug = p.get("prerequisite_slug") or p.get("slug") or p.get("prereq_slug")
                importance = p.get("importance") or "recommended"
            else:
                continue
            if not prereq_slug or prereq_slug not in concept_map or prereq_slug == cslug:
                continue
            prereq_rows.append({
                "concept_id": cid,
                "prerequisite_id": concept_map[prereq_slug]["id"],
                "importance": importance,
            })

    # Resources
    for cslug, c in seed_concept_by_slug.items():
        cid = concept_map[cslug]["id"]
        for r in (c.get("resources") or []):
            rtype = r.get("resource_type")
            title = r.get("title")
            if not rtype or not title:
                continue
            resources_rows.append({
                "concept_id": cid,
                "resource_type": rtype,
                "title": title,
                "url": r.get("url"),
                "description": r.get("description"),
                "source_anchor": r.get("source_anchor"),
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
                "license": r.get("license"),
                "sort_order": int(r.get("sort_order") or 0),
            })

    # -----------------------------
    # 4) Replace-mode deletes (optional)
    # -----------------------------
    if REPLACE_CHILDREN:
        print("♻️  REPLACE mode ON: deleting existing child rows for concepts in this seed...")
        # Order matters for FK constraints (delete children before prereqs/links)
        for table in ["stat_concept_aliases", "stat_formulas", "stat_examples", "stat_resources", "stat_prerequisites"]:
            safe_run(f"delete {table}", lambda t=table: delete_children_for_concepts(t, concept_ids))
        # links table uses from_concept_id/to_concept_id; delete by from_concept_id
        for batch in chunked(concept_ids, DELETE_CHUNK):
            in_list = ",".join([f'"{cid}"' for cid in batch])
            url = f"{SUPABASE_URL}/rest/v1/stat_concept_links?from_concept_id=in.({in_list})"
            resp = http_delete(url)
            if resp.status_code >= 300:
                raise RuntimeError(f"Delete stat_concept_links ({resp.status_code}): {resp.text}")

    # -----------------------------
    # 5) Insert / upsert children
    # -----------------------------
    if aliases_rows:
        print(f"🏷️  Inserting aliases: {len(aliases_rows)}")
        for batch in chunked(aliases_rows, CHILD_BATCH):
            # expression unique index => use ignore-duplicates to stay safe if REPLACE mode off
            safe_run("insert stat_concept_aliases", lambda b=batch: insert("stat_concept_aliases", b, ignore_duplicates=True))

    if links_rows:
        print(f"🔗 Upserting links: {len(links_rows)}")
        for batch in chunked(links_rows, CHILD_BATCH):
            safe_run("upsert stat_concept_links", lambda b=batch: upsert("stat_concept_links", b, on_conflict="from_concept_id,to_concept_id,link_type"))

    if formulas_rows:
        print(f"∑ Inserting formulas: {len(formulas_rows)}")
        for batch in chunked(formulas_rows, CHILD_BATCH):
            safe_run("insert stat_formulas", lambda b=batch: insert("stat_formulas", b, ignore_duplicates=False))

    if examples_rows:
        print(f"🧪 Inserting examples: {len(examples_rows)}")
        for batch in chunked(examples_rows, CHILD_BATCH):
            safe_run("insert stat_examples", lambda b=batch: insert("stat_examples", b, ignore_duplicates=False))

    if prereq_rows:
        print(f"🧷 Upserting prerequisites: {len(prereq_rows)}")
        for batch in chunked(prereq_rows, CHILD_BATCH):
            safe_run("upsert stat_prerequisites", lambda b=batch: upsert("stat_prerequisites", b, on_conflict="concept_id,prerequisite_id"))

    if resources_rows:
        print(f"📚 Inserting resources: {len(resources_rows)}")
        # If re-running without REPLACE, ignore duplicates (unique partial index on url only helps when url is not null)
        for batch in chunked(resources_rows, CHILD_BATCH):
            safe_run("insert stat_resources", lambda b=batch: insert("stat_resources", b, ignore_duplicates=True, on_conflict="url"))

    print("🎉 Import complete.")
    print("Next: verify counts in Supabase tables.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        sys.exit(1)
