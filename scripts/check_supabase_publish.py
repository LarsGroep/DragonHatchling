#!/usr/bin/env python3
"""Diagnose why a dataset does not appear in the deployed ViTreous workbench.

The workbench reads four things over the Supabase **anon** key, and a failure in
any one of them shows up on screen as "the dataset just isn't there". This
script performs exactly the same reads, in the same order, and reports which
step breaks — so you never have to guess between "Supabase is paused", "the
notebook didn't finish", "RLS is blocking anonymous reads" and "Vercel is on
the demo fixture".

Usage::

    python scripts/check_supabase_publish.py \
        --url https://<ref>.supabase.co --anon-key <anon key> [--dataset ham10000]

Both values are the public ones you put in Vercel as NEXT_PUBLIC_SUPABASE_URL
and NEXT_PUBLIC_SUPABASE_ANON_KEY. The anon key is public by design — do not
pass a service-role key to this script; it would grant far more than a
read-only check needs and would mask exactly the RLS problems it exists to
find.

Exit status is 0 only when the dataset is fully publishable to the workbench.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

TIMEOUT = 20


def _get(url: str, anon_key: str, path: str, params: Dict[str, str]) -> Tuple[int, Any]:
    """One PostgREST GET with the anon key. Returns (status, parsed-or-text)."""
    query = urllib.parse.urlencode(params)
    full = f"{url.rstrip('/')}/rest/v1/{path}?{query}"
    req = urllib.request.Request(
        full,
        headers={
            "apikey": anon_key,
            "Authorization": f"Bearer {anon_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except urllib.error.URLError as e:
        return 0, f"{e.reason}"


def _head_object(url: str, bucket: str, key: str) -> int:
    """Is a Storage object publicly readable? Returns the HTTP status."""
    full = f"{url.rstrip('/')}/storage/v1/object/public/{bucket}/{key}"
    req = urllib.request.Request(full, method="GET")
    req.add_header("Range", "bytes=0-0")  # cheap: first byte only
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="https://<ref>.supabase.co")
    ap.add_argument("--anon-key", required=True, help="the PUBLIC anon key (never the service key)")
    ap.add_argument("--dataset", default="ham10000", help="dataset name to check (default: ham10000)")
    ap.add_argument("--bucket", default="packs", help="storage bucket holding packs")
    args = ap.parse_args()

    problems: List[str] = []
    print(f"→ {args.url}   dataset={args.dataset!r}\n")

    # 1. Reachability + auth. A paused project fails here.
    status, body = _get(args.url, args.anon_key, "datasets", {"select": "id", "limit": "1"})
    if status == 0:
        print(f"  ✗ cannot reach the project: {body}")
        print("    → most likely PAUSED: Supabase pauses free projects after a")
        print("      week of inactivity, and a paused project refuses connections")
        print("      while the dashboard still shows it. Resume it, then re-run.")
        print("    → otherwise check the URL, or a proxy/firewall between you and")
        print("      supabase.co (a blocked CONNECT looks identical to a pause).")
        return 1
    if status == 401 or status == 403:
        print(f"  ✗ auth/permission rejected ({status}): {body}")
        print("    → wrong anon key, or RLS denies anonymous SELECT on datasets.")
        return 1
    if status == 404:
        print(f"  ✗ table 'datasets' does not exist ({status})")
        print("    → apply supabase/migrations/0001_init.sql in the SQL editor.")
        return 1
    if status >= 400:
        print(f"  ✗ unexpected error {status}: {body}")
        return 1
    print(f"  ✓ project reachable, 'datasets' readable with the anon key")

    # 2. The dataset row itself.
    status, rows = _get(
        args.url, args.anon_key, "datasets",
        {"select": "id,name,spec", "name": f"eq.{args.dataset}"},
    )
    if status >= 400 or not isinstance(rows, list):
        print(f"  ✗ dataset query failed ({status}): {rows}")
        return 1
    if not rows:
        _, all_rows = _get(args.url, args.anon_key, "datasets", {"select": "name"})
        names = [r.get("name") for r in all_rows] if isinstance(all_rows, list) else []
        print(f"  ✗ no dataset named {args.dataset!r}. Present: {names or '(none)'}")
        print("    → the publishing notebook never inserted it; re-run")
        print("      kaggle/ham10000_live.ipynb to completion.")
        return 1
    ds = rows[0]
    ds_id = ds["id"]
    spec = ds.get("spec") or {}
    print(f"  ✓ dataset row present  (id={ds_id}, {spec.get('num_classes', '?')} classes)")

    # 3. A model row — listDatasets() joins models(id, arch) and the workbench
    #    uses model_id to locate projections and the concept dictionary.
    status, models = _get(
        args.url, args.anon_key, "models", {"select": "id,arch", "dataset_id": f"eq.{ds_id}"}
    )
    if status < 400 and isinstance(models, list) and models:
        print(f"  ✓ model row present    ({models[0].get('arch')})")
    else:
        print("  ✗ no row in 'models' for this dataset")
        print("    → the workbench resolves model_id from this join; projections")
        print("      and the concept tier will not load without it.")
        problems.append("models")

    # 4. Gallery images — without these the workbench has nothing to show.
    status, imgs = _get(
        args.url, args.anon_key, "gallery_images",
        {"select": "id,pack_prefix,thumb_url", "dataset_id": f"eq.{ds_id}", "limit": "200"},
    )
    if status >= 400 or not isinstance(imgs, list):
        print(f"  ✗ gallery_images query failed ({status}): {imgs}")
        return 1
    if not imgs:
        print("  ✗ 0 rows in 'gallery_images' for this dataset")
        print("    → dataset published but the gallery/pack step did not finish.")
        problems.append("gallery_images")
    else:
        print(f"  ✓ {len(imgs)} gallery image(s)")

        # 5. Are the pack assets actually public? A private bucket yields a
        #    dataset that lists fine and then fails to render anything.
        prefix = (imgs[0].get("pack_prefix") or "").lstrip("/")
        if prefix:
            key = f"{prefix.rstrip('/')}/manifest.json"
            code = _head_object(args.url, args.bucket, key)
            if code == 200:
                print(f"  ✓ pack assets public   ({args.bucket}/{key})")
            elif code in (400, 403, 404):
                print(f"  ✗ manifest.json not publicly readable ({code}): {args.bucket}/{key}")
                print("    → the 'packs' bucket must be PUBLIC, or every pack fetch 403s")
                print("      even though the dataset and gallery rows load fine.")
                problems.append("storage")
            else:
                print(f"  ? manifest.json probe returned {code}")

    print()
    if problems:
        print(f"INCOMPLETE — fix: {', '.join(problems)}")
        return 1
    print("OK — this dataset should appear in the workbench.")
    print("If it still does not, the deployment is on the demo fixture: confirm")
    print("NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY are set in")
    print("Vercel and REDEPLOY — NEXT_PUBLIC_* values are inlined at build time,")
    print("so setting them without a rebuild changes nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
