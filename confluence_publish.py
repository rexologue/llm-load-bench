#!/usr/bin/env python3
"""
Publish a bench_report Confluence Storage Format page to a self-hosted
Confluence Server / Data Center in one shot:

  1. uploads every PNG in --assets as a page attachment (create or update),
  2. replaces the page body with the storage-format XML from --body.

Auth: a Personal Access Token (Confluence Server/DC 7.9+). Create one under
Profile -> Settings -> Personal Access Tokens. Pass it via --token or the
CONFLUENCE_TOKEN env var.

Usage:
  python confluence_publish.py \
      --base-url https://confluence.corp.local \
      --page-id 123456789 \
      --token "$CONFLUENCE_TOKEN" \
      --body ./plots/confluence_storage.xml \
      --assets ./plots/html_assets

  # see exactly what would happen, no writes:
  python confluence_publish.py ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def get_page(client: httpx.Client, base: str, page_id: str, token: str) -> dict:
    r = client.get(
        f"{base}/rest/api/content/{page_id}",
        params={"expand": "version,space,title"},
        headers={**auth_headers(token), "Accept": "application/json"},
    )
    r.raise_for_status()
    return r.json()


def upload_attachment(client: httpx.Client, base: str, page_id: str, token: str, png: Path) -> None:
    """Create-or-update an attachment. PUT to /child/attachment updates an
    existing file with the same name; if it doesn't exist yet, fall back to POST."""
    url = f"{base}/rest/api/content/{page_id}/child/attachment"
    headers = {**auth_headers(token), "X-Atlassian-Token": "nocheck"}
    files = {"file": (png.name, png.read_bytes(), "image/png")}
    data = {"minorEdit": "true", "comment": "bench_report figure"}

    r = client.put(url, headers=headers, files=files, data=data)
    if r.status_code in (200, 201):
        return
    # Some versions only create via POST; updates may need POST too.
    r2 = client.post(url, headers=headers, files=files, data=data)
    if r2.status_code in (200, 201):
        return
    raise RuntimeError(
        f"attachment '{png.name}' failed: PUT {r.status_code} {r.text[:200]} | "
        f"POST {r2.status_code} {r2.text[:200]}"
    )


def update_body(client: httpx.Client, base: str, page_id: str, token: str,
                page: dict, storage_xml: str, new_title: str | None) -> None:
    version = int(page["version"]["number"]) + 1
    payload = {
        "id": page_id,
        "type": "page",
        "title": new_title or page["title"],
        "space": {"key": page["space"]["key"]},
        "version": {"number": version, "minorEdit": False},
        "body": {"storage": {"value": storage_xml, "representation": "storage"}},
    }
    r = client.put(
        f"{base}/rest/api/content/{page_id}",
        headers={**auth_headers(token), "Content-Type": "application/json", "Accept": "application/json"},
        json=payload,
    )
    r.raise_for_status()


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish bench_report storage XML + attachments to Confluence Server/DC")
    ap.add_argument("--base-url", required=True, help="e.g. https://confluence.corp.local (no trailing /)")
    ap.add_argument("--page-id", required=True, help="numeric page id of an EXISTING page to overwrite")
    ap.add_argument("--token", default=os.environ.get("CONFLUENCE_TOKEN"), help="Personal Access Token (or CONFLUENCE_TOKEN env)")
    ap.add_argument("--body", required=True, help="confluence_storage.xml from bench_report --confluence")
    ap.add_argument("--assets", required=True, help="html_assets/ directory with the PNGs")
    ap.add_argument("--title", default=None, help="optional new page title (keeps existing if omitted)")
    ap.add_argument("--verify-tls", default="true", choices=["true", "false"], help="set false for self-signed corp certs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.token:
        sys.exit("No token. Pass --token or set CONFLUENCE_TOKEN.")

    base = args.base_url.rstrip("/")
    body_path = Path(args.body)
    assets = sorted(Path(args.assets).glob("*.png"))
    storage_xml = body_path.read_text(encoding="utf-8")

    if not assets:
        sys.exit(f"No PNGs found in {args.assets}")

    print(f"Target page : {base}/pages/viewpage.action?pageId={args.page_id}")
    print(f"Body        : {body_path}  ({len(storage_xml)} chars)")
    print(f"Attachments : {len(assets)} PNGs from {args.assets}")

    if args.dry_run:
        print("\n[dry-run] would upload:")
        for a in assets:
            print("  -", a.name)
        print("[dry-run] would bump page version and replace body. No requests sent.")
        return

    verify = args.verify_tls == "true"
    with httpx.Client(timeout=120.0, verify=verify) as client:
        page = get_page(client, base, args.page_id, args.token)
        print(f"Found page '{page['title']}' (space {page['space']['key']}, version {page['version']['number']})")

        for i, a in enumerate(assets, 1):
            upload_attachment(client, base, args.page_id, args.token, a)
            print(f"  [{i}/{len(assets)}] uploaded {a.name}")

        update_body(client, base, args.page_id, args.token, page, storage_xml, args.title)
        print(f"Page body updated -> version {int(page['version']['number']) + 1}. Done.")


if __name__ == "__main__":
    main()
