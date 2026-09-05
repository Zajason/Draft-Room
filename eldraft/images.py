"""Fetches club crests and player headshots and embeds them as data URIs.

Published artifacts block external images (CSP), so every badge and photo has to travel
inside the HTML.  Raw headshots are ~1 MB each, so they are downscaled hard to small webp
thumbnails — a whole squad of them costs less than a stray screenshot.  Results are cached
on disk keyed by URL, so only the first build pays the download.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import io
import json
import os
import ssl
import urllib.request
from typing import Dict, Optional

from . import fetch
from .config import DATA, EC_HISTORY, EL_HISTORY

CACHE_DIR = os.path.join(DATA, "img_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
IMAGES_JSON = os.path.join(DATA, "images.json")

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
_UA = {"User-Agent": "Mozilla/5.0"}


def _download(url: str) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers=_UA)
        return urllib.request.urlopen(req, timeout=25, context=_CTX).read()
    except Exception:
        return None


def _to_webp(raw: bytes, kind: str) -> Optional[str]:
    """Downscale to a small webp thumbnail and return a data URI, or None on failure."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None
    w, h = im.size
    if kind == "photo":
        # portrait headshot: fix the height, keep aspect
        target_h = 150
        nw = max(1, int(round(target_h * w / h)))
        im = im.resize((nw, target_h), Image.LANCZOS)
        q = 80
    else:  # crest: fit inside a square
        s = 96.0 / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        q = 88
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=q, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def _cached_data_uri(url: str, kind: str) -> Optional[str]:
    if not url:
        return None
    key = hashlib.sha1((kind + "|" + url).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, key + ".txt")
    if os.path.exists(path):
        with open(path) as fh:
            v = fh.read().strip()
        return v or None
    raw = _download(url)
    uri = _to_webp(raw, kind) if raw else None
    with open(path, "w") as fh:
        fh.write(uri or "")
    return uri


def _photo_url_map() -> Dict[str, str]:
    """Player code -> a usable headshot URL, aggregated newest-first across seasons."""
    urls: Dict[str, str] = {}
    for season in EL_HISTORY + EC_HISTORY:
        for r in fetch.season_player_stats(season, "traditional", "accumulated"):
            p = r.get("player") or {}
            code, u = p.get("code"), p.get("imageUrl")
            if code and u and code not in urls:
                urls[code] = u
    return urls


def build_images(board: dict, workers: int = 16, refresh: bool = False) -> dict:
    """Return {'players': {code: dataURI}, 'crests': {clubCode: dataURI}} and cache it."""
    if not refresh and os.path.exists(IMAGES_JSON):
        with open(IMAGES_JSON) as fh:
            return json.load(fh)

    photo_urls = _photo_url_map()
    # (kind, key, url) work items
    work = []
    seen_players = set()
    for p in board["players"]:
        url = photo_urls.get(p["code"]) or photo_urls.get(p.get("stat_code")) \
            or (p.get("image") if isinstance(p.get("image"), str) else None)
        if url and p["code"] not in seen_players:
            work.append(("photo", p["code"], url))
            seen_players.add(p["code"])
    crest_work = []
    for code, c in (board.get("clubs") or {}).items():
        crest = (c.get("images") or {}).get("crest")
        if crest:
            crest_work.append(("crest", code, crest))

    players: Dict[str, str] = {}
    crests: Dict[str, str] = {}
    all_work = work + crest_work
    fetch.log("· images: {} photos + {} crests".format(len(work), len(crest_work)))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_cached_data_uri, url, kind): (kind, key) for (kind, key, url) in all_work}
        done = 0
        for fut in cf.as_completed(futs):
            kind, key = futs[fut]
            uri = fut.result()
            if uri:
                (players if kind == "photo" else crests)[key] = uri
            done += 1
            if done % 60 == 0:
                fetch.log("  .. {}/{}".format(done, len(all_work)))

    out = {"players": players, "crests": crests}
    with open(IMAGES_JSON, "w") as fh:
        json.dump(out, fh)
    fetch.log("· images: {} photos, {} crests embedded".format(len(players), len(crests)))
    return out
