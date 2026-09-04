"""Network layer: pulls rosters, multi-season stats and box scores, with a disk cache.

Every remote call goes through `_get_json`, which memoises the parsed response under
data/raw/.  Re-running the pipeline is therefore free until you pass --refresh.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from .config import APIV2, APIV3, ESPN, FEEDS, RAW, USER_AGENT

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

REFRESH = False          # flipped by the CLI --refresh flag
_VERBOSE = True


class _Throttle:
    """Serialises request *starts* with a minimum gap so we stay under the rate limit."""

    def __init__(self, min_gap: float = 0.55):
        self._lock = threading.Lock()
        self._gap = min_gap
        self._next = 0.0

    def __enter__(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
            self._next = max(now, self._next) + self._gap

    def __exit__(self, *exc):
        return False


_THROTTLE = _Throttle()


def log(msg: str) -> None:
    if _VERBOSE:
        print(msg, file=sys.stderr, flush=True)


def _cache_path(url: str, tag: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:12]
    return os.path.join(RAW, "{}__{}.json".format(tag, h))


def _get_json(url: str, tag: str, retries: int = 8, timeout: int = 40) -> Optional[Any]:
    path = _cache_path(url, tag)
    if not REFRESH and os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (ValueError, OSError):
            pass  # corrupt cache entry -> refetch

    last = None
    for attempt in range(retries):
        try:
            with _THROTTLE:
                r = _SESSION.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 503):
                # Upstream rate limit: honour Retry-After, otherwise back off hard.
                wait = float(r.headers.get("Retry-After") or 0) or min(90.0, 5.0 * (2 ** attempt))
                time.sleep(wait + random.random())
                last = "HTTP {}".format(r.status_code)
                continue
            r.raise_for_status()
            data = r.json()
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
            return data
        except Exception as exc:            # noqa: BLE001 - network layer, keep going
            last = exc
            time.sleep(0.8 * (attempt + 1) + random.random() * 0.4)
    log("  ! failed {} ({})".format(url[:110], last))
    return None


def _parallel(urls_tags, workers: int = 10):
    """Fetch (url, tag, key) triples concurrently, returning {key: payload}."""
    out: Dict[Any, Any] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_get_json, u, t): k for (u, t, k) in urls_tags}
        for fut in cf.as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


# ======================================================================================
# EuroLeague / EuroCup
# ======================================================================================
def clubs(season_code: str) -> List[dict]:
    comp = season_code[0]
    url = "{}/v2/competitions/{}/seasons/{}/clubs".format(FEEDS, comp, season_code)
    d = _get_json(url, "clubs_{}".format(season_code))
    return (d or {}).get("data", []) or []


def club_roster(season_code: str, club_code: str) -> List[dict]:
    comp = season_code[0]
    url = "{}/v2/competitions/{}/seasons/{}/clubs/{}/people".format(
        FEEDS, comp, season_code, club_code
    )
    d = _get_json(url, "roster_{}_{}".format(season_code, club_code))
    return d if isinstance(d, list) else (d or {}).get("data", []) or []


def all_rosters(season_code: str) -> List[dict]:
    """Every person attached to a club this season, flattened and lightly normalised."""
    cs = clubs(season_code)
    log("· rosters {}: {} clubs".format(season_code, len(cs)))
    comp = season_code[0]
    reqs = [
        (
            "{}/v2/competitions/{}/seasons/{}/clubs/{}/people".format(
                FEEDS, comp, season_code, c["code"]
            ),
            "roster_{}_{}".format(season_code, c["code"]),
            c["code"],
        )
        for c in cs
    ]
    res = _parallel(reqs)
    rows: List[dict] = []
    for club_code, payload in res.items():
        items = payload if isinstance(payload, list) else (payload or {}).get("data", [])
        for it in items or []:
            it = dict(it)
            it["_club"] = club_code
            rows.append(it)
    return rows


def season_player_stats(
    season_code: str, kind: str = "traditional", mode: str = "accumulated",
    phase: Optional[str] = None,
) -> List[dict]:
    """Aggregated per-season player statistics from the v3 statistics endpoint.

    `accumulated` returns every player who logged a minute; `perGame` silently applies a
    qualifying filter, so we always pull accumulated totals and derive rates ourselves.
    """
    comp = season_code[0]
    url = (
        "{}/competitions/{}/statistics/players/{}"
        "?SeasonMode=Single&SeasonCode={}&statisticMode={}&limit=1000"
    ).format(APIV3, comp, kind, season_code, mode)
    if phase:
        url += "&phaseTypeCode={}".format(phase)
    d = _get_json(url, "stats_{}_{}_{}".format(season_code, kind, mode))
    return (d or {}).get("players", []) or []


def season_games(season_code: str) -> List[dict]:
    comp = season_code[0]
    url = "{}/competitions/{}/seasons/{}/games?limit=1000".format(APIV2, comp, season_code)
    d = _get_json(url, "games_{}".format(season_code))
    return (d or {}).get("data", []) or []


def season_boxscores(season_code: str, workers: int = 3,
                     cache_only: bool = False) -> Dict[int, dict]:
    """Per-game box scores for a whole season -> {gameCode: payload}.

    `cache_only` reads whatever has already been downloaded without touching the
    network, so the modelling pipeline never blocks on a slow or rate-limited feed.
    """
    games = season_games(season_code)
    played = [g for g in games if g.get("played") or g.get("gameCode")]
    comp = season_code[0]
    reqs = [
        (
            "{}/competitions/{}/seasons/{}/games/{}/stats".format(
                APIV2, comp, season_code, g["gameCode"]
            ),
            "box_{}_{}".format(season_code, g["gameCode"]),
            g["gameCode"],
        )
        for g in played
    ]
    if cache_only:
        out: Dict[int, dict] = {}
        for url, tag, key in reqs:
            path = _cache_path(url, tag)
            if os.path.exists(path):
                try:
                    with open(path) as fh:
                        out[key] = json.load(fh)
                except (ValueError, OSError):
                    continue
        log("· box scores {} (cache): {}/{} games".format(season_code, len(out), len(reqs)))
        return out
    log("· box scores {}: {} games".format(season_code, len(reqs)))
    res = _parallel(reqs, workers=workers)
    return {k: v for k, v in res.items() if v}


# ======================================================================================
# NBA (ESPN)
# ======================================================================================
def nba_season_stats(season: int, page_size: int = 100) -> List[dict]:
    """All NBA players' regular-season averages for the season ENDING in `season`."""
    rows: List[dict] = []
    page = 1
    while True:
        url = (
            "{}/statistics/byathlete?region=us&lang=en&contentorigin=espn"
            "&season={}&seasontype=2&limit={}&page={}"
        ).format(ESPN, season, page_size, page)
        d = _get_json(url, "nba_{}_{}".format(season, page))
        if not d:
            break
        names = {c["name"]: c.get("names", []) for c in d.get("categories", [])}
        athletes = d.get("athletes", []) or []
        for a in athletes:
            ath = a.get("athlete") or {}
            flat: Dict[str, float] = {}
            for cat in a.get("categories", []) or []:
                keys = names.get(cat.get("name"), [])
                for k, v in zip(keys, cat.get("values", []) or []):
                    flat[k] = v
            rows.append(
                {
                    "espn_id": ath.get("id"),
                    "name": ath.get("displayName"),
                    "first": ath.get("firstName"),
                    "last": ath.get("lastName"),
                    "position": ((ath.get("position") or {}).get("abbreviation")),
                    "team": ((ath.get("team") or {}).get("abbreviation")),
                    "season": season,
                    "stats": flat,
                }
            )
        pag = d.get("pagination") or {}
        if page >= int(pag.get("pages") or 1) or not athletes:
            break
        page += 1
    log("· nba {}: {} players".format(season, len(rows)))
    return rows
