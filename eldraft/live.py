"""Payload + assembly for the live draft room (out/live_draft.html).

The page is self-contained: the projections and prices are embedded, and both decision
engines — the exact squad DP and the MCTS lookahead — run in the browser, so it works
offline and needs no server on draft night.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from . import weekly as _weekly
from .config import OUT, RULES, TARGET_SEASON
from .images import build_images

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_template.html")


def _assemble_template() -> str:
    """Rebuild the template from its modular sources in web/ when they are present, so the
    HTML shell, the browser engines and the app can be edited as separate files."""
    shell = os.path.join(WEB, "live_shell.html")
    engine = os.path.join(WEB, "engine.js")
    app = os.path.join(WEB, "app_live.js")
    if all(os.path.exists(x) for x in (shell, engine, app)):
        with open(shell) as fh: s = fh.read()
        with open(engine) as fh: e = fh.read()
        with open(app) as fh: a = fh.read()
        html = s + "\n<script>\n" + e + "\n</script>\n<script>\n" + a + "\n</script>\n"
        with open(TEMPLATE, "w") as fh: fh.write(html)
        return html
    with open(TEMPLATE) as fh:
        return fh.read()


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def _player(p: dict) -> dict:
    r = p.get("ratings") or {}
    return {
        "code": p["code"], "name": p["name"], "pos": p.get("position", "F"),
        "club": p.get("club_name") or p.get("club"), "clubCode": p.get("club"),
        "age": _r(p.get("age"), 1), "price": _r(p.get("price"), 1),
        "priceSource": p.get("price_source"),
        "fp": _r(p.get("fp")), "sigma": _r(p.get("fp_sigma")),
        "fpFit": _r(p.get("fp_raw")), "dur": _r(p.get("durability"), 3),
        "mptg": _r(p.get("mpg_proj"), 1), "rate": _r(p.get("rate_p40"), 1),
        "ovr": r.get("overall"), "pot": r.get("potential"), "val": r.get("value"),
        "leagues": p.get("sample_leagues") or [], "unknown": bool(p.get("unknown")),
        "newTeam": bool(p.get("new_to_team")),
    }


def build_payload(board: dict) -> dict:
    imgs = build_images(board)
    players = [_player(p) for p in board["players"]]
    for pl in players:
        pl["photo"] = imgs["players"].get(pl["code"])
    coaches = []
    for c in board.get("coaches", []):
        coaches.append({
            "code": c["code"], "name": c["name"], "pos": "H",
            "club": c.get("club_name") or c.get("club"), "clubCode": c.get("club"),
            "price": _r(c.get("price"), 1), "priceSource": c.get("price_source"),
            "fp": _r(c.get("fp")), "sigma": _r(c.get("fp_sigma")),
            "ovr": None, "val": None, "leagues": [], "unknown": False,
            "strength": c.get("team_strength"), "photo": imgs["players"].get(c["code"]),
        })
    clubs = sorted({p["club"] for p in players if p["club"]})
    club_names = {}
    for p in board["players"]:
        if p.get("club"):
            club_names[p["club"]] = p.get("club_name") or p["club"]
    for c in board.get("coaches", []):
        if c.get("club"):
            club_names.setdefault(c["club"], c.get("club_name") or c["club"])
    try:
        wk = _weekly.build_weekly()
    except Exception:
        wk = None
    return {
        "meta": {"season": TARGET_SEASON, "generated": board["meta"].get("generated")},
        "rules": {
            "budget": RULES.budget, "slots": RULES.slots,
            "full": RULES.full_credit_slots, "bench": RULES.bench_multiplier,
            "captain": RULES.captain_multiplier,
            "priceMin": RULES.price_min, "priceMax": RULES.price_max,
        },
        "players": players + coaches,
        "clubs": clubs,
        "clubNames": club_names,
        "crests": imgs["crests"],
        "weekly": wk,
    }


DASHBOARD_URL = "https://claude.ai/code/artifact/184625fa-75eb-4811-9f0c-eb4dd987d908"


def build_live(board: dict, out_path: Optional[str] = None, scout_url: str = DASHBOARD_URL) -> str:
    out_path = out_path or os.path.join(OUT, "live_draft.html")
    html = _assemble_template()
    blob = json.dumps(build_payload(board), separators=(",", ":"), allow_nan=False)
    html = html.replace("/*__DATA__*/null", blob)
    html = html.replace("__SCOUT_URL__", scout_url or "")
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path
