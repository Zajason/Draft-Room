"""Builds the scouting dashboard: a single self-contained HTML file.

Everything the page needs is embedded, so the file works offline and can be mailed
around.  The Python side is only responsible for shaping a compact payload - all the
drawing happens in the page.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .config import OUT, RULES, TARGET_SEASON
from .images import build_images
from .optimize import Slots, robust_draft

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_template.html")

# Attributes shown on the scouting radar, in the order they are drawn.
RADAR = [
    ("scoring", "Scoring"),
    ("three_point", "Three-point"),
    ("finishing", "Finishing"),
    ("playmaking", "Playmaking"),
    ("rebounding", "Rebounding"),
    ("defense", "Defence"),
    ("efficiency", "Efficiency"),
    ("role", "Role"),
]

BARS = [
    ("usage", "Usage"),
    ("ball_security", "Ball security"),
    ("motor", "Motor"),
    ("free_throws", "Free throws"),
    ("consistency", "Consistency"),
    ("durability", "Durability"),
    ("form", "Late-season form"),
    ("upside", "Ceiling"),
    ("floor", "Floor"),
]


def _round(x, nd=2):
    return None if x is None else round(float(x), nd)


def _season_rows(p: dict) -> List[dict]:
    """A compact per-season history for the scouting card."""
    rows = []
    for comp, key in (("EuroLeague", "el"), ("EuroCup", "ec")):
        for season, b in sorted((p.get(key) or {}).items(), reverse=True):
            rows.append({
                "comp": comp,
                "season": "{}-{}".format(season[1:], str(int(season[1:]) + 1)[-2:]),
                "team": b.get("team"),
                "gp": _round(b.get("gp"), 0),
                "mpg": _round(b.get("mpg"), 1),
                "pir": _round(b.get("pir_pg"), 1),
                "pts": _round(b.get("pts_pg"), 1),
                "reb": _round(b.get("reb_pg"), 1),
                "ast": _round(b.get("ast_pg"), 1),
                "ts": _round(b.get("ts"), 1),
                "p40": _round(b.get("pir_p40"), 1),
            })
    for season, b in sorted((p.get("nba") or {}).items(), reverse=True):
        rows.append({
            "comp": "NBA",
            "season": "{}-{}".format(int(season) - 1, str(season)[-2:]),
            "team": b.get("team"),
            "gp": _round(b.get("gp"), 0),
            "mpg": _round(b.get("mpg"), 1),
            "pir": _round(b.get("pir_pg"), 1),
            "pts": _round(b.get("pts"), 1),
            "reb": _round(b.get("reb"), 1),
            "ast": _round(b.get("ast"), 1),
            "ts": None,
            "p40": _round((b.get("pir_pg") or 0) / b["mpg"] * 40.0, 1) if b.get("mpg") else None,
        })
    return rows


def _player_payload(p: dict, draft: Dict[str, dict]) -> dict:
    prof = p.get("profile") or {}
    d = draft.get(p["code"], {})
    series = [
        {"r": g.get("r"), "pir": _round(g.get("pir"), 1), "m": _round(g.get("min"), 1),
         "s": 1 if g.get("start") else 0}
        for g in (prof.get("series") or [])
    ]
    return {
        "code": p["code"],
        "name": p["name"],
        "pos": p["position"],
        "club": p.get("club"),
        "clubName": p.get("club_name"),
        "age": p.get("age"),
        "height": p.get("height"),
        "country": p.get("country"),
        "countryCode": p.get("country_code"),
        "dorsal": p.get("dorsal"),
        "price": p.get("price"),
        "priceSource": p.get("price_source"),
        "fp": _round(p.get("fp")),
        "sigma": _round(p.get("fp_sigma")),
        "low": _round(p.get("fp_floor")),
        "high": _round(p.get("fp_ceiling")),
        "mptg": _round(p.get("mpg_proj"), 1),
        "mpgPlaying": _round(p.get("mpg_when_playing"), 1),
        "mpgPrior": _round(p.get("mpg_prior"), 1),
        "squeeze": _round(p.get("role_squeeze"), 2),
        "rank": p.get("rotation_rank"),
        "rate": _round(p.get("rate_p40"), 1),
        "avail": _round(p.get("availability"), 2),
        "teamCtx": _round(p.get("team_context"), 3),
        "sample": _round(p.get("sample_minutes"), 0),
        "leagues": p.get("sample_leagues") or [],
        "newTeam": bool(p.get("new_to_team")),
        "prevTeam": p.get("prev_team"),
        "unknown": bool(p.get("unknown")),
        "ratings": p.get("ratings") or {},
        "ppc": _round(p.get("fp_per_credit"), 3),
        "prof": {
            "gp": prof.get("games_played"),
            "listed": prof.get("games_listed"),
            "mean": prof.get("pir_mean"),
            "sd": prof.get("pir_sd"),
            "floor": prof.get("floor"),
            "median": prof.get("median"),
            "ceil": prof.get("ceiling"),
            "cons": prof.get("consistency"),
            "form": prof.get("form_delta"),
            "start": prof.get("start_rate"),
            "boom": prof.get("boom_rate"),
            "bust": prof.get("bust_rate"),
        },
        "series": series,
        "history": _season_rows(p),
        "available": bool(p.get("available", True)),
        "mine": bool(p.get("mine")),
        "draftValue": d.get("draft_value"),
        "pickRate": d.get("pick_rate"),
        "replaces": d.get("replaces"),
        "inSquad": bool(d.get("in_optimal")),
    }


def build_payload(board: dict, sims: int = 400) -> dict:
    players = board["players"]
    coaches = board.get("coaches", [])

    imgs = build_images(board)
    from .pipeline import draft_pool
    pool, mine, budget = draft_pool(board)
    result = robust_draft(pool, budget, Slots(), n_sims=sims, top_k=0)
    draft = {r["code"]: r for r in result["all_rows"]}

    payload_players = [_player_payload(p, draft) for p in players]
    for pl in payload_players:
        pl["photo"] = imgs["players"].get(pl["code"])
    payload_coaches = []
    for c in coaches:
        d = draft.get(c["code"], {})
        payload_coaches.append({
            "code": c["code"], "name": c["name"], "pos": "H",
            "club": c.get("club"), "clubName": c.get("club_name"),
            "price": c.get("price"), "fp": _round(c.get("fp")),
            "sigma": _round(c.get("fp_sigma")),
            "strength": c.get("team_strength"),
            "draftValue": d.get("draft_value"), "pickRate": d.get("pick_rate"),
            "inSquad": bool(d.get("in_optimal")), "mine": bool(c.get("mine")),
            "available": True, "photo": imgs["players"].get(c["code"]),
        })

    clubs = []
    for code, c in (board.get("clubs") or {}).items():
        squad = [p for p in players if p.get("club") == code]
        clubs.append({
            "code": code,
            "name": c.get("abbreviatedName") or c.get("name"),
            "fullName": c.get("name"),
            "country": (c.get("country") or {}).get("name"),
            "strength": round(sum(p.get("fp") or 0 for p in players if p.get("club") == code), 1),
            "size": len(squad), "crest": imgs["crests"].get(code),
        })
    clubs.sort(key=lambda x: -x["strength"])

    validation = None
    vpath = os.path.join(OUT, "validation.json")
    if os.path.exists(vpath):
        with open(vpath) as fh:
            validation = json.load(fh)
    season_test = None
    spath = os.path.join(OUT, "season_backtest.json")
    if os.path.exists(spath):
        with open(spath) as fh:
            st = json.load(fh)
        # compress the field to a min/median/max band per round for the chart
        fs = st.get("field_series") or []
        band = None
        if fs:
            arr = list(zip(*fs))
            band = {"lo": [round(min(x), 1) for x in arr],
                    "hi": [round(max(x), 1) for x in arr]}
        season_test = {k: v for k, v in st.items() if k != "field_series"}
        season_test["field_band"] = band
    strategy_test = None
    stpath = os.path.join(OUT, "strategy_backtest.json")
    if os.path.exists(stpath):
        with open(stpath) as fh:
            strategy_test = json.load(fh)
    bakeoff = None
    bpath = os.path.join(OUT, "bakeoff.json")
    if os.path.exists(bpath):
        with open(bpath) as fh:
            bakeoff = json.load(fh)

    return {
        "meta": dict(board["meta"], generatedFor=TARGET_SEASON),
        "rules": {
            "budget": RULES.budget, "slots": RULES.slots,
            "fullCredit": RULES.full_credit_slots,
            "benchMultiplier": RULES.bench_multiplier,
            "captainMultiplier": RULES.captain_multiplier,
            "priceMin": RULES.price_min, "priceMax": RULES.price_max,
        },
        "players": payload_players,
        "coaches": payload_coaches,
        "clubs": clubs,
        "crests": imgs["crests"],
        "squad": [p["code"] for p in result["squad"]],
        "squadCost": result["squad_cost"],
        "squadExpected": result["squad_expected_value"],
        "dpPoint": result["dp_point_value"],
        "dpExpected": result["dp_expected_value"],
        "polishSwaps": result["polish_swaps"],
        "nSims": result["n_sims"],
        "radar": RADAR,
        "bars": BARS,
        "validation": validation,
        "seasonTest": season_test,
        "strategyTest": strategy_test,
        "bakeoff": bakeoff,
    }


def build_dashboard(board: dict, out_path: Optional[str] = None, sims: int = 400) -> str:
    out_path = out_path or os.path.join(OUT, "dashboard.html")
    payload = build_payload(board, sims=sims)
    with open(TEMPLATE) as fh:
        html = fh.read()
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html = html.replace("/*__DATA__*/null", blob)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path
