"""End-to-end: raw feeds -> projections -> ratings -> prices -> a draft board."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from . import build, pricing, ratings
from .config import DATA, RULES, norm_position
from .excel_io import link_to_universe, read_availability
from .project import project_all

BOARD_PATH = os.path.join(DATA, "board.json")


def build_board(rebuild: bool = False) -> dict:
    """The full player board: everyone on a 2026-27 roster, projected, rated and priced."""
    if rebuild or not os.path.exists(build.UNIVERSE_PATH):
        uni = build.build_universe()
    else:
        uni = build.load_universe()

    players = uni["players"]
    project_all(players)
    ratings.attach_ratings(players)
    pricing.estimate_prices(players)
    ratings.attach_value_ratings(players)

    team_scores: Dict[str, float] = {}
    for p in players:
        team_scores[p.get("club")] = team_scores.get(p.get("club"), 0.0) + (p.get("fp") or 0.0)
    coaches = pricing.price_coaches(uni.get("coaches", []), team_scores)

    board = {
        "meta": dict(uni["meta"], rules={
            "budget": RULES.budget, "slots": RULES.slots,
            "full_credit_slots": RULES.full_credit_slots,
            "bench_multiplier": RULES.bench_multiplier,
            "captain_multiplier": RULES.captain_multiplier,
        }),
        "clubs": uni["clubs"],
        "players": players,
        "coaches": coaches,
        "team_scores": {k: round(v, 2) for k, v in team_scores.items()},
    }
    with open(BOARD_PATH, "w") as fh:
        json.dump(board, fh)
    return board


def load_board() -> dict:
    with open(BOARD_PATH) as fh:
        return json.load(fh)


# ======================================================================================
# Applying an availability sheet
# ======================================================================================
def apply_availability(board: dict, sheet_path: Optional[str]) -> dict:
    """Overlay the user's spreadsheet: who is available, real prices, who is already mine.

    Without a sheet everybody on a 2026-27 roster is treated as available, which is what
    makes the tool useful before the app opens.
    """
    players = board["players"]
    coaches = board.get("coaches", [])
    for p in players + coaches:
        p["available"] = True
        p["mine"] = False
    report = {"sheet": sheet_path, "matched": 0, "unmatched": [], "priced": 0,
              "available": len(players), "mine": 0}
    if not sheet_path:
        return report

    rows = read_availability(sheet_path)
    link = link_to_universe(rows, players, coaches)
    # Anyone the sheet knows about but does not list as available is off the board.
    listed = set()
    for row, p in link["matched"]:
        listed.add(p["code"])
        if row.get("price") is not None:
            p["price"] = float(row["price"])
            p["price_source"] = "sheet"
            report["priced"] += 1
        # A coach's slot is fixed; never let a stray sheet value move him onto the court.
        if row.get("position") and p.get("role") != "coach" and p.get("position") != "H":
            p["position"] = norm_position(row["position"])
        p["available"] = True if row.get("available") is None else bool(row["available"])
        p["mine"] = bool(row.get("mine"))
    # A sheet of "available players" is a whitelist: if you are not on it, you are gone.
    for p in players + coaches:
        if p["code"] not in listed:
            p["available"] = False

    ratings.attach_value_ratings(players)
    report["matched"] = len(link["matched"])
    report["unmatched"] = [r["sheet_name"] for r in link["unmatched"]]
    report["available"] = sum(1 for p in players if p["available"])
    report["mine"] = sum(1 for p in players if p["mine"])
    return report


def apply_roster(board: dict, roster_path: Optional[str]) -> List[dict]:
    """Mark players you have already drafted (JSON list of names, or a sheet column)."""
    if not roster_path:
        return [p for p in board["players"] if p.get("mine")]
    from .names import NameIndex, split_el_name
    idx = NameIndex()
    for p in board["players"]:
        idx.add(p.get("first") or "", p.get("last") or "", p)
    for c in board.get("coaches", []):
        idx.add(c.get("first") or "", c.get("last") or "", c)

    with open(roster_path) as fh:
        if roster_path.lower().endswith(".json"):
            data = json.load(fh)
            names = data if isinstance(data, list) else data.get("players", [])
        else:
            names = [ln.strip() for ln in fh if ln.strip()]

    got = []
    for entry in names:
        name = entry if isinstance(entry, str) else entry.get("name")
        price = None if isinstance(entry, str) else entry.get("price")
        first, last = split_el_name(name)
        hit = idx.lookup(first, last, max_tier=2)
        if hit is None:
            raise ValueError("roster: could not match '{}' to any 2026-27 roster".format(name))
        hit["mine"] = True
        if price is not None:
            hit["price"] = float(price)
            hit["price_source"] = "roster"
        got.append(hit)
    return got


def draft_pool(board: dict, include_coach: bool = True) -> Tuple[List[dict], List[dict], float]:
    """Split the board into (candidate pool, already-mine, remaining budget)."""
    mine = [p for p in board["players"] if p.get("mine")]
    coaches = board.get("coaches", [])
    mine += [c for c in coaches if c.get("mine")]

    pool = [p for p in board["players"] if p.get("available") and not p.get("mine")]
    if include_coach:
        pool += [c for c in coaches if c.get("available", True) and not c.get("mine")]
    for p in mine:
        p["mandatory"] = True
    return pool + mine, mine, RULES.budget
