"""Football-Manager / NBA2K style 0-99 attribute ratings.

These exist so a human can audit the model's picks quickly.  Every rating is a
percentile of a real underlying statistic, so a 92 Playmaking always means "92nd
percentile among EuroLeague players at this position", never a hand-tuned opinion.
Attributes are ranked *within position* (a centre is not judged against guards for
assists); overall and value ratings are ranked across the whole league.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .config import EL_HISTORY

# label -> (within-position?, extractor)
Extractor = Callable[[dict], Optional[float]]


def _last_el(p: dict) -> dict:
    el = p.get("el") or {}
    for s in EL_HISTORY:
        if s in el and el[s].get("min", 0) > 0:
            return el[s]
    return {}


def _best_block(p: dict) -> dict:
    """Prefer EuroLeague, fall back to EuroCup, then NBA-derived, for shape stats."""
    b = _last_el(p)
    if b.get("min", 0) >= 120:
        return b
    ec = p.get("ec") or {}
    for s in sorted(ec, reverse=True):
        if ec[s].get("min", 0) > 0:
            return ec[s]
    return b


def _per36(b: dict, key: str) -> Optional[float]:
    m = b.get("min", 0.0)
    if not m:
        return None
    return b.get(key, 0.0) / m * 36.0


def _nba_per36(p: dict, key: str) -> Optional[float]:
    nba = p.get("nba") or {}
    if not nba:
        return None
    s = nba[max(nba)]
    if not s.get("mpg"):
        return None
    return s.get(key, 0.0) / s["mpg"] * 36.0


def _stat(p: dict, key: str, nba_key: Optional[str] = None) -> Optional[float]:
    b = _best_block(p)
    v = _per36(b, key) if b else None
    if v is None and nba_key:
        v = _nba_per36(p, nba_key)
    return v


def _defense(p: dict) -> Optional[float]:
    stl = _stat(p, "stl", "stl")
    if stl is None:
        return None
    return stl + 1.4 * (_stat(p, "blk", "blk") or 0.0)


ATTRIBUTES: Dict[str, Dict] = {
    "scoring":      {"pos": True,  "f": lambda p: _stat(p, "pts", "pts")},
    "three_point":  {"pos": True,  "f": lambda p: _stat(p, "fg3m", None)},
    "finishing":    {"pos": True,  "f": lambda p: (_best_block(p).get("fg2m", 0) / _best_block(p)["fg2a"] if _best_block(p).get("fg2a") else None)},
    "free_throws":  {"pos": True,  "f": lambda p: (_best_block(p).get("ftm", 0) / _best_block(p)["fta"] if _best_block(p).get("fta") else None)},
    "playmaking":   {"pos": True,  "f": lambda p: _stat(p, "ast", "ast")},
    "rebounding":   {"pos": True,  "f": lambda p: _stat(p, "reb", "reb")},
    "defense":      {"pos": True,  "f": lambda p: _defense(p)},
    "efficiency":   {"pos": True,  "f": lambda p: (_best_block(p).get("ts") or None)},
    "usage":        {"pos": True,  "f": lambda p: (None if _best_block(p).get("min", 0) <= 0 else (_best_block(p).get("fg2a", 0) + _best_block(p).get("fg3a", 0) + 0.44 * _best_block(p).get("fta", 0) + _best_block(p).get("to", 0)) / _best_block(p)["min"] * 36.0)},
    "ball_security": {"pos": True, "f": lambda p: (None if _stat(p, "to", "to") is None else -float(_stat(p, "to", "to") or 0.0))},
    "motor":        {"pos": True,  "f": lambda p: _stat(p, "fd", None)},
    "role":         {"pos": False, "f": lambda p: p.get("mpg_proj")},
    "durability":   {"pos": False, "f": lambda p: p.get("availability")},
    "consistency":  {"pos": False, "f": lambda p: (p.get("profile") or {}).get("consistency")},
    "form":         {"pos": False, "f": lambda p: (p.get("profile") or {}).get("form_delta")},
    "upside":       {"pos": False, "f": lambda p: p.get("fp_ceiling")},
    "floor":        {"pos": False, "f": lambda p: p.get("fp_floor")},
}


def _percentile_map(values: List[Optional[float]]) -> List[Optional[float]]:
    """Rank -> percentile in [0,1], averaging ties.  None values stay None."""
    idx = [i for i, v in enumerate(values) if v is not None]
    out: List[Optional[float]] = [None] * len(values)
    if not idx:
        return out
    order = sorted(idx, key=lambda i: values[i] or 0.0)  # idx excludes None
    n = len(order)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        pct = ((i + j) / 2.0) / max(1, n - 1)
        for k in range(i, j + 1):
            out[order[k]] = pct
        i = j + 1
    return out


def _to_rating(pct: Optional[float], lo: int = 32, hi: int = 99, gamma: float = 0.92) -> int:
    """Percentile -> a 0-99 style rating with a mild curve so elites separate."""
    if pct is None:
        return lo + 8
    return int(round(lo + (hi - lo) * (pct ** gamma)))


def attach_ratings(players: List[dict]) -> List[dict]:
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(players):
        groups.setdefault(p.get("position", "F"), []).append(i)

    for name, spec in ATTRIBUTES.items():
        if spec["pos"]:
            for _, idxs in groups.items():
                vals = [spec["f"](players[i]) for i in idxs]
                for i, pct in zip(idxs, _percentile_map(vals)):
                    players[i].setdefault("ratings", {})[name] = _to_rating(pct)
        else:
            vals = [spec["f"](p) for p in players]
            for p, pct in zip(players, _percentile_map(vals)):
                p.setdefault("ratings", {})[name] = _to_rating(pct)

    # Overall is the projected fantasy score, ranked league-wide, on a 40-99 scale so it
    # reads like a scouting number rather than a raw percentile.
    ovr_pct = _percentile_map([p.get("fp") for p in players])
    rate_pct = _percentile_map([p.get("rate_p40") for p in players])
    for p, o, r in zip(players, ovr_pct, rate_pct):
        p["ratings"]["overall"] = _to_rating(o, lo=40, hi=99, gamma=0.80)
        p["ratings"]["production_rate"] = _to_rating(r)

        # Potential: young players with a strong per-minute rate but a small role have
        # room to grow into; veterans are capped at roughly their current level.
        age = p.get("age") or 28.0
        head = max(0.0, (28.0 - age)) * 1.7
        rate_edge = max(0.0, (p["ratings"]["production_rate"] - p["ratings"]["overall"]) * 0.45)
        pot = p["ratings"]["overall"] + min(16.0, head + rate_edge)
        p["ratings"]["potential"] = int(round(min(99, max(p["ratings"]["overall"], pot))))

        # Risk is the model's own uncertainty, not a scouting opinion.
        cv = (p.get("fp_sigma") or 0.0) / max(0.5, p.get("fp") or 0.5)
        p["ratings"]["risk"] = int(round(max(1, min(99, cv * 130))))
    return players


def attach_value_ratings(players: List[dict]) -> List[dict]:
    """Value ratings depend on price, so they are attached after pricing."""
    ppc = [(p.get("fp") or 0.0) / p["price"] if p.get("price") else None for p in players]
    for p, pct in zip(players, _percentile_map(ppc)):
        p.setdefault("ratings", {})["value"] = _to_rating(pct)
        p["fp_per_credit"] = round((p.get("fp") or 0.0) / p["price"], 3) if p.get("price") else None
    return players
