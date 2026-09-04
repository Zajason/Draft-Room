"""Credit pricing.

Real prices come from the EuroLeague Fantasy app and should always be preferred - pass
them in the availability spreadsheet and they win.  When they are missing (pre-release,
or a player the sheet does not price) we synthesise one, because the optimiser is a
budget problem and cannot rank anything without a cost on each player.

The synthetic curve is anchored to the game's published 4.0-16.0 band and is deliberately
slightly concave: fantasy games price stars at a premium relative to their raw output,
which is exactly the tension that makes the budget bind.
"""
from __future__ import annotations

from typing import List, Optional

from .config import RULES


def _round_half(x: float) -> float:
    return round(x * 2.0) / 2.0


def estimate_prices(players: List[dict], overwrite: bool = False) -> List[dict]:
    fps = sorted([p.get("fp") or 0.0 for p in players], reverse=True)
    if not fps:
        return players
    # Anchor on the 3rd-best projection rather than the very best, so one outlier does
    # not compress everybody else against the floor.
    ref = fps[min(2, len(fps) - 1)] or 1.0
    lo, hi = RULES.price_min, RULES.price_max

    for p in players:
        if p.get("price") is not None and not overwrite:
            p["price_source"] = p.get("price_source") or "sheet"
            continue
        fp = max(0.0, p.get("fp") or 0.0)
        raw = lo + (hi - lo) * (fp / ref) ** 0.90
        p["price"] = float(min(hi, max(lo, _round_half(raw))))
        p["price_source"] = "model"
    return players


# A head coach's fantasy return tracks how his team performs rather than any box score
# of his own, and it lands in the same range as a solid rotation player.  These bounds
# put the best bench in the league at COACH_FP_MAX and the weakest at COACH_FP_MIN;
# override them once the app publishes real coach scores.
COACH_FP_MIN = 6.0
COACH_FP_MAX = 17.0


def price_coaches(coaches: List[dict], team_scores: dict) -> List[dict]:
    """Score and price coaches from the strength of the side they will actually field."""
    if not coaches:
        return coaches
    vals = [team_scores.get(c.get("club"), 0.0) for c in coaches]
    lo_v, hi_v = min(vals), max(vals)
    span = (hi_v - lo_v) or 1.0
    for c, v in zip(coaches, vals):
        c["position"] = "H"
        q = (v - lo_v) / span                      # 0 = weakest squad, 1 = strongest
        c["team_strength"] = round(q, 3)
        c["fp"] = round(COACH_FP_MIN + (COACH_FP_MAX - COACH_FP_MIN) * q ** 0.85, 3)
        # Coaching returns swing on results, which are noisier than a player's minutes.
        c["fp_sigma"] = round(max(1.2, 0.30 * c["fp"]), 3)
        if c.get("price") is None:
            c["price"] = float(_round_half(4.0 + 8.0 * q ** 0.9))
            c["price_source"] = "model"
        c["fp_per_credit"] = round(c["fp"] / c["price"], 3) if c.get("price") else None
    return coaches
