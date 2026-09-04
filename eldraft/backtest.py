"""Out-of-sample validation.

Rebuilds the world as it looked in the summer of a past season - rosters as they were,
history truncated to seasons that had actually finished - projects that season, and
scores the projection against what really happened.  The model never sees a minute of
the season it is being graded on.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

from . import build
from .config import DATA
from .project import Horizon, project_all


def _prev_seasons(code: str, n: int) -> List[str]:
    comp, year = code[0], int(code[1:])
    return ["{}{}".format(comp, year - i) for i in range(1, n + 1)]


_UNI_CACHE: Dict[str, dict] = {}


def universe_for(target: str, history: int = 3) -> Tuple[dict, Horizon]:
    """The world as it looked before `target` tipped off (memoised across sweeps)."""
    key = "{}:{}".format(target, history)
    el_hist = _prev_seasons(target, history)
    ec_hist = _prev_seasons("U" + target[1:], history)
    nba_end = int(target[1:])           # E2025 -> NBA seasons ending 2025 and earlier
    nba_hist = [nba_end - i for i in range(history)]
    hz = Horizon(el=el_hist, ec=ec_hist, nba=nba_hist, logs=[])
    if key not in _UNI_CACHE:
        _UNI_CACHE[key] = build.build_universe(
            save=True, target_season=target,
            el_history=el_hist, ec_history=ec_hist, nba_history=nba_hist,
            gamelog_seasons=[],         # the only logs we hold are the target season's
            path=os.path.join(DATA, "universe_backtest_{}.json".format(target)),
        )
    return _UNI_CACHE[key], hz


def run(target: str = "E2025", history: int = 3, min_games: int = 10) -> dict:
    """Project `target` using only prior seasons, then compare with the real outcome."""
    uni, hz = universe_for(target, history)
    el_hist = hz.el
    import copy
    players = copy.deepcopy(uni["players"])
    project_all(players, hz)

    # Ground truth: what these same players actually did in the target season.
    actual = build._collect_season(target)
    rows = []
    for p in players:
        a = actual.get(p.get("stat_code") or p["code"])
        if not a or a.get("gp", 0) < min_games:
            continue
        prev = (p.get("el") or {}).get(el_hist[0]) or {}
        rows.append({
            "name": p["name"], "position": p["position"], "club": p.get("club"),
            "proj_fp": p["fp"], "proj_mpg": p["mpg_proj"], "proj_sigma": p["fp_sigma"],
            # Compare like with like: the model projects per TEAM game, so grade it
            # against per-team-game reality, not per-appearance averages.
            "act_fp": a.get("pir_per_team_game", a["pir_pg"]),
            "act_mpg": a.get("min_per_team_game", a["mpg"]),
            "act_fp_per_app": a["pir_pg"], "gp": a["gp"],
            # Naive baselines, for context on whether the model earns its complexity.
            "base_last_pir": prev.get("pir_per_team_game", prev.get("pir_pg", 0.0)),
            "base_rate_x_mpg": prev.get("pir_p40", 0.0) * prev.get("min_per_team_game", 0.0) / 40.0,
        })
    metrics = score(rows)
    metrics["minutes_curve_rmse"] = round(curve_error(players), 3)
    return {"target": target, "n": len(rows), "rows": rows,
            "metrics": metrics, "baselines": baselines(rows), "history": el_hist}


def baselines(rows: List[dict]) -> Dict[str, dict]:
    """Score the same rows with dumb predictors, so the model has something to beat."""
    af = [r["act_fp"] for r in rows]
    out = {}
    for key, label in (("base_last_pir", "last season PIR/game"),
                       ("base_rate_x_mpg", "last season rate x minutes"),
                       ("proj_fp", "model")):
        pf = [r[key] for r in rows]
        n = len(rows)
        k = max(1, n // 10)
        tp = {r["name"] for r in sorted(rows, key=lambda r: -r[key])[:k]}
        ta = {r["name"] for r in sorted(rows, key=lambda r: -r["act_fp"])[:k]}
        out[label] = {
            "pearson": round(_pearson(pf, af), 4),
            "spearman": round(_spearman(pf, af), 4),
            "mae": round(sum(abs(a - b) for a, b in zip(pf, af)) / n, 3),
            "top_decile_hit_rate": round(len(tp & ta) / k, 4),
        }
    return out


# --------------------------------------------------------------------------------------
def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def _rank(v: List[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def _spearman(x: List[float], y: List[float]) -> float:
    return _pearson(_rank(x), _rank(y))


# Observed minutes-per-team-game by season-long rotation rank (2024-25 + 2025-26 mean).
# A projection can rank players correctly and still put the wrong *shape* on a club's
# rotation, which would quietly mislead anyone reading the depth chart, so the shape is
# scored explicitly rather than left to the rank metrics to notice.
OBSERVED_CURVE = [26.45, 23.52, 21.72, 19.68, 18.39, 16.34, 14.64, 12.91,
                  11.61, 9.62, 8.08, 6.03, 3.84, 2.47, 1.37, 0.72]


def curve_error(players: List[dict]) -> float:
    """RMSE between the projected per-club minutes curve and the observed one."""
    import collections
    by_club = collections.defaultdict(list)
    for p in players:
        by_club[p.get("club")].append(p.get("mpg_proj", 0.0))
    curves = [sorted(v, reverse=True) for v in by_club.values() if v]
    if not curves:
        return float("nan")
    err, n = 0.0, 0
    for r, target in enumerate(OBSERVED_CURVE):
        vals = [c[r] for c in curves if len(c) > r]
        if not vals:
            continue
        err += (sum(vals) / len(vals) - target) ** 2
        n += 1
    return math.sqrt(err / n) if n else float("nan")


def score(rows: List[dict]) -> dict:
    if not rows:
        return {}
    pf = [r["proj_fp"] for r in rows]
    af = [r["act_fp"] for r in rows]
    pm = [r["proj_mpg"] for r in rows]
    am = [r["act_mpg"] for r in rows]
    n = len(rows)

    # Calibration of the uncertainty band: how often does the outcome land inside +-1
    # sigma?  A well-calibrated normal-ish interval should cover about 68%.
    inside = sum(
        1 for r in rows
        if abs(r["act_fp"] - r["proj_fp"]) <= max(1e-9, r["proj_sigma"])
    ) / n

    # Top-decile hit rate: of the players the model ranked in its top 10%, how many
    # finished in the real top 10%?  This is the number that matters for a draft.
    k = max(1, n // 10)
    top_proj = {r["name"] for r in sorted(rows, key=lambda r: -r["proj_fp"])[:k]}
    top_act = {r["name"] for r in sorted(rows, key=lambda r: -r["act_fp"])[:k]}

    return {
        "n": n,
        "fp_pearson": round(_pearson(pf, af), 4),
        "fp_spearman": round(_spearman(pf, af), 4),
        "fp_mae": round(sum(abs(a - b) for a, b in zip(pf, af)) / n, 3),
        "fp_bias": round(sum(a - b for a, b in zip(pf, af)) / n, 3),
        "mpg_pearson": round(_pearson(pm, am), 4),
        "mpg_mae": round(sum(abs(a - b) for a, b in zip(pm, am)) / n, 3),
        "mpg_bias": round(sum(a - b for a, b in zip(pm, am)) / n, 3),
        "sigma_coverage_1sd": round(inside, 4),
        "top_decile_hit_rate": round(len(top_proj & top_act) / k, 4),
    }
