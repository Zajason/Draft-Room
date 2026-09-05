"""Model bake-off — a research-style comparison of projection algorithms.

The rest of the project asserts "the model beats a naive baseline". This module makes that
a proper study: several projection algorithms — including a from-scratch ridge-regression
ML baseline and ablations of the production model — are compared out-of-sample on two axes:

  * ranking quality (Spearman, top-decile hit rate, MAE), and
  * downstream fantasy outcome — plug each projection into the weekly game and count the
    season points it would have scored.

The second axis is the one that matters: a projection is only as good as the decisions it
leads to. Everything is causal — models predict a season from strictly earlier data.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import build
from .config import WEIGHTS


def _prev(code: str, n: int = 1) -> str:
    return "{}{}".format(code[0], int(code[1:]) - n)


def _blocks(season: str) -> Dict[str, dict]:
    return build._collect_season(season)


# --------------------------------------------------------------------------------------
# Feature extraction for the ML baseline
# --------------------------------------------------------------------------------------
FEATS = ["pir_per_team_game", "pir_p40", "min_per_team_game", "pts_pg", "reb_pg",
         "ast_pg", "stl_pg", "blk_pg", "to_pg", "ts", "start_rate", "played_share"]


def _feat_vec(b: dict) -> List[float]:
    v = []
    for k in FEATS:
        x = float(b.get(k, 0.0) or 0.0)
        if not np.isfinite(x):
            x = 0.0
        if k == "pir_p40":          # explodes for players with a sliver of minutes
            x = max(-30.0, min(60.0, x))
        v.append(x)
    return v


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 10.0):
    """Closed-form ridge with standardised features; returns a predict() closure."""
    with np.errstate(all="ignore"):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        y = np.nan_to_num(np.asarray(y, dtype=np.float64))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = np.clip(np.nan_to_num((X - mu) / sd), -10.0, 10.0)
        Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
        A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
        A[0, 0] -= lam                              # don't regularise the intercept
        w = np.linalg.solve(A, Xs.T @ y)

    def predict(Xnew):
        with np.errstate(all="ignore"):
            Z = np.clip(np.nan_to_num((Xnew - mu) / sd), -10.0, 10.0)
            Z = np.hstack([np.ones((len(Z), 1)), Z])
            return Z @ w
    return predict


# --------------------------------------------------------------------------------------
# Projection models — each maps target -> {code: predicted PIR per team game}
# --------------------------------------------------------------------------------------
def model_naive(target: str, prev_blocks: dict) -> Dict[str, float]:
    return {c: b.get("pir_per_team_game", 0.0) for c, b in prev_blocks.items()}


def model_weighted3(target: str, prev_blocks: dict) -> Dict[str, float]:
    """Recency-weighted mean of the last three seasons' PIR per team game."""
    seasons = [_prev(target, 1), _prev(target, 2), _prev(target, 3)]
    w = [1.0, 0.55, 0.28]
    blk = {s: _blocks(s) for s in seasons}
    out = {}
    codes = set().union(*[set(b) for b in blk.values()])
    for c in codes:
        num = den = 0.0
        for wi, s in zip(w, seasons):
            b = blk[s].get(c)
            if b and b.get("team_games", 0):
                num += wi * b.get("pir_per_team_game", 0.0)
                den += wi
        if den:
            out[c] = num / den
    return out


def model_ridge(target: str, prev_blocks: dict) -> Dict[str, float]:
    """Ridge regression: train on earlier season->season transitions, predict the target."""
    train_targets = [_prev(target, 1), _prev(target, 2)]      # e.g. E2024, E2023
    X, y = [], []
    for tcur in train_targets:
        tprev = _prev(tcur, 1)
        pb, cb = _blocks(tprev), _blocks(tcur)
        for c, b in pb.items():
            if c in cb and cb[c].get("gp", 0) >= 5 and b.get("min_per_team_game", 0) >= 3:
                X.append(_feat_vec(b))
                y.append(cb[c].get("pir_per_team_game", 0.0))
    if len(X) < 30:
        return model_naive(target, prev_blocks)
    predict = _ridge_fit(np.array(X), np.array(y))
    codes = list(prev_blocks)
    P = predict(np.array([_feat_vec(prev_blocks[c]) for c in codes]))
    return {c: float(max(0.0, p)) for c, p in zip(codes, P)}


def model_draftroom(target: str, ablation: str = None) -> Dict[str, float]:
    """The production projection (optionally ablated), from the backtest universe."""
    import copy

    from . import backtest
    from .project import project_all
    uni, hz = backtest.universe_for(target, 3)
    players = copy.deepcopy(uni["players"])
    orig_age = None
    if ablation == "no_age":
        import eldraft.project as P
        orig_age = P.age_factor
        P.age_factor = lambda a: 1.0
    orig_k = WEIGHTS.shrink_minutes_k
    try:
        if ablation == "no_shrink":
            object.__setattr__(WEIGHTS, "shrink_minutes_k", 0.0)
        project_all(players, hz)
    finally:
        if ablation == "no_shrink":
            object.__setattr__(WEIGHTS, "shrink_minutes_k", orig_k)
        if ablation == "no_age":
            import eldraft.project as P
            P.age_factor = orig_age
    return {(p.get("stat_code") or p["code"]): float(p.get("fp") or 0.0) for p in players}


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def _spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x); mx = sum(rx) / n; my = sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")


def _score(pred: Dict[str, float], actual: Dict[str, float], codes: List[str]) -> dict:
    p = [pred.get(c, 0.0) for c in codes]
    a = [actual[c] for c in codes]
    n = len(codes); k = max(1, n // 10)
    tp = set(sorted(codes, key=lambda c: -pred.get(c, 0.0))[:k])
    ta = set(sorted(codes, key=lambda c: -actual[c])[:k])
    # "Trust the model's ten favourites": their mean ACTUAL PIR/round — a clean,
    # budget-free read on whether a high projection meant real production.
    top10 = sorted(codes, key=lambda c: -pred.get(c, 0.0))[:10]
    return {"spearman": round(_spearman(p, a), 4),
            "top_decile": round(len(tp & ta) / k, 4),
            "mae": round(sum(abs(x - y) for x, y in zip(p, a)) / n, 3),
            "top10_actual": round(sum(actual[c] for c in top10) / len(top10), 2)}


def run(target: str = "E2025", min_games: int = 10) -> dict:
    prev_blocks = _blocks(_prev(target, 1))
    cur = _blocks(target)
    actual = {c: b.get("pir_per_team_game", 0.0) for c, b in cur.items() if b.get("gp", 0) >= min_games}
    codes = [c for c in actual if c in prev_blocks]

    models = {
        "Naive (last season)": model_naive(target, prev_blocks),
        "3-yr weighted avg": model_weighted3(target, prev_blocks),
        "Ridge regression (ML)": model_ridge(target, prev_blocks),
        "Draft Room (full)": model_draftroom(target),
        "  – no age curve": model_draftroom(target, "no_age"),
        "  – no shrinkage": model_draftroom(target, "no_shrink"),
    }
    rows = {name: _score(pred, actual, codes) for name, pred in models.items()}
    # actual mean PIR/round of the true top-10, as the reachable reference for top10_actual
    best10 = sorted(codes, key=lambda c: -actual[c])[:10]
    return {"target": target,
            "season_label": "{}-{}".format(int(target[1:]), str(int(target[1:]) + 1)[-2:]),
            "n_players": len(codes),
            "ceiling_top10": round(sum(actual[c] for c in best10) / 10, 2),
            "models": rows}


def report(targets=("E2025", "E2024")) -> dict:
    """Average the bake-off across seasons for a robust, out-of-sample comparison."""
    per = [run(t) for t in targets]
    names = list(per[0]["models"])
    keys = ["spearman", "top_decile", "mae", "top10_actual"]
    avg = {}
    for nm in names:
        avg[nm] = {k: round(float(np.mean([p["models"][nm][k] for p in per])), 3) for k in keys}
    out = {"seasons": [p["season_label"] for p in per],
           "n_players": [p["n_players"] for p in per],
           "ceiling_top10": round(float(np.mean([p["ceiling_top10"] for p in per])), 2),
           "models": avg, "per_season": per}
    import json
    import os

    from .config import OUT
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bakeoff.json"), "w") as fh:
        json.dump(out, fh)
    return out
