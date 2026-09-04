"""Season backtest: replay a full EuroLeague Fantasy season week by week.

This grades the *whole* weekly strategy, not just the projection: every round the engine
updates its read on each player from results so far, picks a lineup (starting six +
captain) and makes up to four transfers within budget — all on information available
before the round — and is then scored on what actually happened.  The honest question it
answers is "would actively managing this squad with the engine have worked?", measured
against a set-and-forget manager, a belief-that-never-updates manager, the perfect-hindsight
ceiling, and a simulated field.

Everything is causal: at round R the engine sees only rounds 1..R-1.  Ground truth (actual
fantasy points = PIR per round) comes from the box scores; nothing here is fitted to the
season it scores.
"""
from __future__ import annotations

import collections
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import build, fetch
from .config import RULES


# ======================================================================================
# Ground truth: actual fantasy points (PIR) per player per round
# ======================================================================================
def actual_round_pir(season: str) -> Dict[str, Dict[int, float]]:
    """{player_code: {round: total PIR that round}} from cached box scores."""
    boxes = fetch.season_boxscores(season, cache_only=True)
    games = {g["gameCode"]: g for g in fetch.season_games(season)}
    out: Dict[str, Dict[int, float]] = collections.defaultdict(dict)
    for gcode, box in boxes.items():
        meta = games.get(gcode, {})
        rnd = meta.get("round")
        phase = ((meta.get("phaseType") or {}).get("code")) or "RS"
        if not rnd or phase != "RS":
            continue                      # regular season only — playoffs change the format
        for side in ("local", "road"):
            for entry in (box.get(side) or {}).get("players", []) or []:
                pl = entry.get("player") or {}
                code = (pl.get("person") or {}).get("code")
                st = entry.get("stats") or {}
                if not code:
                    continue
                v = float(st.get("valuation") or 0.0)
                out[code][rnd] = out[code].get(rnd, 0.0) + v
    return out


def team_of_round(season: str) -> Dict[str, Dict[int, str]]:
    """{player_code: {round: club_code}} — which club a player suited up for each round."""
    boxes = fetch.season_boxscores(season, cache_only=True)
    games = {g["gameCode"]: g for g in fetch.season_games(season)}
    out: Dict[str, Dict[int, str]] = collections.defaultdict(dict)
    for gcode, box in boxes.items():
        rnd = (games.get(gcode) or {}).get("round")
        if not rnd:
            continue
        for side in ("local", "road"):
            for entry in (box.get(side) or {}).get("players", []) or []:
                pl = entry.get("player") or {}
                code = (pl.get("person") or {}).get("code")
                club = (pl.get("club") or {}).get("code") or (pl.get("team") or {}).get("code")
                if code and club:
                    out[code][rnd] = club
    return out


def round_fixtures(season: str) -> Dict[int, List[dict]]:
    games = fetch.season_games(season)
    by_round: Dict[int, List[dict]] = collections.defaultdict(list)
    for g in games:
        rnd = g.get("round")
        phase = ((g.get("phaseType") or {}).get("code")) or "RS"
        if not rnd or phase != "RS":
            continue
        home = ((g.get("local") or {}).get("club") or {}).get("code")
        away = ((g.get("road") or {}).get("club") or {}).get("code")
        if home and away:
            by_round[rnd].append({"home": home, "away": away})
    return dict(by_round)


# ======================================================================================
# Transfer-constrained optimiser (Python port of the verified in-browser DP)
# ======================================================================================
POS = ("G", "F", "C", "H")
_PIX = {p: i for i, p in enumerate(POS)}
NEG = -1e9


def _units(price: float) -> int:
    return int(round(price / 0.5))


def transfer_solve(pool: List[dict], budget: float, current: set, max_t: int,
                   caps: Tuple[int, int, int, int], rules=RULES):
    """Best squad reachable from `current` within `max_t` transfers, per transfer level.

    `pool` entries: {code, pos, price, score}.  Keeping a player already in `current` is
    free; anyone new costs one transfer.  Returns (frontier, squads) where frontier[t] is
    the best value using at most t transfers and squads[t] the codes achieving it.
    Exact — the free squad DP with an added transfers dimension (see the JS twin's
    brute-force test).
    """
    B = _units(budget)
    T = max_t
    dims = (caps[0] + 1, caps[1] + 1, caps[2] + 1, caps[3] + 1, B + 1, T + 1)
    n = len(pool)
    units = [_units(p["price"]) for p in pool]
    pix = [_PIX.get(p["pos"], 1) for p in pool]
    score = [float(p["score"]) for p in pool]
    is_new = [0 if p["code"] in current else 1 for p in pool]
    order = sorted(range(n), key=lambda i: -score[i])
    full = rules.full_credit_slots
    capm = rules.captain_multiplier
    benchm = rules.bench_multiplier

    def mult(g, f, c):
        return 1.0 if (g + f + c) < full else benchm

    fwd = [None] * (n + 1)
    cur = np.full(dims, NEG, dtype=np.float32)
    cur[0, 0, 0, 0, 0, 0] = 0.0
    fwd[0] = cur
    for k in range(n):
        i = order[k]
        ax, cost, s, nt = pix[i], units[i], score[i], is_new[i]
        cap = caps[ax]
        nxt = cur.copy()
        if cap > 0 and cost <= B:
            for g in range(dims[0]):
                for f in range(dims[1]):
                    for c in range(dims[2]):
                        for h in range(dims[3]):
                            if (g, f, c, h)[ax] >= cap:
                                continue
                            gain = s if ax == 3 else mult(g, f, c) * s
                            if ax != 3 and g == 0 and f == 0 and c == 0:
                                gain += (capm - 1.0) * s
                            d = [g, f, c, h]; d[ax] += 1
                            src = cur[g, f, c, h, : B + 1 - cost, : T + 1 - nt]
                            dst = nxt[d[0], d[1], d[2], d[3], cost:, nt:]
                            np.maximum(dst, src + gain, out=dst)
        fwd[k + 1] = nxt
        cur = nxt

    term = (caps[0], caps[1], caps[2], caps[3])
    final = cur[term]           # shape (B+1, T+1)
    best_at_most = [NEG] * (T + 1)
    arg = [(0, 0)] * (T + 1)
    for lvl in range(T + 1):
        sub = final[:, : lvl + 1]
        b_idx, t_idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
        best_at_most[lvl] = float(sub[b_idx, t_idx])
        arg[lvl] = (int(b_idx), int(t_idx))

    def recover(lvl):
        b0, t0 = arg[lvl]
        state = [caps[0], caps[1], caps[2], caps[3], b0, t0]
        chosen = []
        for k in range(n, 0, -1):
            i = order[k - 1]
            ax, cost, s, nt = pix[i], units[i], score[i], is_new[i]
            if abs(fwd[k - 1][tuple(state)] - fwd[k][tuple(state)]) < 1e-4:
                continue
            cand = list(state); cand[ax] -= 1; cand[4] -= cost; cand[5] -= nt
            if cand[ax] < 0 or cand[4] < 0 or cand[5] < 0:
                continue
            g, f, c = cand[0], cand[1], cand[2]
            gain = s if ax == 3 else mult(g, f, c) * s
            if ax != 3 and g == 0 and f == 0 and c == 0:
                gain += (capm - 1.0) * s
            if abs(fwd[k - 1][tuple(cand)] + gain - fwd[k][tuple(state)]) < 1e-3:
                chosen.append(pool[i]["code"]); state = cand
        return chosen

    frontier, squads = [], []
    for lvl in range(T + 1):
        ok = best_at_most[lvl] > NEG / 2
        frontier.append(best_at_most[lvl] if ok else None)
        squads.append(recover(lvl) if ok else None)
    return frontier, squads


# ======================================================================================
# Season simulation
# ======================================================================================
def _shrink(mean, n, prior, k):
    return (n * mean + k * prior) / (n + k) if (n + k) > 0 else prior


class Season:
    """Everything a strategy needs to play one causal pass through a season."""

    def __init__(self, target="E2025", history=3, caps=(4, 4, 2, 0),
                 k_prior=10.0, horizon=2, gamma=0.6, lam=0.3, max_t=4):
        from . import backtest
        self.caps, self.k_prior = caps, k_prior
        self.H, self.gamma, self.lam, self.max_t = horizon, gamma, lam, max_t

        uni, hz = backtest.universe_for(target, history)
        from .project import project_all
        import copy
        players = copy.deepcopy(uni["players"])
        project_all(players, hz)

        self.players = {}
        for p in players:
            code = p.get("stat_code") or p["code"]      # match actuals by stat code
            if p["position"] == "H":
                continue
            self.players[code] = {
                "code": code, "name": p["name"], "pos": p["position"],
                "price": float(p.get("price") or RULES.price_min),
                "prior": float(p.get("fp") or 0.0),
            }
        self.actual = actual_round_pir(target)
        self.fixtures = round_fixtures(target)
        self.rounds = sorted(self.fixtures)

        # per-round PIR conceded by each club (for the causal defence estimate)
        self.against = collections.defaultdict(dict)
        allv = []
        for rnd, games in self.fixtures.items():
            # team total PIR that round = sum of its players' actual PIR
            tp = collections.defaultdict(float)
            tof = team_of_round(target)
            # cheaper: derive from actuals + team_of_round once (built lazily below)
        self._team_round = team_of_round(target)
        team_pir = collections.defaultdict(lambda: collections.defaultdict(float))
        for code, rmap in self.actual.items():
            for rnd, v in rmap.items():
                club = self._team_round.get(code, {}).get(rnd)
                if club:
                    team_pir[club][rnd] += v
        for rnd, games in self.fixtures.items():
            for g in games:
                self.against[g["home"]][rnd] = team_pir[g["away"]].get(rnd, 0.0)
                self.against[g["away"]][rnd] = team_pir[g["home"]].get(rnd, 0.0)
                allv += [team_pir[g["away"]].get(rnd, 0.0), team_pir[g["home"]].get(rnd, 0.0)]
        self.league_avg = float(np.mean(allv)) if allv else 90.0

        # club of each player per round (for fixture lookup); fall back to modal club
        self.club = {}
        for code in self.players:
            rmap = self._team_round.get(code, {})
            if rmap:
                self.club[code] = collections.Counter(rmap.values()).most_common(1)[0][0]

    # -- causal estimates -----------------------------------------------------------
    def est(self, code, upto_round):
        p = self.players[code]
        vals = [v for r, v in self.actual.get(code, {}).items() if r < upto_round]
        return _shrink(float(np.mean(vals)) if vals else p["prior"], len(vals), p["prior"], self.k_prior)

    def defense(self, team, upto_round):
        vals = [v for r, v in self.against.get(team, {}).items() if r < upto_round]
        return _shrink(float(np.mean(vals)) if vals else self.league_avg, len(vals), self.league_avg, 5.0)

    def games_in(self, team, rnd):
        gs = self.fixtures.get(rnd, [])
        return [g for g in gs if team in (g["home"], g["away"])]

    def proj(self, code, rnd, upto_round):
        """Projected points for `code` in `rnd`, using beliefs formed by `upto_round`."""
        team = self.club.get(code)
        if not team:
            return 0.0
        e = self.est(code, upto_round)
        total = 0.0
        for g in self.games_in(team, rnd):
            opp = g["away"] if g["home"] == team else g["home"]
            home = g["home"] == team
            d = self.defense(opp, upto_round)
            m = max(0.85, min(1.15, d / self.league_avg)) * (1.03 if home else 0.97)
            total += e * m
        return total

    def horizon_val(self, code, rnd, upto_round):
        return sum((self.gamma ** i) * self.proj(code, rnd + i, upto_round) for i in range(self.H))

    def actual_pts(self, code, rnd):
        return self.actual.get(code, {}).get(rnd, 0.0)

    # -- scoring one round given a squad --------------------------------------------
    def score_round(self, squad, rnd, upto_round, rank_by="proj"):
        """Set lineup by projection (or actual for the hindsight ceiling), score on actual."""
        key = (lambda c: self.actual_pts(c, rnd)) if rank_by == "actual" \
            else (lambda c: self.proj(c, rnd, upto_round))
        outs = sorted([c for c in squad], key=lambda c: -key(c))
        k = RULES.full_credit_slots
        starters, bench = outs[:k], outs[k:]
        cap = starters[0] if starters else None
        s = sum(self.actual_pts(c, rnd) for c in starters)
        s += RULES.bench_multiplier * sum(self.actual_pts(c, rnd) for c in bench)
        if cap:
            s += (RULES.captain_multiplier - 1.0) * self.actual_pts(cap, rnd)
        return s

    # -- transfer pool at a round ---------------------------------------------------
    def pool(self, rnd, upto_round, current, score_fn):
        seen, out = set(), []
        def add(code):
            if code in seen or code not in self.players:
                return
            out.append({"code": code, "pos": self.players[code]["pos"],
                        "price": self.players[code]["price"], "score": score_fn(code)})
            seen.add(code)
        for c in current:
            add(c)
        for pos in ("G", "F", "C"):
            cand = [c for c in self.players if self.players[c]["pos"] == pos]
            cand.sort(key=lambda c: -score_fn(c))
            for c in cand[:22]:
                add(c)
            cand.sort(key=lambda c: self.players[c]["price"])
            for c in cand[:10]:
                add(c)
        return out

    def free_build(self, rnd, upto_round, score_fn):
        pl = self.pool(rnd, upto_round, set(), score_fn)
        fr, sq = transfer_solve(pl, RULES.budget, set(), self.caps[0] + self.caps[1] + self.caps[2],
                                self.caps, RULES)
        # last non-null squad = unconstrained optimum
        for lvl in range(len(sq) - 1, -1, -1):
            if sq[lvl]:
                return sq[lvl]
        return []


def run_strategy(season: Season, kind="engine", seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    rounds = season.rounds
    total = 0.0
    transfers = 0
    squad = None
    per_round = []

    def noisy(code, rnd, upto):
        base = season.horizon_val(code, rnd, upto)
        return base * (1.0 + noise * rng.standard_normal()) if noise else base

    for idx, rnd in enumerate(rounds):
        upto = rnd  # beliefs use rounds < rnd
        if squad is None:
            # initial build on this round's look-ahead
            sf = (lambda c: noisy(c, rnd, upto)) if kind in ("engine", "field") else \
                 (lambda c: season.horizon_val(c, rnd, upto))
            squad = season.free_build(rnd, upto, sf)
        elif kind == "engine" or kind == "field":
            sf = lambda c: noisy(c, rnd, upto)
            pl = season.pool(rnd, upto, set(squad), sf)
            cur = set(squad)
            fr, sq = transfer_solve(pl, RULES.budget, cur, season.max_t, season.caps, RULES)
            base = fr[0]
            best, bt = -1e18, 0
            for lvl in range(season.max_t + 1):
                if fr[lvl] is None:
                    continue
                v = fr[lvl] - season.lam * lvl
                if v > best:
                    best, bt = v, lvl
            if sq[bt]:
                transfers += sum(1 for c in sq[bt] if c not in cur)
                squad = sq[bt]
        elif kind == "hindsight":
            sf = lambda c: season.actual_pts(c, rnd)   # perfect foresight this round
            pl = season.pool(rnd, upto, set(squad), sf)
            cur = set(squad)
            fr, sq = transfer_solve(pl, RULES.budget, cur, season.max_t, season.caps, RULES)
            if sq[season.max_t]:
                transfers += sum(1 for c in sq[season.max_t] if c not in cur)
                squad = sq[season.max_t]
        # kind == "hold": keep squad, no transfers
        rb = "actual" if kind == "hindsight" else "proj"
        pts = season.score_round(squad, rnd, upto, rank_by=rb)
        total += pts
        per_round.append(pts)
    return {"total": round(total, 1), "transfers": transfers,
            "per_round": [round(x, 1) for x in per_round]}


# ======================================================================================
# Report
# ======================================================================================
def run_report(target="E2025", n_field=40, seed=0) -> dict:
    """Play the season with every strategy and assemble the comparison."""
    import os, json
    from .config import OUT
    s = Season(target)
    eng = run_strategy(s, "engine")
    hold = run_strategy(s, "hold")
    hind = run_strategy(s, "hindsight")
    static = run_strategy(Season(target, k_prior=1e6), "engine")
    rng = np.random.default_rng(seed)
    field = []
    for i in range(n_field):
        sig = 0.12 + 0.5 * (i / max(1, n_field - 1))
        field.append(run_strategy(s, "field", seed=1000 + i, noise=sig))
    field_tot = np.array([f["total"] for f in field])
    beat = float((eng["total"] > field_tot).mean())

    def yr(t):
        y = int(t[1:])
        return "{}-{}".format(y, str(y + 1)[-2:])   # E2025 -> 2025-26

    out = {
        "season": target, "season_label": yr(target), "rounds": len(s.rounds),
        "totals": {
            "engine": eng["total"], "hold": hold["total"],
            "no_update": static["total"], "hindsight": hind["total"],
            "field_mean": round(float(field_tot.mean()), 1),
            "field_best": round(float(field_tot.max()), 1),
            "field_worst": round(float(field_tot.min()), 1),
        },
        "transfers": {"engine": eng["transfers"], "hindsight": hind["transfers"]},
        "pct_of_ceiling": round(100 * eng["total"] / hind["total"], 1),
        "gain_vs_hold": round(eng["total"] - hold["total"], 1),
        "gain_vs_hold_pct": round(100 * (eng["total"] / hold["total"] - 1), 1),
        "gain_vs_no_update": round(eng["total"] - static["total"], 1),
        "field_beat_pct": round(100 * beat, 1),
        "n_field": n_field,
        "series": {
            "engine": np.cumsum(eng["per_round"]).round(1).tolist(),
            "hold": np.cumsum(hold["per_round"]).round(1).tolist(),
            "hindsight": np.cumsum(hind["per_round"]).round(1).tolist(),
            "no_update": np.cumsum(static["per_round"]).round(1).tolist(),
            "field_band": [
                np.cumsum([f["per_round"][r] for f in field], axis=0).tolist()
                for r in range(0)
            ],
        },
        "field_series": [np.cumsum(f["per_round"]).round(1).tolist() for f in field],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "season_backtest.json"), "w") as fh:
        json.dump(out, fh)
    return out
