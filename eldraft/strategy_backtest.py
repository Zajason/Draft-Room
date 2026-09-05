"""Strategy backtests on last season, using last season's *realised* ability.

The out-of-sample test in season_backtest.py asks "how good is the projection?".  These
two ask a different question — "how good are the *decisions*?" — by handing every player
his true last-season level and then making the manager navigate the parts that are still
genuinely hard and causal: who is injured, which matchups are soft, and (for the classic
game) prices that drift week to week.

  * classic:  the salary-cap game.  Build within budget, then each round re-price every
              player from form, let your squad's value drift, and make up to four transfers.
  * draft:    an exclusive snake draft at tip-off, then hold the roster all season, setting
              the best lineup each week and using waivers to cover injuries.

Injuries are detected causally: a player whose club played the last round(s) but who did
not appear is treated as doubtful, then out — so the manager benches or replaces him using
only what was knowable before the round, exactly as in real life.
"""
from __future__ import annotations

import collections
from typing import Dict, List

import numpy as np

from .config import RULES
from .season_backtest import Season, transfer_solve

CAPS_CLASSIC = (4, 4, 2, 0)     # coach dropped — his weekly score isn't in the feed
PRICE_REF = 22.0                # PIR that maps to a top-of-scale price
PRICE_STEP = 0.25               # max credit move per round
PRICE_PULL = 0.30               # how fast price chases form


def augment(season: Season) -> Season:
    """Precompute the club schedule, realised ability, injury lookups and price paths."""
    if getattr(season, "_augmented", False):
        return season
    # rounds each club actually played, and rounds each player appeared
    club_rounds: Dict[str, List[int]] = collections.defaultdict(list)
    for rnd, games in season.fixtures.items():
        for g in games:
            club_rounds[g["home"]].append(rnd)
            club_rounds[g["away"]].append(rnd)
    season.club_rounds = {t: sorted(rs) for t, rs in club_rounds.items()}
    season.played = {c: set(m.keys()) for c, m in season.actual.items()}

    # realised ability = mean PIR over rounds the player appeared in (his true level)
    season.realized = {}
    for code, p in season.players.items():
        vals = list(season.actual.get(code, {}).values())
        season.realized[code] = float(np.mean(vals)) if vals else p["prior"]

    # deterministic weekly price path, drifting from the modelled pre-season price
    # toward each player's trailing form.
    rounds = season.rounds
    season.price = {c: {} for c in season.players}
    for code, p in season.players.items():
        pr = float(p["price"])
        played = sorted(season.actual.get(code, {}).items())
        for i, rnd in enumerate(rounds):
            # trailing average PIR over appearances strictly before this round
            prior_vals = [v for r, v in played if r < rnd]
            form = float(np.mean(prior_vals)) if prior_vals else season.realized[code]
            target = 4.0 + 12.0 * max(0.0, min(1.4, form / PRICE_REF)) ** 0.9
            target = max(RULES.price_min, min(RULES.price_max, target))
            pr = pr + max(-PRICE_STEP, min(PRICE_STEP, PRICE_PULL * (target - pr)))
            pr = max(RULES.price_min, min(RULES.price_max, pr))
            season.price[code][rnd] = round(pr * 2) / 2.0     # half-credit granularity
    season._augmented = True
    return season


# ======================================================================================
# Injury + realised-ability projection
# ======================================================================================
def injury_factor(season: Season, code: str, rnd: int) -> float:
    """Causal availability for `rnd`: 1 fit, 0.4 doubtful (missed last), 0 out (missed 2)."""
    team = season.club.get(code)
    if not team:
        return 1.0
    prev = [r for r in season.club_rounds.get(team, []) if r < rnd][-2:]
    if not prev:
        return 1.0
    missed = [r for r in prev if r not in season.played.get(code, set())]
    if len(prev) >= 2 and len(missed) == 2:
        return 0.0                                   # two straight DNPs -> out
    if prev[-1] in missed:
        return 0.4                                   # missed the last round -> doubtful
    return 1.0


def proj(season: Season, code: str, rnd: int, upto: int, injury: bool = True) -> float:
    """Realised-ability projection for a round: ability x matchup x games x availability."""
    team = season.club.get(code)
    if not team:
        return 0.0
    inj = injury_factor(season, code, upto) if injury else 1.0
    if inj <= 0:
        return 0.0
    a = season.realized.get(code, 0.0)
    total = 0.0
    for g in season.games_in(team, rnd):
        opp = g["away"] if g["home"] == team else g["home"]
        home = g["home"] == team
        d = season.defense(opp, upto)
        m = max(0.85, min(1.15, d / season.league_avg)) * (1.03 if home else 0.97)
        total += a * m
    return total * inj


def horizon(season: Season, code: str, rnd: int, upto: int, H: int, gamma: float,
            injury: bool = True) -> float:
    return sum((gamma ** i) * proj(season, code, rnd + i, upto, injury) for i in range(H))


def score_round(season: Season, squad: List[str], rnd: int, upto: int,
                rank_by="proj", injury: bool = True) -> float:
    """Set lineup by projection (or by actual for the ceiling), score on actual PIR."""
    if rank_by == "actual":
        key = lambda c: season.actual_pts(c, rnd)
    else:
        key = lambda c: proj(season, c, rnd, upto, injury)
    outs = sorted(squad, key=lambda c: -key(c))
    k = RULES.full_credit_slots
    starters, bench = outs[:k], outs[k:]
    s = sum(season.actual_pts(c, rnd) for c in starters)
    s += RULES.bench_multiplier * sum(season.actual_pts(c, rnd) for c in bench)
    if starters:
        s += (RULES.captain_multiplier - 1.0) * season.actual_pts(starters[0], rnd)
    return s


# ======================================================================================
# Backtest A - Classic salary-cap game, with weekly-changing credits
# ======================================================================================
def _pool(season, rnd, upto, current, prices, score_fn):
    seen, out = set(), []

    def add(code):
        if code in seen or code not in season.players:
            return
        out.append({"code": code, "pos": season.players[code]["pos"],
                    "price": prices[code][rnd], "score": score_fn(code)})
        seen.add(code)

    for c in current:
        add(c)
    for pos in ("G", "F", "C"):
        cand = [c for c in season.players if season.players[c]["pos"] == pos]
        cand.sort(key=lambda c: -score_fn(c))
        for c in cand[:22]:
            add(c)
        cand.sort(key=lambda c: prices[c][rnd])
        for c in cand[:10]:
            add(c)
    return out


def run_classic(season, kind="engine", H=2, gamma=0.6, lam=0.3, max_t=4,
                injury=True, seed=0, noise=0.0):
    """One causal pass through the classic game with credits that drift each round."""
    rng = np.random.default_rng(seed)
    prices = season.price
    squad, bank, total, transfers = None, 0.0, 0.0, 0
    worth_path, per_round = [], []
    for rnd in season.rounds:
        upto = rnd

        def base(c):
            return horizon(season, c, rnd, upto, H, gamma, injury)
        sf = (lambda c: base(c) * (1.0 + noise * rng.standard_normal())) if noise else base

        if squad is None:
            pool = _pool(season, rnd, upto, set(), prices, sf)
            fr, sq = transfer_solve(pool, RULES.budget, set(), 10, CAPS_CLASSIC, RULES)
            squad = next((sq[l] for l in range(len(sq) - 1, -1, -1) if sq[l]), [])
            bank = RULES.budget - sum(prices[c][rnd] for c in squad)
        elif kind in ("engine", "field"):
            worth = sum(prices[c][rnd] for c in squad) + bank
            pool = _pool(season, rnd, upto, set(squad), prices, sf)
            fr, sq = transfer_solve(pool, worth, set(squad), max_t, CAPS_CLASSIC, RULES)
            best, bt = -1e18, 0
            for lvl in range(max_t + 1):
                if fr[lvl] is None:
                    continue
                v = fr[lvl] - lam * lvl
                if v > best:
                    best, bt = v, lvl
            if sq[bt]:
                transfers += sum(1 for c in sq[bt] if c not in set(squad))
                squad = sq[bt]
                bank = worth - sum(prices[c][rnd] for c in squad)
        elif kind == "hindsight":
            worth = sum(prices[c][rnd] for c in squad) + bank
            sfa = lambda c: season.actual_pts(c, rnd)
            pool = _pool(season, rnd, upto, set(squad), prices, sfa)
            fr, sq = transfer_solve(pool, worth, set(squad), max_t, CAPS_CLASSIC, RULES)
            if sq[max_t]:
                transfers += sum(1 for c in sq[max_t] if c not in set(squad))
                squad = sq[max_t]
                bank = worth - sum(prices[c][rnd] for c in squad)
        # kind == "hold": keep the squad, set best lineup, never transfer

        rb = "actual" if kind == "hindsight" else "proj"
        pts = score_round(season, squad, rnd, upto, rank_by=rb, injury=injury)
        total += pts
        per_round.append(round(pts, 1))
        worth_path.append(round(sum(prices[c][rnd] for c in squad) + bank, 1))
    return {"total": round(total, 1), "transfers": transfers,
            "final_worth": worth_path[-1] if worth_path else RULES.budget,
            "per_round": per_round, "worth_path": worth_path}


# ======================================================================================
# Backtest B - Exclusive draft at tip-off, then hold and manage the roster
# ======================================================================================
def _snake(n_teams, picks_each):
    order = []
    for r in range(picks_each):
        seats = range(n_teams) if r % 2 == 0 else range(n_teams - 1, -1, -1)
        order.extend(seats)
    return order


def run_draft(season, n_teams=10, engine_seat=0, waivers=True, seed=0, noise_field=0.45):
    """Snake-draft on last-season ability, then play the year out holding the roster.

    No credits — draft leagues are exclusive ownership.  The engine drafts by value over
    the next available player (positional scarcity), the field drafts by ability with
    noise.  Through the season every team sets its best lineup, and — if waivers are on —
    swaps an injured player for the best free agent, which is where good injury management
    shows up in a draft league.
    """
    rng = np.random.default_rng(seed)
    ability = {c: season.realized.get(c, 0.0) for c in season.players}
    pos = {c: season.players[c]["pos"] for c in season.players}
    need_tpl = {"G": 4, "F": 4, "C": 2}
    rosters = [[] for _ in range(n_teams)]
    needs = [dict(need_tpl) for _ in range(n_teams)]
    taken = set()

    for seat in _snake(n_teams, 10):
        need_pos = [p for p, k in needs[seat].items() if k > 0]
        cand = [c for c in season.players if c not in taken and pos[c] in need_pos]
        if not cand:
            continue
        if seat == engine_seat:
            # value over next-available: ability minus the replacement likely to survive
            repl = {}
            for p in need_pos:
                vals = sorted((ability[c] for c in cand if pos[c] == p), reverse=True)
                repl[p] = vals[min(len(vals) - 1, n_teams)] if vals else 0.0
            pick = max(cand, key=lambda c: ability[c] - repl[pos[c]])
        else:
            pick = max(cand, key=lambda c: ability[c] * (1.0 + noise_field * rng.standard_normal()))
        rosters[seat].append(pick)
        taken.add(pick)
        needs[seat][pos[pick]] -= 1

    totals = [0.0] * n_teams
    for rnd in season.rounds:
        upto = rnd
        for seat in range(n_teams):
            roster = rosters[seat]
            if waivers:
                for i, c in enumerate(roster):
                    if injury_factor(season, c, rnd) <= 0.0:
                        p = pos[c]
                        avail = [x for x in season.players
                                 if x not in taken and pos[x] == p
                                 and injury_factor(season, x, rnd) > 0.0]
                        if avail:
                            if seat == engine_seat:
                                best = max(avail, key=lambda x: proj(season, x, rnd, upto))
                            else:
                                best = max(avail, key=lambda x: ability[x] * (1.0 + noise_field * rng.standard_normal()))
                            taken.discard(c)
                            taken.add(best)
                            roster[i] = best
            totals[seat] += score_round(season, roster, rnd, upto, rank_by="proj")

    order = sorted(range(n_teams), key=lambda s: -totals[s])
    rank = order.index(engine_seat) + 1
    return {"engine_total": round(totals[engine_seat], 1),
            "totals": [round(t, 1) for t in totals],
            "rank": rank, "n_teams": n_teams,
            "won": rank == 1, "top3": rank <= 3}


def report_draft(season, n_teams=10, n_drafts=20):
    """Average the engine's finish over many drafts (seat + field vary each time)."""
    ranks, wins, top3, tots, fields = [], 0, 0, [], []
    for d in range(n_drafts):
        seat = d % n_teams
        r = run_draft(season, n_teams=n_teams, engine_seat=seat, seed=1000 + d)
        ranks.append(r["rank"])
        wins += r["won"]
        top3 += r["top3"]
        tots.append(r["engine_total"])
        fields.append(np.mean([r["totals"][s] for s in range(n_teams) if s != seat]))
    return {
        "n_teams": n_teams, "n_drafts": n_drafts,
        "avg_rank": round(float(np.mean(ranks)), 2),
        "win_pct": round(100 * wins / n_drafts, 1),
        "top3_pct": round(100 * top3 / n_drafts, 1),
        "engine_mean": round(float(np.mean(tots)), 1),
        "field_mean": round(float(np.mean(fields)), 1),
    }


# ======================================================================================
# Combined report
# ======================================================================================
def run_report(target="E2025", n_field=24, n_drafts=24) -> dict:
    import json
    import os

    from .config import OUT
    s = augment(Season(target))
    y = int(target[1:])
    label = "{}-{}".format(y, str(y + 1)[-2:])

    # --- classic ---
    eng = run_classic(s, "engine")
    hold = run_classic(s, "hold")
    noinj = run_classic(s, "engine", injury=False)
    hind = run_classic(s, "hindsight")
    field = np.array([run_classic(s, "field", seed=200 + i,
                                  noise=0.12 + 0.5 * (i / max(1, n_field - 1)))["total"]
                      for i in range(n_field)])
    classic = {
        "engine": eng["total"], "hold": hold["total"], "no_injury": noinj["total"],
        "hindsight": hind["total"], "field_mean": round(float(field.mean()), 1),
        "field_best": round(float(field.max()), 1),
        "pct_of_ceiling": round(100 * eng["total"] / hind["total"], 1),
        "beat_field_pct": round(100 * float((eng["total"] > field).mean()), 1),
        "gain_transfers": round(eng["total"] - hold["total"], 1),
        "gain_injury": round(eng["total"] - noinj["total"], 1),
        "worth_start": eng["worth_path"][0], "worth_end": eng["final_worth"],
        "transfers": eng["transfers"],
        "worth_path": eng["worth_path"],
        "series_engine": np.cumsum(eng["per_round"]).round(1).tolist(),
        "series_hold": np.cumsum(hold["per_round"]).round(1).tolist(),
        "series_hind": np.cumsum(hind["per_round"]).round(1).tolist(),
        "series_noinj": np.cumsum(noinj["per_round"]).round(1).tolist(),
    }

    # --- draft (with waivers = injury management on, vs off) ---
    dr_on = report_draft(s, n_teams=10, n_drafts=n_drafts)
    ranks_off = []
    for d in range(n_drafts):
        r = run_draft(s, n_teams=10, engine_seat=d % 10, seed=1000 + d, waivers=False)
        ranks_off.append(r["rank"])
    draft = dict(dr_on)
    draft["avg_rank_no_waivers"] = round(float(np.mean(ranks_off)), 2)

    out = {"season": target, "season_label": label, "rounds": len(s.rounds),
           "classic": classic, "draft": draft}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "strategy_backtest.json"), "w") as fh:
        json.dump(out, fh)
    return out
