"""Weekly lineup scoring under the real Classic rules, with reactive substitutions.

The game does NOT freeze your full/bench split on pre-week projections.  Each round you
field a starting five in a G-F-C formation, plus a "sixth man" who is always a guard - six
players at full credit, the captain doubled - and four bench players at half.  Games are
spread Tuesday to Friday and you may re-set players who have not tipped off yet, so the
score you end up with reflects an informed, *reactive* choice of who counts full.

That optionality is the whole point of a good bench, and the old objective threw it away
by ranking on the mean and calling the top six "starters".  Here we:

  * :func:`best_lineup` - the exact best legal split of a set of realised scores (the
    within-week ceiling, and the leaf valuation for the reactive policy);
  * :func:`reactive_round` - the realistic score under day-by-day information;
  * :func:`weekly_ev` - Monte-Carlo expected weekly points for a squad, which is what the
    optimiser should actually maximise.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RULES

# Valid starting-five formations as (guards, forwards, centers).
FORMATIONS: Tuple[Tuple[int, int, int], ...] = (
    (2, 1, 2), (2, 2, 1), (1, 3, 1), (3, 1, 1), (1, 2, 2),
)

# The sixth man is ALWAYS a guard, so the full-credit six is a formation five plus one
# extra guard.  These are therefore the only legal (G, F, C) counts for the full six.
FULL6: Tuple[Tuple[int, int, int], ...] = tuple((g + 1, f, c) for g, f, c in FORMATIONS)
_FULL6SET = frozenset(FULL6)

_FULL = RULES.full_credit_slots            # 6 players score full
_BENCHMULT = RULES.bench_multiplier        # the other 4 score half
_CAPX = RULES.captain_multiplier - 1.0     # captain bonus on top of a full slot


def _by_pos(scores: Sequence[float], pos: Sequence[str]) -> Dict[str, List[float]]:
    d: Dict[str, List[float]] = {"G": [], "F": [], "C": []}
    for s, p in zip(scores, pos):
        d.get(p, d["F"]).append(s)
    for k in d:
        d[k].sort(reverse=True)
    return d


def best_lineup(scores: Sequence[float], pos: Sequence[str]) -> float:
    """Highest weekly total for these realised scores under formation + 6th-man + captain.

    Full-credit six = a formation five plus one extra guard; the captain (the best of the
    six) is doubled; everyone else scores half.  Exact: we try every legal six-man shape,
    fill each position with its best available scorers, and keep the maximum.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    bp = _by_pos(scores, pos)
    total_all = sum(scores)
    best = None
    for g6, f6, c6 in FULL6:
        if len(bp["G"]) < g6 or len(bp["F"]) < f6 or len(bp["C"]) < c6:
            continue
        # Full six = top scorers filling this formation-plus-a-guard shape.
        full = bp["G"][:g6] + bp["F"][:f6] + bp["C"][:c6]
        full_sum = sum(full)
        # Bench (everyone not in the full six) scores half; captain doubles the best full.
        bench_sum = total_all - full_sum
        cap = max(full) if full else 0.0
        val = full_sum + _BENCHMULT * bench_sum + _CAPX * cap
        if best is None or val > best:
            best = val
    if best is None:                        # squad can't field a legal six (degenerate)
        s = sorted(scores, reverse=True)
        full = s[:_FULL]
        best = sum(full) + _BENCHMULT * sum(s[_FULL:]) + (_CAPX * full[0] if full else 0.0)
    return best


def _choose_full(values: Sequence[float], pos: Sequence[str],
                 force_in: Optional[set] = None,
                 force_out: Optional[set] = None) -> Optional[frozenset]:
    """Indices of the best full-credit six (formation-valid) maximising sum(values).

    ``force_in`` / ``force_out`` pin players an earlier game-day already locked.  Small n
    (a 10-man squad) so an exact search over the top candidates is cheap and clearly right.
    """
    import itertools
    force_in = force_in or set()
    force_out = force_out or set()
    n = len(values)
    if len(force_in) > _FULL:
        return None
    free = [i for i in range(n) if i not in force_in and i not in force_out]
    need = _FULL - len(force_in)
    best, best_set = None, None
    for extra in itertools.combinations(free, min(need, len(free))):
        full = set(force_in) | set(extra)
        if len(full) != _FULL and len(full) != n:      # must field six unless squad < 6
            if len(full) < _FULL and len(free) + len(force_in) >= _FULL:
                continue
        cnt = {"G": 0, "F": 0, "C": 0}
        for i in full:
            cnt[pos[i]] = cnt.get(pos[i], 0) + 1
        # Full-six must be a formation five plus the extra guard - an exact legal shape.
        if (cnt["G"], cnt["F"], cnt["C"]) not in _FULL6SET:
            continue
        v = sum(values[i] for i in full)
        if best is None or v > best:
            best, best_set = v, frozenset(full)
    return best_set


def choose_full_fast(key: Sequence[float], pos: Sequence[str],
                     available: Optional[Sequence[bool]] = None) -> frozenset:
    """Fast formation-legal best-six by ``key`` (e.g. projection), preferring available.

    Hot path for the Monte-Carlo objective.  Equivalent to :func:`_choose_full` with no
    forced players; unavailable players are pushed to the back but may still fill a slot
    the formation leaves no legal alternative for.
    """
    n = len(key)
    if available is None:
        available = [True] * n
    # Rank key, with a big penalty for the unavailable so they are chosen only if forced.
    adj = [key[i] - (0.0 if available[i] else 1e6) for i in range(n)]
    by_pos = {"G": [], "F": [], "C": []}
    for i in range(n):
        by_pos.get(pos[i], by_pos["F"]).append(i)
    for k in by_pos:
        by_pos[k].sort(key=lambda i: -adj[i])
    best_val, best_set = None, None
    for g6, f6, c6 in FULL6:
        if len(by_pos["G"]) < g6 or len(by_pos["F"]) < f6 or len(by_pos["C"]) < c6:
            continue
        full = by_pos["G"][:g6] + by_pos["F"][:f6] + by_pos["C"][:c6]
        val = sum(adj[i] for i in full)
        if best_val is None or val > best_val:
            best_val, best_set = val, frozenset(full)
    if best_set is None:
        order = sorted(range(n), key=lambda i: -adj[i])
        best_set = frozenset(order[:_FULL])
    return best_set


def _score_full(realized: Sequence[float], full: set) -> float:
    full_sum = sum(realized[i] for i in full)
    bench_sum = sum(realized[i] for i in range(len(realized)) if i not in full)
    cap = max((realized[i] for i in full), default=0.0)
    return full_sum + _BENCHMULT * bench_sum + _CAPX * cap


def reactive_round(realized: Sequence[float], mean: Sequence[float],
                   pos: Sequence[str], day: Sequence[int]) -> float:
    """Weekly score under day-by-day information: you lock each player before he tips off.

    Before each game-day you may still move anyone who has not played, so the split for
    that day's players is decided on the results already in (realised) plus expectations
    (the mean) for everyone still to come.  Once a player has tipped off his slot is fixed.
    """
    n = len(realized)
    locked_full: set = set()
    locked_bench: set = set()
    for d in sorted(set(day)):
        val = [realized[i] if day[i] < d else mean[i] for i in range(n)]
        full = _choose_full(val, pos, force_in=locked_full, force_out=locked_bench)
        if full is None:
            full = locked_full
        for i in range(n):
            if day[i] == d and i not in locked_full and i not in locked_bench:
                (locked_full if i in full else locked_bench).add(i)
    return _score_full(realized, locked_full)


def commit_lineup(mean: Sequence[float], pos: Sequence[str]) -> float:
    """Old behaviour, kept for comparison: full/bench fixed on the mean, scored on mean.

    (Feed realised scores as `mean` to reproduce ``score_round(rank_by="proj")``-style
    scoring where the split is decided by a *separate* ranking - see the backtest.)
    """
    order = sorted(range(len(mean)), key=lambda i: -mean[i])
    full = order[:_FULL]
    bench = order[_FULL:]
    s = sum(mean[i] for i in full) + _BENCHMULT * sum(mean[i] for i in bench)
    if full:
        s += _CAPX * mean[full[0]]
    return s


# ======================================================================================
# Monte-Carlo squad objective: value a squad by its availability-aware weekly points
# ======================================================================================
def default_pplay(p: dict) -> float:
    """Prior probability a player suits up for a given round, from his projected role.

    Rotation regulars miss the odd game; deep-bench and fringe players miss more.  Scaled
    off projected minutes when we have them, else a flat rotation prior.
    """
    mpg = p.get("mpg_proj") or p.get("mpg_when_playing")
    if mpg is None:
        return 0.88
    return float(min(0.97, max(0.70, 0.70 + 0.012 * float(mpg))))


class WeeklyObjective:
    """Expected weekly points for a squad under availability-aware lineup selection.

    Each simulated round draws every player's availability (a DNP scores nothing) and his
    fantasy score, then counts his *best legal full-six among the players who suit up* -
    so a bench player earns full credit exactly in the weeks a starter is out.  That is
    what the old linear "top-six by mean" objective never priced, and why it bought
    four-credit bench filler.  Common random numbers (fixed draws) make swap comparisons
    low-noise.  For speed the vectorised score takes the six best available by mean and
    does not enforce the exact guard-heavy six-man shape - a mild approximation, which is
    why this is an analysis tool rather than the shipped optimiser.
    """

    def __init__(self, pool: Sequence[dict], sims: int = 512, seed: int = 0):
        self.pool = list(pool)
        n = len(self.pool)
        self.pos = [p.get("position", "F") for p in self.pool]
        self.mean = np.array([float(p.get("fp") or 0.0) for p in self.pool])
        sigma = np.array([float(p.get("fp_sigma") or 0.5 * (p.get("fp") or 0.0)) for p in self.pool])
        pplay = np.array([float(p.get("pplay", default_pplay(p))) for p in self.pool])
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, sims))
        u = rng.random((n, sims))
        self.sims = sims
        self.avail = u < pplay[:, None]                                   # [n, sims]
        self.realized = np.maximum(0.0, self.mean[:, None] + sigma[:, None] * z) * self.avail

    def value(self, idx: Sequence[int]) -> float:
        """Expected weekly points for the squad given by pool indices `idx` (10 outfield)."""
        idx = list(idx)
        R = self.realized[idx]                       # [k, sims] realised points
        A = self.avail[idx]                          # [k, sims] availability
        mean = self.mean[idx]
        order = np.argsort(-mean)                    # start your highest-projected first
        Ro, Ao = R[order], A[order]
        cum = np.cumsum(Ao, axis=0)                  # rank among the available
        full = Ao & (cum <= _FULL)                   # the six best available count full
        full_sum = (Ro * full).sum(axis=0)
        total = Ro.sum(axis=0)
        cap = np.where(full, Ro, -np.inf).max(axis=0)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        per_round = full_sum + _BENCHMULT * (total - full_sum) + _CAPX * cap
        return float(per_round.mean())


def optimise_squad(pool: Sequence[dict], seed_idx: Sequence[int], budget: float,
                   caps: Tuple[int, int, int] = (4, 4, 2), sims: int = 512,
                   seed: int = 0, max_passes: int = 8) -> dict:
    """Local search over the squad under :class:`WeeklyObjective`, seeded by `seed_idx`.

    Single-position swaps keep the G/F/C shape legal by construction; a swap is taken only
    if it raises the (common-random-number) expected weekly score within budget.
    """
    obj = WeeklyObjective(pool, sims=sims, seed=seed)
    price = [float(p.get("price") or 0.0) for p in pool]
    pos = [p.get("position", "F") for p in pool]
    idx = list(seed_idx)
    cur_val = obj.value(idx)
    by_pos: Dict[str, List[int]] = {"G": [], "F": [], "C": []}
    for j in range(len(pool)):
        by_pos.get(pos[j], by_pos["F"]).append(j)
    for _ in range(max_passes):
        improved = False
        for slot in range(len(idx)):
            out = idx[slot]
            spent_wo = sum(price[i] for i in idx) - price[out]
            best_j, best_v = None, cur_val
            for j in by_pos.get(pos[out], []):
                if j in idx or spent_wo + price[j] > budget + 1e-6:
                    continue
                trial = idx[:slot] + [j] + idx[slot + 1:]
                v = obj.value(trial)
                if v > best_v + 1e-9:
                    best_v, best_j = v, j
            if best_j is not None:
                idx[slot] = best_j
                cur_val = best_v
                improved = True
        if not improved:
            break
    return {"idx": idx, "value": cur_val, "cost": sum(price[i] for i in idx)}
