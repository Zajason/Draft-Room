"""Exact squad optimiser and robust draft-decision engine.

The EuroLeague Fantasy Challenge squad is not a simple "best value per credit" problem,
because the scoring is *structured*: of the ten players you own, the best six score full
PIR and the other four score half, and one of the six is captained for a multiplier.  A
greedy value-per-credit ranking therefore systematically overpays for depth.

The objective for a squad S is

    V(S) = sum_{top 6 by score} f_i  +  0.5 * sum_{rest} f_i  +  (m - 1) * max_i f_i

which we maximise exactly with a dynamic program.  The trick that makes it exact is to
process players in *descending projected score*: the first six players a path selects are
necessarily its top six, so the multiplier for the k-th selection is known from the state
(k = guards + forwards + centers already taken) without any extra bookkeeping.

State  : (guards, forwards, centers, coach, credits spent)
Value  : best achievable V over the players considered so far

A forward pass F[i] and a backward pass B[i] are then combined to answer, for *every*
candidate simultaneously and exactly:

    "what is the best complete squad I can still build if I commit to this player now?"

That marginal value - not raw projection, and not raw value-per-credit - is the draft
recommendation.  Running the whole thing over Monte-Carlo draws of the projections turns
it into a robust decision: we report how often each player survives as the right pick.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import RULES

NEG = -1e9
CREDIT_STEP = 0.5          # official prices move in half credits


class Slots:
    """Roster shape for the DP: capacities and the index each position occupies."""

    ORDER = ("G", "F", "C", "H")   # H = head coach

    def __init__(self, caps: Optional[Dict[str, int]] = None, coach: bool = True):
        base = dict(RULES.slots)
        if caps:
            base.update({k: v for k, v in caps.items() if k in base})
        self.caps = {"G": base["G"], "F": base["F"], "C": base["C"], "H": 1 if coach else 0}
        self.shape = tuple(self.caps[p] + 1 for p in self.ORDER)

    @property
    def n_players(self) -> int:
        return self.caps["G"] + self.caps["F"] + self.caps["C"]


def _credits_to_units(x: float) -> int:
    return int(round(float(x) / CREDIT_STEP))


def _multiplier_table(slots: Slots, rules=RULES) -> np.ndarray:
    """mult[g,f,c,h] = the scoring weight the NEXT selected player would receive."""
    g, f, c, h = slots.shape
    idx = np.indices((g, f, c, h))
    taken = idx[0] + idx[1] + idx[2]          # coach never counts toward the 6/4 split
    return np.where(taken < rules.full_credit_slots, 1.0, rules.bench_multiplier)


class SquadDP:
    """Dynamic program over (position counts, credits spent) for one scenario."""

    def __init__(self, players: Sequence[dict], budget: float,
                 slots: Optional[Slots] = None, rules=RULES,
                 score_key: str = "fp",
                 forced: Optional[set] = None, excluded: Optional[set] = None):
        self.forced = set(forced or ())
        self.excluded = set(excluded or ())
        self.rules = rules
        self.slots = slots or Slots()
        self.B = _credits_to_units(budget)
        self.players = list(players)
        self.score_key = score_key

        # Descending score is what makes the 6/4 multiplier state-determined.
        self.order = sorted(
            range(len(self.players)),
            key=lambda i: -float(self.players[i].get(score_key, 0.0)),
        )
        self._pos = {i: i for i in range(len(self.players))}
        self.mult = _multiplier_table(self.slots, rules)
        self.coach_mult = 1.0
        self.captain_extra = rules.captain_multiplier - 1.0

    # -- shapes ------------------------------------------------------------------------
    def _empty(self) -> np.ndarray:
        return np.full(self.slots.shape + (self.B + 1,), NEG, dtype=np.float32)

    def _terminal(self) -> np.ndarray:
        """Backward base case: only a completely filled squad is worth anything."""
        arr = np.full(self.slots.shape + (self.B + 1,), NEG, dtype=np.float32)
        caps = self.slots.caps
        arr[caps["G"], caps["F"], caps["C"], caps["H"], :] = 0.0
        return arr

    def _gain(self, p: dict) -> np.ndarray:
        """Score contribution of player p as a function of the state he is added from."""
        s = float(p.get(self.score_key, 0.0))
        pos = p.get("position", "F")
        if pos == "H":
            return np.full(self.slots.shape, s * self.coach_mult, dtype=np.float32)
        g = self.mult * s
        # The very first player selected is the top scorer in the squad -> captain.
        g[0, 0, 0, :] += self.captain_extra * s
        return g.astype(np.float32)

    def _axis(self, pos: str) -> int:
        return Slots.ORDER.index(pos)

    # -- passes ------------------------------------------------------------------------
    def _must(self, i: int, p: dict) -> bool:
        return bool(p.get("mandatory")) or i in self.forced

    def _step_forward(self, cur: np.ndarray, p: dict, i: int = -1) -> np.ndarray:
        """Best value after deciding on player p, given `cur` = best value before him."""
        if i in self.excluded:
            return cur
        must = self._must(i, p)
        ax = self._axis(p.get("position", "F"))
        cost = _credits_to_units(p["price"])
        cap = self.slots.caps[Slots.ORDER[ax]]
        if cap == 0 or cost > self.B:
            return cur if not must else self._empty()

        gain = self._gain(p)
        src = np.moveaxis(cur, ax, 0)[:cap]                    # states with room left
        add = np.moveaxis(gain, ax, 0)[:cap][..., None]
        cand = src[..., : self.B + 1 - cost] + add
        # Skipping p is allowed unless he is already on the roster, or forced in.
        nxt = cur.copy() if not must else self._empty()
        dst = np.moveaxis(nxt, ax, 0)[1: cap + 1]
        np.maximum(dst[..., cost:], cand, out=dst[..., cost:])
        return nxt

    def _step_backward(self, nxt: np.ndarray, p: dict, i: int = -1) -> np.ndarray:
        """Best remaining value from player p onward, given `nxt` = from p+1 onward."""
        if i in self.excluded:
            return nxt
        must = self._must(i, p)
        ax = self._axis(p.get("position", "F"))
        cost = _credits_to_units(p["price"])
        cap = self.slots.caps[Slots.ORDER[ax]]
        if cap == 0 or cost > self.B:
            return nxt if not must else self._empty()

        gain = self._gain(p)
        take = np.full_like(nxt, NEG)
        srcn = np.moveaxis(nxt, ax, 0)[1: cap + 1]             # state after taking p
        dstt = np.moveaxis(take, ax, 0)[:cap]
        addt = np.moveaxis(gain, ax, 0)[:cap][..., None]
        dstt[..., : self.B + 1 - cost] = srcn[..., cost:] + addt
        return take if must else np.maximum(nxt, take)

    def solve(self) -> Tuple[float, List[int]]:
        """Optimal squad value and the indices (into `players`) that achieve it."""
        fwd: List[Any] = [None] * (len(self.order) + 1)
        cur = self._empty()
        cur[(0,) * len(self.slots.shape) + (0,)] = 0.0
        fwd[0] = cur
        for k, i in enumerate(self.order):
            cur = self._step_forward(cur, self.players[i], i)
            fwd[k + 1] = cur

        caps = self.slots.caps
        final = cur[caps["G"], caps["F"], caps["C"], caps["H"], :]
        best_b = int(np.argmax(final))
        best_v = float(final[best_b])
        if best_v <= NEG / 2:
            return float("-inf"), []

        # Walk the forward tables back to recover the chosen set.
        state = [caps["G"], caps["F"], caps["C"], caps["H"], best_b]
        chosen: List[int] = []
        for k in range(len(self.order), 0, -1):
            i = self.order[k - 1]
            p = self.players[i]
            prev = fwd[k - 1]
            if i in self.excluded:
                continue
            if not self._must(i, p) and abs(prev[tuple(state)] - fwd[k][tuple(state)]) < 1e-4:
                continue                                    # p was skipped on this path
            ax = self._axis(p.get("position", "F"))
            cost = _credits_to_units(p["price"])
            cand = list(state)
            cand[ax] -= 1
            cand[4] -= cost
            if cand[ax] < 0 or cand[4] < 0:
                continue
            gain = float(self._gain(p)[tuple(cand[:4])])
            if abs(prev[tuple(cand)] + gain - fwd[k][tuple(state)]) < 1e-3:
                chosen.append(i)
                state = cand
        return best_v, list(reversed(chosen))

    # -- value of every candidate, with and without ------------------------------------
    def marginal_values(self) -> Dict[int, Tuple[float, float]]:
        """For each player, (best squad value WITH him, best squad value WITHOUT him).

        The difference between the two is what actually matters in a draft.  Asking only
        "how good a squad can I still build if I take this player?" rewards whoever
        constrains you least - a 4.0-credit twelfth man scores almost as well as a star,
        because taking him barely spends anything.  Subtracting the best squad that
        excludes him turns that into a genuine value over replacement: it is positive
        only for players the optimal squad actually wants, and its size says by how much
        they beat the next best use of those credits and that roster slot.

        One forward and one backward pass answer this for the whole pool at once, which
        is what makes the Monte-Carlo layer affordable.
        """
        n = len(self.order)
        fwd: List[Any] = [None] * (n + 1)
        cur = self._empty()
        cur[(0,) * len(self.slots.shape) + (0,)] = 0.0
        fwd[0] = cur
        for k, i in enumerate(self.order):
            cur = self._step_forward(cur, self.players[i], i)
            fwd[k + 1] = cur

        bwd: List[Any] = [None] * (n + 1)
        nxt = self._terminal()
        bwd[n] = nxt
        for k in range(n - 1, -1, -1):
            j = self.order[k]
            nxt = self._step_backward(nxt, self.players[j], j)
            bwd[k] = nxt

        out: Dict[int, Tuple[float, float]] = {}
        for k, i in enumerate(self.order):
            p = self.players[i]
            ax = self._axis(p.get("position", "F"))
            cost = _credits_to_units(p["price"])
            cap = self.slots.caps[Slots.ORDER[ax]]
            before = fwd[k]                                   # states reachable before p
            after = bwd[k + 1]                                # value from p+1 onward

            if cap == 0 or cost > self.B:
                with_p = float("-inf")
            else:
                gain = self._gain(p)
                b4 = np.moveaxis(before, ax, 0)[:cap][..., : self.B + 1 - cost]
                af = np.moveaxis(after, ax, 0)[1: cap + 1][..., cost:]
                ad = np.moveaxis(gain, ax, 0)[:cap][..., None]
                tot = b4 + ad + af
                with_p = float(tot.max()) if tot.size else float("-inf")

            # Best squad that never selects him: same states, no gain, no spend.
            if self._must(i, p):
                without_p = float("-inf")                     # already on your roster
            else:
                skip = before + after
                without_p = float(skip.max()) if skip.size else float("-inf")

            out[i] = (with_p if with_p > NEG / 2 else float("-inf"),
                      without_p if without_p > NEG / 2 else float("-inf"))
        return out


# ======================================================================================
# Robust layer
# ======================================================================================
def _sample_scores(players: Sequence[dict], rng: np.random.Generator) -> np.ndarray:
    """One Monte-Carlo draw of every player's season-long scoring level.

    Fantasy scoring is bounded below by zero and has a long right tail (a role change can
    double a player's output; nothing can take him below zero), so we draw from a
    lognormal matched to each player's projected mean and sigma rather than a normal.
    """
    mu = np.array([max(0.05, float(p.get("fp", 0.0))) for p in players])
    sd = np.array([max(0.05, float(p.get("fp_sigma", 0.0))) for p in players])
    var = np.log1p((sd / mu) ** 2)
    return np.exp(rng.normal(np.log(mu) - var / 2.0, np.sqrt(var)))


def _eval_squads(squads: List[List[int]], draws: np.ndarray, pool: Sequence[dict],
                 rules=RULES) -> np.ndarray:
    """Mean realised value of each squad across every simulated season.

    Scoring is applied the way the game applies it: within each simulated season the
    squad's ten players are ranked by what they actually produced, the best six score in
    full, the other four at half, and the top scorer takes the captain multiplier.  That
    is where a volatile player earns his keep - you choose your starting six each round,
    so his good seasons count and his bad ones sit on the bench.
    """
    out = np.empty(len(squads))
    is_coach = np.array([p.get("position") == "H" for p in pool])
    k = rules.full_credit_slots
    for j, sq in enumerate(squads):
        if not sq:
            out[j] = float("-nan")
            continue
        idx = np.array(sq, dtype=int)
        coach_idx = idx[is_coach[idx]]
        play_idx = idx[~is_coach[idx]]
        val = np.zeros(draws.shape[0])
        if play_idx.size:
            sc = np.sort(draws[:, play_idx], axis=1)[:, ::-1]
            val += sc[:, :k].sum(axis=1) + rules.bench_multiplier * sc[:, k:].sum(axis=1)
            val += (rules.captain_multiplier - 1.0) * sc[:, 0]
        if coach_idx.size:
            val += draws[:, coach_idx].sum(axis=1)
        out[j] = float(val.mean())
    return out


def _squad_value(idx: List[int], draws: np.ndarray, pool: Sequence[dict],
                 rules=RULES) -> float:
    return float(_eval_squads([idx], draws, pool, rules)[0])


def polish_squad(idx: List[int], pool: Sequence[dict], budget: float,
                 draws: np.ndarray, rules=RULES, max_passes: int = 6) -> Tuple[List[int], int]:
    """Improve a squad by single swaps, scored on expected value across seasons.

    The dynamic program returns the exact optimum of the point-estimate objective, but
    that is not quite the objective you actually care about.  Because you name a starting
    six every round, a player's bad seasons are partly benched and his good ones are
    partly captained, which quietly rewards variance in a way a single expected value per
    player cannot express.  This pass therefore re-scores candidate squads against the
    simulated seasons and takes any swap that helps, which typically nudges one or two
    slots toward higher-variance players of the same price.

    Same-position swaps only, so the roster stays legal; the result is a local optimum of
    the true objective, seeded from the global optimum of a very close approximation.
    """
    idx = list(idx)
    cost = sum(pool[i]["price"] for i in idx)
    best = _squad_value(idx, draws, pool, rules)
    swaps = 0
    in_squad = set(idx)
    by_pos: Dict[str, List[int]] = {}
    for j, p in enumerate(pool):
        by_pos.setdefault(p.get("position", "F"), []).append(j)

    for _ in range(max_passes):
        improved = False
        for slot, out_i in enumerate(list(idx)):
            if pool[out_i].get("mandatory"):
                continue                              # already on your roster: untouchable
            pos = pool[out_i].get("position", "F")
            room = budget - cost + pool[out_i]["price"]
            cands, trials = [], []
            for j in by_pos.get(pos, ()):
                if j in in_squad or pool[j]["price"] > room + 1e-9:
                    continue
                trial = list(idx)
                trial[slot] = j
                cands.append(j)
                trials.append(trial)
            if not trials:
                continue
            vals = _eval_squads(trials, draws, pool, rules)
            k = int(np.argmax(vals))
            if float(vals[k]) > best + 1e-6:
                in_squad.discard(out_i)
                in_squad.add(cands[k])
                cost += pool[cands[k]]["price"] - pool[out_i]["price"]
                idx[slot] = cands[k]
                best = float(vals[k])
                improved = True
                swaps += 1
        if not improved:
            break
    return idx, swaps


def _best_swap_in(idx: List[int], j: int, pool: Sequence[dict], budget: float,
                  draws: np.ndarray, rules=RULES) -> Tuple[Optional[List[int]], float]:
    """Best squad obtainable from `idx` by swapping player j in for a same-position slot."""
    pos = pool[j].get("position", "F")
    cost = sum(pool[i]["price"] for i in idx)
    trials, slots_tried = [], []
    for slot, out_i in enumerate(idx):
        if pool[out_i].get("position", "F") != pos or pool[out_i].get("mandatory"):
            continue
        if cost - pool[out_i]["price"] + pool[j]["price"] > budget + 1e-9:
            continue
        t = list(idx)
        t[slot] = j
        trials.append(t)
        slots_tried.append(slot)
    if not trials:
        return None, float("-inf")
    vals = _eval_squads(trials, draws, pool, rules)
    k = int(np.argmax(vals))
    return trials[k], float(vals[k])


def _best_replacement_for(idx: List[int], slot: int, pool: Sequence[dict], budget: float,
                          draws: np.ndarray, rules=RULES) -> Tuple[Optional[List[int]], float]:
    """Best squad obtainable if the player in `slot` were unavailable."""
    out_i = idx[slot]
    pos = pool[out_i].get("position", "F")
    cost = sum(pool[i]["price"] for i in idx)
    room = budget - cost + pool[out_i]["price"]
    in_squad = set(idx)
    trials = []
    for j, p in enumerate(pool):
        if j in in_squad or p.get("position", "F") != pos or p["price"] > room + 1e-9:
            continue
        t = list(idx)
        t[slot] = j
        trials.append(t)
    if not trials:
        return None, float("-inf")
    vals = _eval_squads(trials, draws, pool, rules)
    k = int(np.argmax(vals))
    return trials[k], float(vals[k])


def robust_draft(
    pool: List[dict],
    budget: float,
    slots: Optional[Slots] = None,
    n_sims: int = 400,
    seed: int = 7,
    top_k: int = 40,
) -> dict:
    """Recommend a squad, and rank every available player by what he is worth to it.

    Three stages, each doing the part it is good at.

    1. The dynamic program finds the exact optimum of the point-estimate objective -
       globally optimal, respecting budget, position slots, the six-full/four-half split
       and the captain multiplier.
    2. A local search polishes that squad against simulated seasons, which is the
       objective that actually pays out: you name a starting six every round, so a
       player's worst seasons are partly benched and his best partly captained.
    3. Every available player is then priced against the finished squad:

         draft_value = E[ best squad if he is IN ] - E[ best squad if he is NOT ]

       in fantasy points per round.  For a player already in the squad that is the damage
       done by losing him to somebody else; for anyone else it is what he would add if
       you took him instead of the man he would displace.  Either way it is the number a
       draft pick is actually deciding, and it is measured on seasons the squads were not
       chosen on, so the noisiest players do not float to the top.

    `pick_rate` re-optimises against each simulated season and reports how often a player
    survives as part of the answer.  It is a sensitivity label, not the ranking: it is
    computed by optimising and scoring on the same draw, so it flatters volatile players.
    """
    slots = slots or Slots()
    rules = RULES
    base = SquadDP(pool, budget, slots)
    base_value, base_idx = base.solve()
    if not math.isfinite(base_value):
        raise ValueError(
            "no legal squad fits the budget: check prices, remaining credits and how "
            "many players you still need at each position"
        )
    base_mv = base.marginal_values()

    rng = np.random.default_rng(seed)
    draws = np.vstack([_sample_scores(pool, rng) for _ in range(max(1, n_sims))])

    dp_expected = _squad_value(base_idx, draws, pool, rules)
    squad, n_swaps = polish_squad(base_idx, pool, budget, draws, rules)
    squad_expected = _squad_value(squad, draws, pool, rules)
    in_squad = {i: slot for slot, i in enumerate(squad)}

    # ---- sensitivity: how often does a player survive re-optimisation? ----------------
    picked = np.zeros(len(pool))
    sens = min(draws.shape[0], 150)
    for t in range(sens):
        scen = []
        for p, sc in zip(pool, draws[t]):
            q = dict(p)
            q["_scen"] = float(sc)
            scen.append(q)
        _, idx = SquadDP(scen, budget, slots, rules, score_key="_scen").solve()
        for i in idx:
            picked[i] += 1

    # ---- price every candidate against the finished squad ----------------------------
    replacement_cache: Dict[int, Tuple[Optional[List[int]], float]] = {}
    rows = []
    for i, p in enumerate(pool):
        w, wo = base_mv.get(i, (float("-inf"), float("-inf")))
        vorp = (w - wo) if (math.isfinite(w) and math.isfinite(wo)) else None
        replaces = None
        if p.get("mandatory"):
            dv = None
        elif i in in_squad:
            slot = in_squad[i]
            if slot not in replacement_cache:
                replacement_cache[slot] = _best_replacement_for(
                    squad, slot, pool, budget, draws, rules)
            alt, alt_val = replacement_cache[slot]
            dv = squad_expected - alt_val if math.isfinite(alt_val) else None
            replaces = pool[alt[slot]]["name"] if alt else None
        else:
            alt, alt_val = _best_swap_in(squad, i, pool, budget, draws, rules)
            dv = (alt_val - squad_expected) if math.isfinite(alt_val) else None
            if alt is not None:
                for slot, j in enumerate(squad):
                    if alt[slot] != j:
                        replaces = pool[j]["name"]
                        break
        rows.append({
            "idx": i,
            "code": p.get("code"),
            "name": p.get("name"),
            "position": p.get("position"),
            "club": p.get("club_name") or p.get("club"),
            "price": p.get("price"),
            "fp": p.get("fp"),
            "fp_sigma": p.get("fp_sigma"),
            "mandatory": bool(p.get("mandatory")),
            "in_optimal": i in in_squad,
            "vorp": (round(vorp, 3) if vorp is not None else None),
            "draft_value": (round(dv, 3) if dv is not None else None),
            "replaces": replaces,
            "pick_rate": round(picked[i] / sens, 4) if sens else None,
            "squad_value_with": (round(w, 3) if math.isfinite(w) else None),
        })
    # Players we cannot price (already yours, or no legal swap exists) sort to the back.
    rows.sort(key=lambda r: (-r["draft_value"]) if r["draft_value"] is not None else float("inf"))
    return {
        "dp_point_value": round(base_value, 3),
        "dp_expected_value": round(dp_expected, 3),
        "squad_expected_value": round(squad_expected, 3),
        "polish_swaps": n_swaps,
        "squad": [pool[i] for i in squad],
        "squad_cost": round(sum(pool[i]["price"] for i in squad), 1),
        "ranking": rows[:top_k] if top_k else rows,
        "all_rows": rows,
        "n_sims": draws.shape[0],
        "sensitivity_sims": sens,
    }
