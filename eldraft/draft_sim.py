"""A live exclusive draft as a sequential decision problem.

The salary-cap game the rest of the tool solves is a one-shot optimisation: everyone can
own everyone, so the only question is which legal squad scores most.  A *draft* is a
different animal — players leave the board as they are taken, so the value of a pick now
depends on what will still be there at your next turn.  That is the setting where
lookahead can beat the one-shot optimiser, and this module is the environment the MCTS
drafter plays in.

State is deliberately light (integer indices + a boolean availability mask) so a full
draft can be rolled out thousands of times per decision.  Everything expensive — the
projections, prices and the exact squad valuation — is reused from the modelling pipeline;
nothing is re-estimated here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import RULES
from .optimize import _credits_to_units

POS = ("G", "F", "C", "H")
POS_IX = {p: i for i, p in enumerate(POS)}


# ======================================================================================
# Immutable description of the player universe (shared by every simulated draft)
# ======================================================================================
class DraftUniverse:
    """Column arrays for the pool, plus the per-position price ladders feasibility needs."""

    def __init__(self, pool: Sequence[dict], rules=RULES, coach: bool = True):
        self.rules = rules
        self.coach = coach
        self.n = len(pool)
        self.pool = list(pool)
        self.code = [p.get("code") for p in pool]
        self.name = [p.get("name") for p in pool]
        self.club = [p.get("club_name") or p.get("club") for p in pool]
        self.pos = np.array([POS_IX.get(p.get("position", "F"), 1) for p in pool], dtype=np.int8)
        self.fp = np.array([float(p.get("fp") or 0.0) for p in pool], dtype=np.float64)
        self.sigma = np.array([float(p.get("fp_sigma") or 0.0) for p in pool], dtype=np.float64)
        # A real 0.0 (a snake draft with prices zeroed) must be kept, not read as "missing"
        # and back-filled to price_min - otherwise a zero budget makes nothing affordable.
        _pr = [rules.price_min if p.get("price") is None else float(p["price"]) for p in pool]
        self.price = np.array(_pr, dtype=np.float64)
        self.units = np.array([_credits_to_units(x) for x in _pr], dtype=np.int32)
        self.by_code = {c: i for i, c in enumerate(self.code)}

        # Slot template every team must fill, in POS order.
        self.need_template = np.array(
            [rules.slots["G"], rules.slots["F"], rules.slots["C"], 1 if coach else 0], dtype=np.int8
        )
        self.squad_slots = int(self.need_template.sum())

        # Per position, indices sorted cheapest-first — used to price the minimum legal
        # completion of a roster without scanning the whole pool each time.
        self._pos_sorted = [
            np.array(sorted((i for i in range(self.n) if self.pos[i] == pi),
                            key=lambda i: self.units[i]), dtype=np.int32)
            for pi in range(4)
        ]
        self._units_sorted = [self.units[o] for o in self._pos_sorted]

    def min_completion_units(self, avail: np.ndarray, needs: np.ndarray) -> int:
        """Cheapest way to fill `needs` from currently available players, in credit units.

        Returns a large sentinel if some position cannot be filled at all — that roster is
        a dead end and the caller must treat the move that produced it as illegal.  The
        per-position price ladders are pre-sorted, so this is a couple of vectorised slices
        rather than a scan of the whole board.
        """
        total = 0
        for pi in range(4):
            k = int(needs[pi])
            if k <= 0:
                continue
            order = self._pos_sorted[pi]
            live = order[avail[order]]
            if live.size < k:
                return 10 ** 9
            total += int(self._units_sorted[pi][:live.size][:k].sum()) if False else int(self.units[live[:k]].sum())
        return total


# ======================================================================================
# A single team's roster within a draft
# ======================================================================================
@dataclass
class Team:
    seat: int
    picks: List[int] = field(default_factory=list)
    spent_units: int = 0
    needs: np.ndarray = None  # remaining slots by position

    def clone(self) -> "Team":
        t = Team(self.seat, list(self.picks), self.spent_units, self.needs.copy())
        return t


# ======================================================================================
# Draft state
# ======================================================================================
class DraftState:
    """Mutable draft position: rosters, the board, and whose turn it is."""

    def __init__(self, uni: DraftUniverse, n_teams: int, budget_units: int,
                 order: List[int], ptr: int, teams: List[Team], avail: np.ndarray):
        self.uni = uni
        self.n_teams = n_teams
        self.budget_units = budget_units
        self.order = order
        self.ptr = ptr
        self.teams = teams
        self.avail = avail

    def clone(self) -> "DraftState":
        return DraftState(self.uni, self.n_teams, self.budget_units, self.order, self.ptr,
                          [t.clone() for t in self.teams], self.avail.copy())

    @property
    def done(self) -> bool:
        return self.ptr >= len(self.order)

    @property
    def current_seat(self) -> int:
        return self.order[self.ptr]

    # -- legal moves -------------------------------------------------------------------
    def legal_actions(self, seat: Optional[int] = None) -> np.ndarray:
        """Indices this seat may legally take: available, at a position still needed,
        affordable now, and leaving enough budget to complete the rest of the roster."""
        seat = self.current_seat if seat is None else seat
        t = self.teams[seat]
        remaining = self.budget_units - t.spent_units
        # candidate = available and at a needed position
        need_pos = np.where(t.needs > 0)[0]
        if need_pos.size == 0:
            return np.array([], dtype=np.int32)
        mask = self.avail & np.isin(self.uni.pos, need_pos)
        cand = np.where(mask)[0]
        if cand.size == 0:
            return cand
        out = []
        for i in cand:
            cost = int(self.uni.units[i])
            if cost > remaining:
                continue
            # after taking i, can the remaining slots still be filled within budget?
            needs2 = t.needs.copy()
            needs2[self.uni.pos[i]] -= 1
            if int(needs2.sum()) == 0:
                out.append(i)
                continue
            avail2 = self.avail
            avail2[i] = False
            floor = self.uni.min_completion_units(avail2, needs2)
            avail2[i] = True
            if floor <= remaining - cost:
                out.append(i)
        return np.array(out, dtype=np.int32)

    def feasible(self, i: int, seat: Optional[int] = None) -> bool:
        """Can this seat legally take player i right now (available, needed, affordable,
        and still able to complete a legal roster afterwards)?"""
        seat = self.current_seat if seat is None else seat
        if not self.avail[i]:
            return False
        pos = self.uni.pos[i]
        t = self.teams[seat]
        if t.needs[pos] <= 0:
            return False
        remaining = self.budget_units - t.spent_units
        cost = int(self.uni.units[i])
        if cost > remaining:
            return False
        needs2 = t.needs.copy()
        needs2[pos] -= 1
        if int(needs2.sum()) == 0:
            return True
        self.avail[i] = False
        floor = self.uni.min_completion_units(self.avail, needs2)
        self.avail[i] = True
        return floor <= remaining - cost

    def candidate_indices(self, seat: Optional[int] = None) -> np.ndarray:
        """Available players at a position this seat still needs and can afford — cheap,
        skips the completion check (callers that need strict legality use `feasible`)."""
        seat = self.current_seat if seat is None else seat
        t = self.teams[seat]
        remaining = self.budget_units - t.spent_units
        need_pos = np.where(t.needs > 0)[0]
        if need_pos.size == 0:
            return np.array([], dtype=np.int32)
        mask = self.avail & np.isin(self.uni.pos, need_pos) & (self.uni.units <= remaining)
        return np.where(mask)[0]

    def apply(self, i: int) -> None:
        """Advance the draft by team `current_seat` taking player index i (in place)."""
        seat = self.current_seat
        t = self.teams[seat]
        t.picks.append(int(i))
        t.spent_units += int(self.uni.units[i])
        t.needs[self.uni.pos[i]] -= 1
        self.avail[i] = False
        self.ptr += 1


# ======================================================================================
# Draft construction
# ======================================================================================
def snake_order(n_teams: int, rounds: int) -> List[int]:
    order = []
    for r in range(rounds):
        seats = range(n_teams) if r % 2 == 0 else range(n_teams - 1, -1, -1)
        order.extend(seats)
    return order


def new_draft(uni: DraftUniverse, n_teams: int, taken: Optional[Sequence[int]] = None,
              mine: Optional[Dict[int, List[int]]] = None, my_seat: int = 0,
              budget: float = None) -> DraftState:
    """Fresh draft state.

    `taken` are players already off the board with no owner recorded (rivals in a live
    draft you are only half-tracking); `mine` maps seat -> already-owned indices so a
    partially-completed draft can be resumed.
    """
    budget = RULES.budget if budget is None else budget
    bu = _credits_to_units(budget)
    rounds = uni.squad_slots
    order = snake_order(n_teams, rounds)
    teams = [Team(s, [], 0, uni.need_template.copy()) for s in range(n_teams)]
    avail = np.ones(uni.n, dtype=bool)
    for i in (taken or []):
        avail[i] = False
    picks_made = [0] * n_teams
    for seat, idxs in (mine or {}).items():
        for i in idxs:
            teams[seat].picks.append(i)
            teams[seat].spent_units += int(uni.units[i])
            teams[seat].needs[uni.pos[i]] -= 1
            avail[i] = False
            picks_made[seat] += 1
    # advance ptr past picks already made, honouring snake order
    ptr = 0
    counts = [0] * n_teams
    target = list(picks_made)
    while ptr < len(order) and counts[order[ptr]] < target[order[ptr]]:
        counts[order[ptr]] += 1
        ptr += 1
    return DraftState(uni, n_teams, bu, order, ptr, teams, avail)


# ======================================================================================
# Valuation
# ======================================================================================
def roster_value(uni: DraftUniverse, picks: Sequence[int],
                 scores: Optional[np.ndarray] = None, rules=RULES) -> float:
    """Structured fantasy value of a (possibly partial) roster under the game's rules:
    the best six outfielders score full, the rest half, the top scorer is captained, and
    the coach is added flat.  `scores` overrides the point projection (for simulation)."""
    fp = uni.fp if scores is None else scores
    out, coach = [], 0.0
    for i in picks:
        if uni.pos[i] == POS_IX["H"]:
            coach += fp[i]
        else:
            out.append(fp[i])
    out.sort(reverse=True)
    k = rules.full_credit_slots
    v = sum(out[:k]) + rules.bench_multiplier * sum(out[k:])
    if out:
        v += (rules.captain_multiplier - 1.0) * out[0]
    return v + coach


# ======================================================================================
# Policies
# ======================================================================================
def heuristic_pick(state: DraftState, seat: int, rng: np.random.Generator,
                   temp: float = 0.6) -> int:
    """A fast, DP-free policy for opponents and rollouts.

    Models a competent-but-not-optimal rival: take a good player who fills a slot you
    still need and can afford, with `temp` controlling how sharply he chases the best
    projection.  Feasibility is already guaranteed by legal_actions, so a rival can never
    strand himself unable to complete a legal roster.
    """
    cand = state.candidate_indices(seat)
    if cand.size == 0:
        return -1
    fp = state.uni.fp[cand]
    if temp <= 1e-6:
        order = cand[np.argsort(-fp)]
    else:
        z = (fp - fp.max()) / max(1e-6, temp * (fp.std() + 1e-6))
        p = np.exp(z); p /= p.sum()
        # Draw a short priority order, then take the first pick that keeps the roster
        # completable — cheaper than pricing every candidate up front.
        order = rng.choice(cand, size=min(cand.size, 8), replace=False, p=p)
    for i in order:
        if state.feasible(int(i), seat):
            return int(i)
    # Fall back to a strict scan only if the sampled shortlist was all dead ends.
    for i in cand[np.argsort(state.uni.units[cand])]:
        if state.feasible(int(i), seat):
            return int(i)
    return -1


def advance_field(state: DraftState, my_seat: int, rng: np.random.Generator,
                  temp: float = 0.6) -> None:
    """Let every seat other than `my_seat` take its turn until the draft returns to me."""
    while not state.done and state.current_seat != my_seat:
        i = heuristic_pick(state, state.current_seat, rng, temp)
        if i < 0:
            state.ptr += 1  # a stuck seat forfeits its pick rather than hanging the sim
            continue
        state.apply(i)


def vona_pick(state: DraftState, seat: int) -> int:
    """Value-over-next-available: the fast, strong drafting heuristic.

    A pick is worth its projection minus the projection of the player who would replace him
    if you waited — the r-th best still on the board at his position, where r is how many
    more of that position the whole league still needs.  This encodes positional scarcity
    without a full optimisation, so it is a far better completion policy than raw
    projection and cheap enough to run inside every rollout.
    """
    uni = state.uni
    cand = state.candidate_indices(seat)
    if cand.size == 0:
        return -1
    # replacement level per position = fp of the next player likely gone before your turn.
    league_need = np.zeros(4, dtype=int)
    for t in state.teams:
        league_need += t.needs
    # replacement level per position, computed once
    repl = np.zeros(4)
    for pi in range(4):
        pool_pi = np.where(state.avail & (uni.pos == pi))[0]
        if pool_pi.size:
            fps = np.sort(uni.fp[pool_pi])[::-1]
            r = min(len(fps) - 1, max(1, int(league_need[pi])))
            repl[pi] = fps[r]
    score = uni.fp[cand] - repl[uni.pos[cand]]
    for i in cand[np.argsort(-score)]:
        if state.feasible(int(i), seat):
            return int(i)
    return -1


def rollout_value(state: DraftState, my_seat: int, rng: np.random.Generator,
                  temp: float = 0.6, my_policy: str = "vona") -> float:
    """Play the rest of the draft out and score my finished roster.

    Opponents use the fast heuristic (they are the field, not optimisers); I complete my
    own roster with the stronger VONA policy, so a leaf is valued by how good a team I can
    realistically still build from here — not by how a coin-flip heuristic would fill it.
    """
    while not state.done:
        seat = state.current_seat
        if seat == my_seat and my_policy == "vona":
            i = vona_pick(state, seat)
        else:
            i = heuristic_pick(state, seat, rng, temp)
        if i < 0:
            state.ptr += 1
            continue
        state.apply(i)
    return roster_value(state.uni, state.teams[my_seat].picks)


# ======================================================================================
# VORP prior (the analytic policy shared with the one-shot engine)
# ======================================================================================
def vorp_scores(state: DraftState, seat: int, budget: float = None) -> Dict[int, float]:
    """Value-over-replacement of every legal pick, given this seat's current roster.

    Reuses the exact squad dynamic program: build the best full squad this seat can still
    assemble with and without each candidate, and take the difference.  This is the same
    quantity the one-shot recommender ranks by, evaluated at a mid-draft state, so a
    greedy agent built on it *is* the current tool playing the draft pick by pick.
    """
    from .optimize import Slots, SquadDP
    uni = state.uni
    budget = RULES.budget if budget is None else budget
    t = state.teams[seat]
    legal = set(int(i) for i in state.legal_actions(seat))
    if not legal:
        return {}

    # Sub-pool for the DP: my committed picks (forced in) plus everyone still available.
    sub, gindex, forced = [], [], set()
    for i in t.picks:
        d = dict(uni.pool[i]); d["mandatory"] = True
        forced.add(len(sub)); sub.append(d); gindex.append(i)
    avail_idx = np.where(state.avail)[0]
    for i in avail_idx:
        sub.append(uni.pool[int(i)]); gindex.append(int(i))

    dp = SquadDP(sub, budget, Slots(coach=uni.coach), score_key="fp")
    mv = dp.marginal_values()
    out = {}
    for local_i, (w, wo) in mv.items():
        g = gindex[local_i]
        if g in legal and np.isfinite(w) and np.isfinite(wo):
            out[g] = w - wo
    return out


def greedy_pick(state: DraftState, seat: int, budget: float = None) -> int:
    """The current tool's pick: the highest-VORP legal player at this state."""
    sc = vorp_scores(state, seat, budget)
    if not sc:
        legal = state.legal_actions(seat)
        return int(legal[int(np.argmax(state.uni.fp[legal]))]) if legal.size else -1
    return max(sc, key=sc.get)


def dp_complete_value(uni: DraftUniverse, my_picks: List[int], avail: np.ndarray,
                      budget: float = None, shortlist: int = 24) -> float:
    """Value of my best legal completion, given the board `avail`.

    Only the top few available players per position can plausibly be in an optimal squad,
    so the dynamic program runs over that shortlist plus my committed picks — near-exact
    and several times faster than the full board.
    """
    from .optimize import Slots, SquadDP
    budget = RULES.budget if budget is None else budget
    sub, forced = [], set()
    for i in my_picks:
        d = dict(uni.pool[i]); d["mandatory"] = True
        forced.add(len(sub)); sub.append(d)
    for pi in range(4):
        idxs = np.where(avail & (uni.pos == pi))[0]
        if idxs.size:
            top = idxs[np.argsort(-uni.fp[idxs])[:shortlist]]
            for i in top:
                sub.append(uni.pool[int(i)])
    val, _ = SquadDP(sub, budget, Slots(coach=uni.coach)).solve()
    return val


def leaf_value(state: DraftState, my_seat: int, rng: np.random.Generator,
               temp: float = 0.6, shortlist: int = 24) -> float:
    """Strong, honest leaf value: let the rivals deplete the board to the end of the
    draft, then solve my optimal completion of what is left.

    This is what lifts the search above its own rollout policy — the value of a pick is
    the best team I can *optimally* still build after the board has realistically emptied,
    not the team a cheap heuristic happens to assemble.  Opponent turns are simulated;
    my turns are skipped (I do not pre-commit picks), so the completion is free to choose.
    """
    s = state.clone()
    while not s.done:
        if s.current_seat == my_seat:
            s.ptr += 1                      # leave my future picks open for the DP
            continue
        i = heuristic_pick(s, s.current_seat, rng, temp)
        if i < 0:
            s.ptr += 1
            continue
        s.apply(i)
    val = dp_complete_value(state.uni, list(state.teams[my_seat].picks), s.avail,
                            budget=state.budget_units / 2.0, shortlist=shortlist)
    if not np.isfinite(val):
        # depletion left me unable to complete legally; fall back to a heuristic finish
        return rollout_value(state.clone(), my_seat, rng, temp)
    return val


def remaining_snake(needs_per_team: List[np.ndarray], my_seat: int, n_teams: int) -> List[int]:
    """A snake order over the picks each team still owes, starting with `my_seat`."""
    order: List[int] = []
    rem = [int(n.sum()) for n in needs_per_team]
    seq = [my_seat] + [s for s in range(n_teams) if s != my_seat]
    rnd = 0
    while any(r > 0 for r in rem) and rnd < 400:
        ring = seq if rnd % 2 == 0 else list(reversed(seq))
        for s in ring:
            if rem[s] > 0:
                order.append(s)
                rem[s] -= 1
        rnd += 1
    return order


def state_at_my_turn(uni: DraftUniverse, n_teams: int, my_seat: int,
                     mine_idx: Sequence[int], taken_idx: Sequence[int],
                     budget: float = None) -> DraftState:
    """Reconstruct a mid-draft state where it is `my_seat`'s turn: my picks are mine,
    rivals' picks are distributed across the other seats, and the remaining order snakes
    from me — matching how the in-browser MCTS models a live board."""
    budget = RULES.budget if budget is None else budget
    bu = _credits_to_units(budget)
    teams = [Team(s, [], 0, uni.need_template.copy()) for s in range(n_teams)]
    avail = np.ones(uni.n, dtype=bool)
    for i in mine_idx:
        teams[my_seat].picks.append(int(i))
        teams[my_seat].spent_units += int(uni.units[i])
        teams[my_seat].needs[uni.pos[i]] -= 1
        avail[i] = False
    opp = [s for s in range(n_teams) if s != my_seat]
    k = 0
    for i in taken_idx:
        if not avail[i]:
            continue
        pos = uni.pos[i]
        placed = False
        for t in range(len(opp)):
            s = opp[(k + t) % len(opp)]
            if teams[s].needs[pos] > 0:
                teams[s].picks.append(int(i))
                teams[s].spent_units += int(uni.units[i])
                teams[s].needs[pos] -= 1
                avail[i] = False
                k = (k + t + 1) % len(opp)
                placed = True
                break
        if not placed:
            avail[i] = False        # nobody needs that position — just off the board
    order = remaining_snake([t.needs for t in teams], my_seat, n_teams)
    return DraftState(uni, n_teams, bu, order, 0, teams, avail)
