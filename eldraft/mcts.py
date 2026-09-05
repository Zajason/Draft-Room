"""Monte-Carlo tree search over a live exclusive draft — the AlphaZero-style option.

This is the search half of AlphaZero grafted onto the tooling already built.  The two
learned networks AlphaZero trains by self-play are replaced by things this project can
compute exactly and cheaply, which is the whole reason a trained net is not worth it here:

  * the *value network* becomes the squad valuation itself — any finished roster is scored
    by the game's real rules, so we never have to approximate how good a position is;
  * the *policy network* becomes the value-over-replacement prior from the one-shot
    engine — a strong, analytic hint about which picks are worth searching.

What MCTS adds over the one-shot recommender is the one thing that engine structurally
cannot see: the board changing under you.  By rolling the draft forward against a model of
the other teams, the search values a pick by the squad it leads to *after* rivals have
taken their share — so it will reach for a scarce centre before a run empties the position,
or wait on a deep one, where the greedy pick cannot.

The tree is open-loop: nodes are sequences of *my* picks, and the opponents are re-sampled
on every simulation, so a single tree averages over many ways the board might fall.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .config import RULES
from .draft_sim import (
    DraftState,
    DraftUniverse,
    advance_field,
    leaf_value,
    roster_value,
    vorp_scores,
)


class Node:
    """One of my decision points. Children are keyed by the player index I would take."""
    __slots__ = ("children", "N", "W", "P", "expanded", "cand")

    def __init__(self):
        self.children: Dict[int, "Node"] = {}
        self.N = 0
        self.W = 0.0
        self.P: Dict[int, float] = {}
        self.expanded = False
        self.cand: List[int] = []

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


def _softmax_prior(scores: Dict[int, float], temp: float) -> Dict[int, float]:
    if not scores:
        return {}
    keys = list(scores)
    v = np.array([scores[k] for k in keys], dtype=float)
    v = (v - v.max()) / max(1e-9, temp)
    p = np.exp(v)
    p /= p.sum()
    return {k: float(pi) for k, pi in zip(keys, p)}


class MCTS:
    def __init__(self, uni: DraftUniverse, my_seat: int, n_teams: int,
                 sims: int = 300, c_puct: float = 1.6, top_k: int = 12,
                 opp_temp: float = 0.6, prior_temp: float = 1.2,
                 rollouts: int = 1, seed: int = 0, shortlist: int = 24,
                 dp_prior_depth: int = 1, rules=RULES):
        self.uni = uni
        self.my_seat = my_seat
        self.n_teams = n_teams
        self.sims = sims
        self.c_puct = c_puct
        self.top_k = top_k
        self.opp_temp = opp_temp
        self.prior_temp = prior_temp
        self.rollouts = rollouts
        self.shortlist = shortlist
        self.dp_prior_depth = dp_prior_depth
        self.rng = np.random.default_rng(seed)
        self.rules = rules
        self._value_norm = None  # scale rewards to ~[0,1] so c_puct behaves predictably

    # -- priors ------------------------------------------------------------------------
    def _prior(self, state: DraftState, seat: int, use_dp: bool) -> Dict[int, float]:
        if use_dp:
            sc = vorp_scores(state, seat)
            if sc:
                # VORP can be tiny/negative; shift so every legal pick keeps some weight.
                lo = min(sc.values())
                sc = {k: v - lo + 0.05 for k, v in sc.items()}
                return _softmax_prior(sc, self.prior_temp * (0.15 + 0.85))
        legal = state.legal_actions(seat)
        if legal.size == 0:
            return {}
        fp = {int(i): float(self.uni.fp[i]) for i in legal}
        return _softmax_prior(fp, self.prior_temp)

    def _expand(self, node: Node, state: DraftState, use_dp: bool) -> None:
        prior = self._prior(state, self.my_seat, use_dp)
        if not prior:
            node.expanded = True
            return
        # Keep the top_k most promising actions; renormalise their prior.
        top = sorted(prior, key=prior.get, reverse=True)[: self.top_k]
        z = sum(prior[a] for a in top) or 1.0
        node.P = {a: prior[a] / z for a in top}
        node.cand = top
        node.expanded = True

    # -- selection ---------------------------------------------------------------------
    def _select(self, node: Node, legal: set) -> int:
        avail = [a for a in node.cand if a in legal]
        if not avail:
            return -1
        sqrtN = math.sqrt(max(1, node.N))
        best, best_a = -1e18, avail[0]
        for a in avail:
            child = node.children.get(a)
            q = child.Q if child else 0.0
            u = self.c_puct * node.P.get(a, 1e-3) * sqrtN / (1 + (child.N if child else 0))
            s = q + u
            if s > best:
                best, best_a = s, a
        return best_a

    # -- one simulation ----------------------------------------------------------------
    def _simulate(self, root: Node, root_state: DraftState) -> None:
        state = root_state.clone()
        node = root
        path = [root]
        depth = 0
        while True:
            if state.done:
                value = roster_value(self.uni, state.teams[self.my_seat].picks)
                break
            if not node.expanded:
                if state.candidate_indices(self.my_seat).size == 0:
                    value = roster_value(self.uni, state.teams[self.my_seat].picks)
                    break
                self._expand(node, state, use_dp=(depth <= self.dp_prior_depth))
                value = self._leaf_value(state)
                break
            legal = set(a for a in node.cand if state.feasible(a, self.my_seat))
            a = self._select(node, legal)
            if a < 0:
                value = self._leaf_value(state)
                break
            state.apply(a)                                   # my pick
            advance_field(state, self.my_seat, self.rng, self.opp_temp)
            child = node.children.get(a)
            if child is None:
                child = Node()
                node.children[a] = child
            node = child
            path.append(node)
            depth += 1
        self._backup(path, value)

    def _leaf_value(self, state: DraftState) -> float:
        vals = []
        for _ in range(self.rollouts):
            vals.append(leaf_value(state, self.my_seat, self.rng, self.opp_temp,
                                   shortlist=self.shortlist))
        return float(np.mean(vals))

    def _backup(self, path: List[Node], value: float) -> None:
        if self._value_norm is None:
            self._value_norm = max(1.0, abs(value))
        v = value / self._value_norm
        for node in path:
            node.N += 1
            node.W += v

    # -- public ------------------------------------------------------------------------
    def run(self, root_state: DraftState) -> Dict:
        assert root_state.current_seat == self.my_seat, "root must be my turn"
        root = Node()
        for _ in range(self.sims):
            self._simulate(root, root_state)
        rows = []
        for a in root.cand:
            child = root.children.get(a)
            rows.append({
                "idx": a,
                "code": self.uni.code[a],
                "name": self.uni.name[a],
                "pos": ["G", "F", "C", "H"][self.uni.pos[a]],
                "club": self.uni.club[a],
                "price": float(self.uni.price[a]),
                "fp": float(self.uni.fp[a]),
                "visits": child.N if child else 0,
                "value": (child.Q * (self._value_norm or 1.0)) if child else 0.0,
                "prior": root.P.get(a, 0.0),
            })
        rows.sort(key=lambda r: (-r["visits"], -r["value"]))
        best = rows[0]["idx"] if rows else -1
        return {"best": best, "ranking": rows, "root_visits": root.N}


def mcts_pick(uni: DraftUniverse, state: DraftState, my_seat: int, n_teams: int,
              **kw) -> int:
    return MCTS(uni, my_seat, n_teams, **kw).run(state)["best"]
