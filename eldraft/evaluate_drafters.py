"""Does lookahead actually help? A head-to-head tournament of draft strategies.

Each simulated draft seats one MCTS drafter and one greedy (current-engine) drafter at
random positions, with the remaining seats filled by a heuristic "field".  Both
contestants face the same board in the same draft, so the comparison is paired: the only
thing that differs is how they pick.  Final rosters are scored the same way the dashboard
scores squads — expected fantasy points per round across simulated seasons — and we report
how often, and by how much, MCTS ends up with the better team.

If MCTS does not beat greedy here, it does not belong in the product, and this script is
what says so.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

from .config import OUT
from .draft_sim import DraftUniverse, greedy_pick, heuristic_pick, new_draft, roster_value
from .mcts import MCTS
from .optimize import _eval_squads, _sample_scores


# ======================================================================================
# Agents
# ======================================================================================
class FieldAgent:
    name = "field"
    def __init__(self, temp=0.6):
        self.temp = temp
    def pick(self, state, seat, rng):
        return heuristic_pick(state, seat, rng, self.temp)


class BestFPAgent:
    name = "best-projection"
    def pick(self, state, seat, rng):
        cand = state.candidate_indices(seat)
        if cand.size == 0:
            return -1
        for i in cand[np.argsort(-state.uni.fp[cand])]:
            if state.feasible(int(i), seat):
                return int(i)
        return -1


class GreedyAgent:
    name = "greedy (VORP)"
    def pick(self, state, seat, rng):
        return greedy_pick(state, seat)


class MCTSAgent:
    name = "MCTS"
    def __init__(self, n_teams, base_sims=200, **kw):
        self.n_teams = n_teams
        self.base_sims = base_sims
        self.kw = kw
        self.seed = 0
    def pick(self, state, seat, rng):
        remaining = int(state.teams[seat].needs.sum())
        # Spend the search budget where it matters: the early, deep picks.
        sims = max(40, int(self.base_sims * remaining / 8))
        self.seed += 1
        return MCTS(state.uni, seat, self.n_teams, sims=sims, seed=self.seed,
                    **self.kw).run(state)["best"]


# ======================================================================================
# One draft
# ======================================================================================
def run_draft(uni, n_teams, seat_agents, rng) -> List[List[int]]:
    state = new_draft(uni, n_teams=n_teams, my_seat=0)
    guard = 0
    while not state.done and guard < len(state.order) + 5:
        seat = state.current_seat
        i = seat_agents[seat].pick(state, seat, rng)
        if i is None or i < 0:
            state.ptr += 1
            guard += 1
            continue
        state.apply(int(i))
        guard += 1
    return [t.picks for t in state.teams]


# ======================================================================================
# Tournament
# ======================================================================================
def evaluate(n_drafts=24, n_teams=8, base_sims=200, score_sims=600,
             seed=0, verbose=True) -> Dict:
    from .pipeline import apply_availability, draft_pool, load_board
    board = load_board()
    apply_availability(board, None)
    pool, _, _ = draft_pool(board)
    uni = DraftUniverse(pool)

    rng = np.random.default_rng(seed)
    draws = np.vstack([_sample_scores(uni.pool, rng) for _ in range(score_sims)])

    def mc(picks):
        return float(_eval_squads([list(picks)], draws, uni.pool)[0])

    mcts_agent = MCTSAgent(n_teams, base_sims=base_sims, c_puct=1.6, top_k=12, opp_temp=0.6)
    greedy_agent = GreedyAgent()
    field = FieldAgent(0.6)

    rec = {"mcts": [], "greedy": [], "field": [], "diff": [], "mcts_pt": [], "greedy_pt": []}
    t0 = time.time()
    for d in range(n_drafts):
        seats = list(range(n_teams))
        rng.shuffle(seats)
        # Contestants at two random seats; swap roles each draft to cancel slot effects.
        s_mcts, s_greedy = seats[0], seats[1]
        if d % 2 == 1:
            s_mcts, s_greedy = s_greedy, s_mcts
        agents = {s: field for s in range(n_teams)}
        agents[s_mcts] = mcts_agent
        agents[s_greedy] = greedy_agent

        rosters = run_draft(uni, n_teams, agents, rng)
        vm, vg = mc(rosters[s_mcts]), mc(rosters[s_greedy])
        fields = [mc(rosters[s]) for s in range(n_teams) if s not in (s_mcts, s_greedy)]
        rec["mcts"].append(vm); rec["greedy"].append(vg)
        rec["field"].append(float(np.mean(fields)))
        rec["diff"].append(vm - vg)
        rec["mcts_pt"].append(roster_value(uni, rosters[s_mcts]))
        rec["greedy_pt"].append(roster_value(uni, rosters[s_greedy]))
        if verbose:
            print("  draft %2d/%d  MCTS %6.1f  greedy %6.1f  diff %+5.1f  (%.0fs)"
                  % (d + 1, n_drafts, vm, vg, vm - vg, time.time() - t0),
                  file=sys.stderr, flush=True)

    diff = np.array(rec["diff"])
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    out = {
        "config": {"n_drafts": n_drafts, "n_teams": n_teams, "base_sims": base_sims,
                   "score_sims": score_sims},
        "mcts_mean": round(float(np.mean(rec["mcts"])), 3),
        "greedy_mean": round(float(np.mean(rec["greedy"])), 3),
        "field_mean": round(float(np.mean(rec["field"])), 3),
        "mean_diff": round(float(diff.mean()), 3),
        "diff_stderr": round(float(se), 3),
        "t_stat": round(float(diff.mean() / se), 2) if se and not np.isnan(se) else None,
        "mcts_beats_greedy_rate": round(float((diff > 0).mean()), 3),
        "mcts_point_mean": round(float(np.mean(rec["mcts_pt"])), 3),
        "greedy_point_mean": round(float(np.mean(rec["greedy_pt"])), 3),
        "seconds": round(time.time() - t0, 1),
        "per_draft_diff": [round(x, 2) for x in rec["diff"]],
    }
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=24)
    ap.add_argument("--teams", type=int, default=8)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(OUT, "drafter_eval.json"))
    a = ap.parse_args(argv)
    res = evaluate(n_drafts=a.drafts, n_teams=a.teams, base_sims=a.sims)
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("-> " + a.out)


if __name__ == "__main__":
    main()
