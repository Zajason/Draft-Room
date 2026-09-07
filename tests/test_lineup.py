"""Lineup scoring invariants: formation-legal best-six and the reactive policy."""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eldraft.lineup import (  # noqa: E402
    _BENCHMULT,
    _CAPX,
    FORMATIONS,
    _choose_full,
    _score_full,
    best_lineup,
    choose_full_fast,
    reactive_round,
)

POS10 = ["G", "G", "G", "G", "F", "F", "F", "F", "C", "C"]


def _brute_best(scores, pos):
    idx = list(range(len(scores)))
    best = None
    for five in itertools.combinations(idx, 5):
        cnt = {"G": 0, "F": 0, "C": 0}
        for i in five:
            cnt[pos[i]] += 1
        if (cnt["G"], cnt["F"], cnt["C"]) not in FORMATIONS:
            continue
        for sixth in idx:
            if sixth in five:
                continue
            full = set(five) | {sixth}
            fs = sum(scores[i] for i in full)
            bs = sum(scores[i] for i in idx if i not in full)
            val = fs + _BENCHMULT * bs + _CAPX * max(scores[i] for i in full)
            if best is None or val > best:
                best = val
    return best


def test_best_lineup_matches_brute_force():
    random.seed(0)
    for _ in range(2000):
        pos = POS10[:]
        random.shuffle(pos)
        scores = [round(random.uniform(-3, 35), 1) for _ in range(10)]
        assert abs(best_lineup(scores, pos) - _brute_best(scores, pos)) < 1e-6


def test_fast_selector_matches_exact():
    random.seed(1)
    for _ in range(2000):
        pos = POS10[:]
        random.shuffle(pos)
        key = [round(random.uniform(0, 30), 2) for _ in range(10)]
        a = choose_full_fast(key, pos)
        b = _choose_full(key, pos)
        assert abs(sum(key[i] for i in a) - sum(key[i] for i in b)) < 1e-6


def test_reactive_between_floor_and_ceiling():
    random.seed(2)
    for _ in range(2000):
        pos = POS10[:]
        random.shuffle(pos)
        mean = [max(0.5, random.uniform(2, 28)) for _ in range(10)]
        realized = [max(0.0, random.gauss(mean[i], 0.5 * mean[i])) for i in range(10)]
        day = [random.randint(0, 3) for _ in range(10)]
        floor = _score_full(realized, _choose_full(mean, pos))     # commit on the mean
        react = reactive_round(realized, mean, pos, day)
        ceil = best_lineup(realized, pos)
        assert floor - 1e-6 <= react <= ceil + 1e-6
