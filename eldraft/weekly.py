"""Weekly (salary-cap) mode data: schedule, matchups and opponent defence.

The draft is a one-off; the normal EuroLeague Fantasy game is played week by week, and the
same player is worth very different amounts in different rounds.  Two things drive that and
neither is in the season-long projection:

  * how many games his team plays that week (a bye is zero, a double-gameweek is ~2x), and
  * who he plays — output rises against a soft defence and falls against a stiff one.

This module extracts both from the feeds so the weekly optimiser can re-weight every
player's projection for a specific round before choosing a squad.
"""
from __future__ import annotations

import collections
import datetime as dt
from typing import Dict, List, Optional

import numpy as np

from . import fetch
from .config import TARGET_SEASON

# Matchup swing is deliberately bounded — one game against a soft defence is an edge, not a
# licence to ignore talent — and home advantage is small but real.
MATCHUP_CLAMP = (0.85, 1.15)
HOME_ADV = 1.03
AWAY_ADV = 0.97


def team_defense(season_code: str = "E2025") -> Dict:
    """Average total PIR each club *conceded* per game — the lower, the tougher the defence."""
    boxes = fetch.season_boxscores(season_code, cache_only=True)
    allowed = collections.defaultdict(list)

    def code_of(side):
        for p in side.get("players", []) or []:
            pl = p.get("player") or {}
            t = (pl.get("club") or {}).get("code") or (pl.get("team") or {}).get("code")
            if t:
                return t
        return None

    def pir_of(side):
        return sum((p.get("stats") or {}).get("valuation", 0) for p in side.get("players", []) or [])

    for box in boxes.values():
        loc, road = box.get("local") or {}, box.get("road") or {}
        lc, rc = code_of(loc), code_of(road)
        if lc and rc:
            allowed[lc].append(pir_of(road))
            allowed[rc].append(pir_of(loc))
    rating = {t: float(np.mean(v)) for t, v in allowed.items() if v}
    league = float(np.mean([x for v in allowed.values() for x in v])) if allowed else 92.0
    return {"rating": {t: round(r, 1) for t, r in rating.items()},
            "league_avg": round(league, 1)}


def schedule(season_code: str = None) -> Dict:
    """Fixtures grouped by round: {round: [{home, away, date}]} for the whole season."""
    season_code = season_code or TARGET_SEASON
    games = fetch.season_games(season_code)
    by_round: Dict[int, List[dict]] = collections.defaultdict(list)
    for g in games:
        rnd = g.get("round")
        if not rnd:
            continue
        home = ((g.get("local") or {}).get("club") or {}).get("code")
        away = ((g.get("road") or {}).get("club") or {}).get("code")
        if not home or not away:
            continue
        by_round[rnd].append({
            "home": home, "away": away,
            "date": (str(g.get("date"))[:10] if g.get("date") else None),
            "phase": ((g.get("phaseType") or {}).get("code")),
        })
    return {int(r): sorted(v, key=lambda x: x["date"] or "") for r, v in by_round.items()}


def next_round(sched: Dict, today: Optional[dt.date] = None) -> int:
    """The soonest round whose first game is still in the future (else round 1)."""
    today = today or dt.date.today()
    best, best_date = None, None
    for rnd, games in sched.items():
        dates = [g["date"] for g in games if g["date"]]
        if not dates:
            continue
        first = min(dates)
        if first >= today.isoformat():
            if best_date is None or first < best_date:
                best, best_date = rnd, first
    return best or (min(sched) if sched else 1)


def build_weekly() -> Dict:
    """Everything the weekly optimiser needs, ready to embed in the page payload."""
    sched = schedule(TARGET_SEASON)
    defense = team_defense("E2025")
    rounds = sorted(sched)
    # Games per team per round, so the UI can flag byes and double-gameweeks at a glance.
    counts = {}
    for rnd in rounds:
        c = collections.Counter()
        for g in sched[rnd]:
            c[g["home"]] += 1
            c[g["away"]] += 1
        counts[rnd] = dict(c)
    return {
        "defense": defense["rating"],
        "leagueAvg": defense["league_avg"],
        "clampLo": MATCHUP_CLAMP[0], "clampHi": MATCHUP_CLAMP[1],
        "homeAdv": HOME_ADV, "awayAdv": AWAY_ADV,
        "schedule": {str(r): sched[r] for r in rounds},
        "gameCounts": {str(r): counts[r] for r in rounds},
        "rounds": rounds,
        "nextRound": next_round(sched),
    }


def weekly_fp(per_game_fp: float, team: str, rnd_games: List[dict],
              defense: Dict[str, float], league_avg: float) -> float:
    """A player's projected points for one round, summed over the games his team plays."""
    total = 0.0
    for g in rnd_games:
        if team not in (g["home"], g["away"]):
            continue
        opp = g["away"] if g["home"] == team else g["home"]
        home = g["home"] == team
        d = defense.get(opp, league_avg)
        mult = max(MATCHUP_CLAMP[0], min(MATCHUP_CLAMP[1], d / league_avg))
        mult *= HOME_ADV if home else AWAY_ADV
        total += per_game_fp * mult
    return total
