"""Assembles one unified player universe from rosters, season stats, box scores and NBA."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from typing import Dict, List, Optional

from . import fetch
from .config import (DATA, EC_HISTORY, EL_HISTORY, GAMELOG_SEASONS, NBA_HISTORY,
                     TARGET_SEASON, norm_position)
from .names import NameIndex, display_name, split_el_name

UNIVERSE_PATH = os.path.join(DATA, "universe.json")


def _f(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace("%", "").strip()
            if not x:
                return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _joined_by_cutoff(row: dict, target_season: str, cutoff_month: int = 12) -> bool:
    """True if the player was on the roster early enough to matter for that season."""
    start = row.get("startDate")
    if not start:
        return True
    try:
        d = dt.datetime.fromisoformat(str(start).replace("Z", "")).date()
    except ValueError:
        return True
    try:
        year = int(target_season[1:])
    except ValueError:
        return True
    return d <= dt.date(year, cutoff_month, 1)


def _age_on(birth: Optional[str], ref: dt.date) -> Optional[float]:
    if not birth:
        return None
    try:
        b = dt.datetime.fromisoformat(str(birth).replace("Z", "")).date()
    except ValueError:
        return None
    return round((ref - b).days / 365.25, 1)


# ======================================================================================
# Season statistic blocks
# ======================================================================================
def _traditional_block(row: dict) -> dict:
    """v3 'traditional' accumulated row -> season totals + derived per-game rates."""
    gp = _f(row.get("gamesPlayed"))
    mins = _f(row.get("minutesPlayed"))
    b = {
        "gp": gp,
        "gs": _f(row.get("gamesStarted")),
        "min": mins,
        "pir": _f(row.get("pir")),
        "pts": _f(row.get("pointsScored")),
        "reb": _f(row.get("totalRebounds")),
        "oreb": _f(row.get("offensiveRebounds")),
        "dreb": _f(row.get("defensiveRebounds")),
        "ast": _f(row.get("assists")),
        "stl": _f(row.get("steals")),
        "to": _f(row.get("turnovers")),
        "blk": _f(row.get("blocks")),
        "blka": _f(row.get("blocksAgainst")),
        "fc": _f(row.get("foulsCommited")),
        "fd": _f(row.get("foulsDrawn")),
        "fg2m": _f(row.get("twoPointersMade")),
        "fg2a": _f(row.get("twoPointersAttempted")),
        "fg3m": _f(row.get("threePointersMade")),
        "fg3a": _f(row.get("threePointersAttempted")),
        "ftm": _f(row.get("freeThrowsMade")),
        "fta": _f(row.get("freeThrowsAttempted")),
        "team": ((row.get("player") or {}).get("team") or {}).get("code"),
    }
    b["mpg"] = mins / gp if gp else 0.0
    b["pir_pg"] = b["pir"] / gp if gp else 0.0
    b["pts_pg"] = b["pts"] / gp if gp else 0.0
    b["reb_pg"] = b["reb"] / gp if gp else 0.0
    b["ast_pg"] = b["ast"] / gp if gp else 0.0
    b["stl_pg"] = b["stl"] / gp if gp else 0.0
    b["blk_pg"] = b["blk"] / gp if gp else 0.0
    b["to_pg"] = b["to"] / gp if gp else 0.0
    b["pir_per_min"] = b["pir"] / mins if mins > 0 else 0.0
    b["pir_p40"] = b["pir_per_min"] * 40.0
    b["start_rate"] = (b["gs"] / gp) if gp else 0.0
    return b


def _advanced_block(row: dict) -> dict:
    return {
        "efg": _f(row.get("effectiveFieldGoalPercentage")),
        "ts": _f(row.get("trueShootingPercentage")),
        "oreb_pct": _f(row.get("offensiveReboundsPercentage")),
        "dreb_pct": _f(row.get("defensiveReboundsPercentage")),
        "reb_pct": _f(row.get("reboundsPercentage")),
        "ast_to": _f(row.get("assistsToTurnoversRatio")),
        "ast_ratio": _f(row.get("assistsRatio")),
        "to_ratio": _f(row.get("turnoversRatio")),
        "fg2a_ratio": _f(row.get("twoPointAttemptsRatio")),
        "fg3a_ratio": _f(row.get("threePointAttemptsRatio")),
        "ft_rate": _f(row.get("freeThrowsRate")),
        "poss": _f(row.get("possesions")),
    }


def _season_name_index(seasons: Dict[str, Dict[str, dict]]) -> NameIndex:
    """Lookup of historical stat lines by name, for roster entries the feed has not yet
    assigned a person code to (typically the summer's new signings)."""
    idx = NameIndex()
    seen = set()
    for season_code, table in seasons.items():
        for code, blk in table.items():
            key = (blk.get("name") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            first, last = split_el_name(blk.get("name") or "")
            idx.add(first, last, {"code": code, "name": blk.get("name")})
    return idx


def _collect_season(season_code: str) -> Dict[str, dict]:
    trad = fetch.season_player_stats(season_code, "traditional", "accumulated")
    adv = fetch.season_player_stats(season_code, "advanced", "accumulated")
    adv_by = {(r.get("player") or {}).get("code"): r for r in adv}
    out: Dict[str, dict] = {}
    for r in trad:
        p = r.get("player") or {}
        code = p.get("code")
        if not code:
            continue
        blk = _traditional_block(r)
        blk.update(_advanced_block(adv_by.get(code, {})))
        blk["name"] = p.get("name")
        out[code] = blk

    # Per-game averages are taken over games the player *appeared in*, so they cannot be
    # summed across a roster - a squad's mpg column adds up to far more than the 200
    # minutes a team actually spends.  Re-express everything per TEAM game as well, which
    # is both additive and the unit fantasy scoring actually pays out in.
    team_games: Dict[str, float] = {}
    for blk in out.values():
        t = blk.get("team")
        if t:
            team_games[t] = max(team_games.get(t, 0.0), blk.get("gp", 0.0))
    league_games = max(team_games.values()) if team_games else 1.0
    for blk in out.values():
        tg = team_games.get(blk.get("team")) or league_games or 1.0
        blk["team_games"] = tg
        blk["min_per_team_game"] = blk["min"] / tg if tg else 0.0
        blk["pir_per_team_game"] = blk["pir"] / tg if tg else 0.0
        blk["played_share"] = min(1.0, blk["gp"] / tg) if tg else 0.0
    return out


# ======================================================================================
# Box-score game logs
# ======================================================================================
def _collect_gamelogs(season_code: str, cache_only: bool = True) -> Dict[str, List[dict]]:
    """Per-player game-by-game lines, used for variance / form / role modelling."""
    boxes = fetch.season_boxscores(season_code, cache_only=cache_only)
    games = {g["gameCode"]: g for g in fetch.season_games(season_code)}
    logs: Dict[str, List[dict]] = {}
    for gcode, box in boxes.items():
        meta = games.get(gcode, {})
        rnd = meta.get("round")
        date = meta.get("date") or meta.get("localDate")
        phase = ((meta.get("phaseType") or {}).get("code")) or "RS"
        for side in ("local", "road"):
            block = box.get(side) or {}
            for entry in block.get("players", []) or []:
                pl = entry.get("player") or {}
                person = pl.get("person") or {}
                code = person.get("code")
                st = entry.get("stats") or {}
                if not code:
                    continue
                # timePlayed is reported in seconds in the box-score feed.
                mins = _f(st.get("timePlayed")) / 60.0
                logs.setdefault(code, []).append(
                    {
                        "g": gcode,
                        "round": rnd,
                        "date": (str(date)[:10] if date else None),
                        "phase": phase,
                        "home": side == "local",
                        "min": mins,
                        "pir": _f(st.get("valuation")),
                        "pts": _f(st.get("points")),
                        "reb": _f(st.get("totalRebounds")),
                        "ast": _f(st.get("assistances")),
                        "pm": _f(st.get("plusMinus")),
                        "start": bool(st.get("startFive") or st.get("startFive2")),
                        "dnp": mins <= 0.0,
                    }
                )
    for code in logs:
        logs[code].sort(key=lambda r: (r["round"] or 0, r["g"]))
    return logs


# ======================================================================================
# NBA
# ======================================================================================
def _nba_pir_estimate(s: dict) -> float:
    """PIR is not published for the NBA, so reconstruct it from the box score.

    PIR = (PTS+REB+AST+STL+BLK+FoulsDrawn) - (missedFG+missedFT+TO+BlocksAgainst+Fouls).
    ESPN exposes neither fouls drawn nor blocks against; those two terms are of similar
    magnitude and opposite sign, so dropping both is a better approximation than
    dropping one.  Documented here because it feeds a cross-league comparison.
    """
    fga, fgm = _f(s.get("avgFieldGoalsAttempted")), _f(s.get("avgFieldGoalsMade"))
    fta, ftm = _f(s.get("avgFreeThrowsAttempted")), _f(s.get("avgFreeThrowsMade"))
    pos = (
        _f(s.get("avgPoints")) + _f(s.get("avgRebounds")) + _f(s.get("avgAssists"))
        + _f(s.get("avgSteals")) + _f(s.get("avgBlocks"))
    )
    neg = (fga - fgm) + (fta - ftm) + _f(s.get("avgTurnovers")) + _f(s.get("avgFouls"))
    return round(pos - neg, 2)


def _collect_nba(nba_history: List[int] = None) -> NameIndex:
    idx = NameIndex()
    per_player: Dict[str, dict] = {}
    for season in (nba_history if nba_history is not None else NBA_HISTORY):
        for row in fetch.nba_season_stats(season):
            s = row["stats"]
            gp = _f(s.get("gamesPlayed"))
            if gp < 1:
                continue
            key = str(row.get("espn_id"))
            rec = per_player.setdefault(
                key,
                {"espn_id": row.get("espn_id"), "name": row.get("name"),
                 "first": row.get("first"), "last": row.get("last"),
                 "position": row.get("position"), "seasons": {}},
            )
            rec["seasons"][str(season)] = {
                "team": row.get("team"),
                "gp": gp,
                "mpg": _f(s.get("avgMinutes")),
                "pts": _f(s.get("avgPoints")),
                "reb": _f(s.get("avgRebounds")),
                "ast": _f(s.get("avgAssists")),
                "stl": _f(s.get("avgSteals")),
                "blk": _f(s.get("avgBlocks")),
                "to": _f(s.get("avgTurnovers")),
                "fg_pct": _f(s.get("fieldGoalPct")),
                "fg3_pct": _f(s.get("threePointFieldGoalPct")),
                "ft_pct": _f(s.get("freeThrowPct")),
                "pir_pg": _nba_pir_estimate(s),
            }
    for rec in per_player.values():
        idx.add(rec.get("first") or "", rec.get("last") or "", rec)
    return idx


# ======================================================================================
# Universe
# ======================================================================================
def build_universe(save: bool = True, target_season: str = None,
                   el_history: List[str] = None, ec_history: List[str] = None,
                   nba_history: List[int] = None, gamelog_seasons: List[str] = None,
                   path: str = None) -> dict:
    """Assemble the player universe for `target_season` from strictly prior seasons.

    The history arguments exist so backtest.py can rebuild the world as it looked before
    a past season and grade the model against what actually happened.
    """
    target_season = target_season or TARGET_SEASON
    el_history = list(el_history if el_history is not None else EL_HISTORY)
    ec_history = list(ec_history if ec_history is not None else EC_HISTORY)
    nba_history = list(nba_history if nba_history is not None else NBA_HISTORY)
    gamelog_seasons = list(gamelog_seasons if gamelog_seasons is not None else GAMELOG_SEASONS)

    today = dt.date.today()
    fetch.log("· building universe for {}".format(target_season))

    el_seasons = {s: _collect_season(s) for s in el_history}
    ec_seasons = {s: _collect_season(s) for s in ec_history}
    logs = {s: _collect_gamelogs(s) for s in gamelog_seasons}
    nba_idx = _collect_nba(nba_history)

    el_name_idx = _season_name_index(el_seasons)
    ec_name_idx = _season_name_index(ec_seasons)

    roster_rows = fetch.all_rosters(target_season)
    clubs = {c["code"]: c for c in fetch.clubs(target_season)}

    players: List[dict] = []
    coaches: List[dict] = []
    synthetic = 0
    for row in roster_rows:
        person = row.get("person") or {}
        code = person.get("code")
        raw_name = person.get("name") or ""
        first, last = split_el_name(raw_name)
        if not code:
            # New signings often appear on a roster before the feed mints a person code.
            # Give them a stable synthetic id and resolve their history by name instead.
            if not raw_name.strip():
                continue
            code = "N" + hashlib.sha1(raw_name.encode()).hexdigest()[:8]
            synthetic += 1
        club_code = row.get("_club")
        rec = {
            "code": code,
            "raw_name": raw_name,
            "name": display_name(raw_name),
            "first": first,
            "last": last,
            "club": club_code,
            "club_name": (clubs.get(club_code) or {}).get("abbreviatedName")
            or (clubs.get(club_code) or {}).get("name"),
            "country": (person.get("country") or {}).get("name"),
            "country_code": (person.get("country") or {}).get("code"),
            "height": person.get("height") or None,
            "weight": person.get("weight") or None,
            "birth_date": (str(person.get("birthDate"))[:10] if person.get("birthDate") else None),
            "age": _age_on(person.get("birthDate"), today),
            "dorsal": row.get("dorsalRaw") or row.get("dorsal"),
            "image": (person.get("images") or {}).get("profile")
            or (row.get("images") or {}).get("profile"),
            "active": bool(row.get("active", True)),
        }
        # The feed's person types: J = player, E = head coach, plus assistant coaches
        # and other staff.  Past-season rosters list everyone who passed through the
        # club, staff included, so filter strictly on type rather than on typeName.
        if row.get("type") == "E":
            rec["role"] = "coach"
            coaches.append(rec)
            continue
        if row.get("type") != "J":
            continue
        # Players who joined late in a completed season were never really squad options
        # for that campaign; ignore them so the depth chart reflects the actual rotation.
        if not _joined_by_cutoff(row, target_season):
            continue

        rec["role"] = "player"
        rec["position"] = norm_position(row.get("positionName"))
        rec["position_raw"] = row.get("positionName")
        # Resolve historical codes: the roster code first, then a name match for the
        # codeless entries.  Name matching is restricted to strict tiers.
        el_code = code if any(code in el_seasons[s] for s in el_history) else None
        ec_code = code if any(code in ec_seasons[s] for s in ec_history) else None
        if el_code is None:
            hit = el_name_idx.lookup(first, last, max_tier=1)
            el_code = hit["code"] if hit else None
        if ec_code is None:
            hit = ec_name_idx.lookup(first, last, max_tier=1)
            ec_code = hit["code"] if hit else None
        rec["el"] = {s: el_seasons[s][el_code] for s in el_history
                     if el_code and el_code in el_seasons[s]}
        rec["ec"] = {s: ec_seasons[s][ec_code] for s in ec_history
                     if ec_code and ec_code in ec_seasons[s]}
        rec["stat_code"] = el_code or ec_code
        rec["code_synthetic"] = code.startswith("N") and len(code) == 9
        log_code = el_code or code
        rec["logs"] = {s: logs[s].get(log_code, []) for s in gamelog_seasons
                       if logs[s].get(log_code)}

        nba = nba_idx.lookup(first, last, max_tier=1)
        rec["nba"] = nba["seasons"] if nba else {}
        rec["nba_name"] = nba["name"] if nba else None

        # Did this player appear for a different EuroLeague club last season?
        prev = rec["el"].get(el_history[0]) if el_history else None
        rec["prev_team"] = prev.get("team") if prev else None
        rec["new_to_team"] = bool(prev and prev.get("team") and prev["team"] != club_code)
        players.append(rec)

    uni = {
        "meta": {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "target_season": target_season,
            "el_history": el_history,
            "ec_history": ec_history,
            "nba_history": nba_history,
            "n_players": len(players),
            "n_synthetic_codes": synthetic,
        },
        "clubs": clubs,
        "players": players,
        "coaches": coaches,
    }
    if save:
        with open(path or UNIVERSE_PATH, "w") as fh:
            json.dump(uni, fh)
        fetch.log("· universe -> {} ({} players)".format(path or UNIVERSE_PATH, len(players)))
    return uni


def load_universe() -> dict:
    with open(UNIVERSE_PATH) as fh:
        return json.load(fh)
