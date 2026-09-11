"""Projection engine.

Produces, for every player on a 2026-27 EuroLeague roster, a projected fantasy score per
round together with an honest uncertainty band.  The chain is:

  1. rate    - PIR per 40 minutes, pooled across EuroLeague / EuroCup / NBA with recency
               decay, league-difficulty conversion and reliability weighting;
  2. shrink  - empirical-Bayes pull toward a positional prior, strength = 420 minutes;
  3. age     - a peak-age curve applied from the sample's mean age to the target season;
  4. minutes - a zero-sum depth-chart allocation: every club distributes 200 minutes per
               game across its actual 2026-27 roster, so a big summer signing genuinely
               takes minutes off the incumbent ahead of him;
  5. score   - projected PIR per game = rate x minutes, scaled by team context and
               availability, with sigma from game-log variance plus projection error.

Nothing here is fitted to the season we are projecting; every input is prior-season data.
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple

from .config import EC_HISTORY, EL_HISTORY, GAMELOG_SEASONS, NBA_HISTORY, WEIGHTS


class Horizon:
    """Which past seasons the model is allowed to see.

    Defaults to the live configuration.  The backtest passes a horizon that ends before
    the season being predicted, which is what keeps the evaluation honest - the model
    never sees a minute of the season it is scoring against.
    """

    def __init__(self, el=None, ec=None, nba=None, logs=None):
        self.el = list(el if el is not None else EL_HISTORY)
        self.ec = list(ec if ec is not None else EC_HISTORY)
        self.nba = list(nba if nba is not None else NBA_HISTORY)
        self.logs = list(logs if logs is not None else GAMELOG_SEASONS)


DEFAULT_HORIZON = Horizon()

# Minutes available per club per game, and the share that normally goes to each position
# group (two guards, two forwards and one big on the floor, adjusted for small-ball).
CLUB_MINUTES = 200.0
POSITION_MINUTE_SHARE = {"G": 0.40, "F": 0.38, "C": 0.22}

# All minute figures below are EXPECTED MINUTES PER TEAM GAME, not minutes per
# appearance: they are additive across a roster (each club spends exactly 200) and they
# already carry the games a player is expected to miss.  No EuroLeague player has ever
# averaged much above 30 on this basis, so that is the ceiling.
MPG_CAP = 30.5
MPG_KNEE = 23.0
MPG_FLOOR = 0.0


def _soft_cap(m: float) -> float:
    """Compress heavy workloads toward the ceiling instead of clipping them flat.

    A hard cap makes every star identical at exactly the limit, which destroys real
    separation right where a draft needs it most.  Below the knee nothing changes;
    above it, minutes approach MPG_CAP asymptotically and never quite arrive.
    """
    if m <= MPG_KNEE:
        return m
    span = MPG_CAP - MPG_KNEE
    return MPG_KNEE + span * math.tanh((m - MPG_KNEE) / span)

# Minutes per TEAM game by season-long rotation rank, averaged over the 2024-25 and
# 2025-26 EuroLeague seasons (38 club-seasons).  Note how much flatter this is than the
# equivalent curve measured game by game: the man who leads a single box score is rarely
# the same man twice, so a per-game curve badly overstates the season-level top end.
# Normalised to CLUB_MINUTES below.
ROTATION_CURVE = [26.45, 23.52, 21.72, 19.68, 18.39, 16.34, 14.64, 12.91,
                  11.61, 9.62, 8.08, 6.03, 3.84, 2.47, 1.37, 0.72, 0.30, 0.10]
_CURVE_SUM = sum(ROTATION_CURVE)

# How hard to push each position group back toward its nominal share of the floor.
# 1.0 would force every club into an identical positional split; 0 would ignore
# position entirely.  Teams genuinely do play big or small, so we correct partway.
POSITION_CORRECTION = 0.35

# How sharply minutes concentrate on the best players.  1.0 shares the floor in
# proportion to each player's claim, which is far too flat - real EuroLeague rotations
# are top-heavy.  Values above 1 concentrate; the value below is chosen by the sweep in
# backtest.py, not by taste.
MINUTE_CONCENTRATION = 1.6

# The depth chart now allocates minutes-WHEN-FIT; availability is a separate, explicit
# multiplier (see _durability), so these older single-blend constants are retired.
INJURY_FORGIVENESS = 0.55
LEAGUE_AVAIL = 0.88

# Durability = expected share of his club's games a player actually suits up for, learned
# from played_share (games played / the team's real games) across seasons, recency-
# weighted, and shrunk toward the league norm so one thin season can't brand a player.
# EuroLeague/EuroCup only: NBA games-played conflates load-management rest with injury.
DUR_PRIOR = 0.85           # league-average availability
DUR_SHRINK = 1.0           # season-equivalents of prior mixed in
DUR_FLOOR = 0.40

# "Trust proven minutes": a player with a stable, established role at the SAME club has
# already earned those minutes despite that club's depth, so the zero-sum re-allocation
# shouldn't tax him again for competition his own history survived.  When his allocation
# comes in below his proven same-club minutes, pull it back up (then the club is re-
# normalised, so the lift comes out of the churn around him - which is the honest trade).
STABILITY_W_MAX = 0.6      # most of the way back to proven minutes for a rock-stable role
MOVER_TRUST = 0.5          # a proven starter who changed clubs is trusted, but less (new role)

# Width of the projection's own uncertainty, as a coefficient of variation: a floor for
# a player we know well, plus a term that grows as the sample thins.  Calibrated so that
# roughly 68% of outcomes land inside one sigma in the backtests.
SIGMA_BASE = 0.11
SIGMA_THIN_SLOPE = 0.32

# Weight given to the measured rank curve versus the continuous proportional split.
# The curve reproduces the right *shape* but throws away how far ahead of the next man
# a player is; blending keeps both.
CURVE_BLEND = 0.80


def _curve_minutes(rank: int, squad_size: int) -> float:
    """Minutes for the r-th man in the pecking order, from the measured curve."""
    raw = ROTATION_CURVE[rank] if rank < len(ROTATION_CURVE) else 0.02
    return raw * CLUB_MINUTES / _CURVE_SUM


# ======================================================================================
# Observation gathering
# ======================================================================================
def _observations(p: dict, hz: Horizon = None) -> List[dict]:
    """Flatten a player's history into weighted (rate, minutes, age) observations."""
    hz = hz or DEFAULT_HORIZON
    obs: List[dict] = []
    decay = WEIGHTS.season_decay
    strength = WEIGHTS.league_strength
    reliability = WEIGHTS.league_reliability
    age_now = p.get("age")

    for i, season in enumerate(hz.el):
        b = (p.get("el") or {}).get(season)
        if b and b.get("min", 0) > 0:
            obs.append({
                "league": "EL", "season": season, "years_ago": i + 1,
                "min": b["min"], "gp": b["gp"], "mpg": b["mpg"],
                "mptg": b.get("min_per_team_game", b["mpg"]),
                "played_share": b.get("played_share", 1.0),
                "rate": b["pir_p40"] * strength["EL"],
                "w": decay[min(i, len(decay) - 1)] * reliability["EL"],
                "age": (age_now - (i + 1)) if age_now else None,
                "start_rate": b.get("start_rate", 0.0),
            })
    for i, season in enumerate(hz.ec):
        b = (p.get("ec") or {}).get(season)
        if b and b.get("min", 0) > 0:
            obs.append({
                "league": "EC", "season": season, "years_ago": i + 1,
                "min": b["min"], "gp": b["gp"], "mpg": b["mpg"],
                "mptg": b.get("min_per_team_game", b["mpg"]),
                "played_share": b.get("played_share", 1.0),
                "rate": b["pir_p40"] * strength["EC"],
                "w": decay[min(i, len(decay) - 1)] * reliability["EC"],
                "age": (age_now - (i + 1)) if age_now else None,
                "start_rate": b.get("start_rate", 0.0),
            })
    nba = p.get("nba") or {}
    # ESPN season keys end the season (2026 == 2025-26).  The newest season the horizon
    # allows defines "one year ago"; anything more recent is invisible to the model.
    nba_ref = (max(int(x) for x in hz.nba) + 1) if hz.nba else 0
    allowed = {str(x) for x in hz.nba}
    for key, b in nba.items():
        if key not in allowed:
            continue
        try:
            years_ago = nba_ref - int(key)
        except ValueError:
            continue
        gp, mpg = b.get("gp", 0.0), b.get("mpg", 0.0)
        mins = gp * mpg
        if mins <= 0:
            continue
        idx = max(0, years_ago - 1)
        obs.append({
            "league": "NBA", "season": key, "years_ago": years_ago,
            "min": mins, "gp": gp, "mpg": mpg,
            # NBA regular seasons are 82 games; scale to a per-team-game basis the same way.
            "mptg": mpg * min(1.0, gp / 82.0),
            "played_share": min(1.0, gp / 82.0),
            "rate": (b.get("pir_pg", 0.0) / mpg * 40.0) * strength["NBA"] if mpg else 0.0,
            "w": decay[min(idx, len(decay) - 1)] * reliability["NBA"],
            "age": (age_now - years_ago) if age_now else None,
            "start_rate": 0.0,
        })
    return obs


# ======================================================================================
# Age curve
# ======================================================================================
def age_factor(age: Optional[float]) -> float:
    """Multiplicative production level for a given age, normalised to 1.0 at peak."""
    if age is None:
        return 0.98
    w = WEIGHTS
    if age <= w.peak_age:
        f = 1.0 - w.age_rise_per_year * (w.peak_age - age)
    else:
        f = 1.0 - w.age_decline_per_year * (age - w.peak_age)
    return max(0.62, min(1.02, f))


# ======================================================================================
# Rate projection (steps 1-3)
# ======================================================================================
def _positional_priors(players: List[dict], hz: Horizon = None) -> Dict[str, Tuple[float, float]]:
    """Minutes-weighted mean PIR/40 and mean MPG per position, from the last EL season."""
    hz = hz or DEFAULT_HORIZON
    acc: Dict[str, List[Tuple[float, float, float]]] = {}
    last = hz.el[0]
    for p in players:
        b = (p.get("el") or {}).get(last)
        if b and b.get("min", 0) >= 100:
            acc.setdefault(p["position"], []).append(
                (b["min"], b["pir_p40"], b.get("min_per_team_game", b["mpg"])))
    priors: Dict[str, Tuple[float, float]] = {}
    for pos, rows in acc.items():
        tot = sum(r[0] for r in rows) or 1.0
        rate = sum(r[0] * r[1] for r in rows) / tot
        mpg = sum(r[2] for r in rows) / len(rows)
        priors[pos] = (rate, mpg)
    for pos in ("G", "F", "C"):
        priors.setdefault(pos, (10.0, 17.0))
    return priors


def project_rate(p: dict, priors: Dict[str, Tuple[float, float]], hz: Horizon = None) -> dict:
    """Shrunk, age-adjusted EuroLeague-equivalent PIR per 40 minutes."""
    obs = _observations(p, hz)
    prior_rate, prior_mpg = priors.get(p["position"], (10.0, 17.0))

    if not obs:
        # No professional record we can see: sit on the positional prior, discounted,
        # and flag the player as unknown so the optimiser prices in the risk.
        return {
            "rate": prior_rate * 0.62 * age_factor(p.get("age")),
            "eff_minutes": 0.0, "raw_rate": None, "n_obs": 0,
            "sample_leagues": [], "el_minutes": 0.0, "unknown": True,
        }

    wsum = sum(o["min"] * o["w"] for o in obs)
    raw = sum(o["min"] * o["w"] * o["rate"] for o in obs) / wsum if wsum else prior_rate

    k = WEIGHTS.shrink_minutes_k
    shrunk = (wsum * raw + k * prior_rate) / (wsum + k)

    # Translate the sample's effective age to the season we are projecting.
    mean_age = None
    ages = [(o["min"] * o["w"], o["age"]) for o in obs if o.get("age") is not None]
    if ages:
        tw = sum(a[0] for a in ages) or 1.0
        mean_age = sum(a[0] * a[1] for a in ages) / tw
    age_mult = 1.0
    if mean_age is not None and p.get("age") is not None:
        base = age_factor(mean_age)
        age_mult = (age_factor(p["age"]) / base) if base > 0 else 1.0
        age_mult = max(0.80, min(1.22, age_mult))

    el_minutes = sum(o["min"] for o in obs if o["league"] == "EL")
    return {
        "rate": shrunk * age_mult,
        "raw_rate": raw,
        "eff_minutes": wsum,
        "n_obs": len(obs),
        "sample_leagues": sorted({o["league"] for o in obs}),
        "el_minutes": el_minutes,
        "age_mult": age_mult,
        "unknown": False,
    }


# ======================================================================================
# Minutes projection (step 4) - the depth chart
# ======================================================================================
def _raw_minute_prior(p: dict, priors: Dict[str, Tuple[float, float]], hz: Horizon = None) -> float:
    """Pre-allocation expectation of minutes, before club-level competition is applied."""
    obs = _observations(p, hz)
    _, prior_mpg = priors.get(p["position"], (10.0, 17.0))
    if not obs:
        return prior_mpg * 0.45

    # Weight by games (not minutes) so a 30 mpg starter with 12 games is not swamped.
    wsum = sum(o["gp"] * o["w"] for o in obs)
    if wsum <= 0:
        return prior_mpg * 0.45
    # EuroCup and NBA minutes both translate down: EuroCup rotations are shallower and
    # NBA role does not carry over cleanly to a 40-minute European game.
    conv = {"EL": 1.0, "EC": 0.86, "NBA": 0.92}
    # Minutes WHEN FIT (per appearance): the depth chart allocates the role a player holds
    # when he suits up.  Availability is applied separately and explicitly as durability,
    # so an injury season shrinks his value once, through durability, not twice.
    raw = sum(
        o["gp"] * o["w"] * conv[o["league"]] * o["mpg"]
        for o in obs
    ) / wsum

    k = WEIGHTS.shrink_games_k
    shrunk = (wsum * raw + k * prior_mpg * 0.7) / (wsum + k)

    # A player joining a new club has a less certain role; pull him toward the mean - but
    # much less for an established high-minute veteran, whose proven workload travels with
    # him (he was signed to keep playing), than for a young or fringe mover we must guess at.
    if p.get("new_to_team"):
        trust = _career_minute_stability(p, hz)
        pull = 0.18 * (1.0 - trust)
        shrunk = (1.0 - pull) * shrunk + pull * prior_mpg
    # Ageing bigs and guards lose rotation minutes faster than they lose efficiency.
    if p.get("age") and p["age"] > 33:
        shrunk *= max(0.72, 1.0 - 0.045 * (p["age"] - 33))
    return max(0.0, shrunk)


def _durability(p: dict, hz: Horizon = None) -> float:
    """Expected share of club games the player suits up for, from multi-season history.

    Each season is one observation of his durability regardless of how few games it
    contains - a four-game injury year is weighted by recency like any other, not shrunk
    away for being small, because "he was hurt" is exactly the signal we want to keep.
    """
    obs = _observations(p, hz)
    rows = [(o["w"], o["played_share"]) for o in obs
            if o["league"] in ("EL", "EC") and o.get("played_share") is not None]
    if not rows:
        return DUR_PRIOR
    wsum = sum(w for w, _ in rows)
    psum = sum(w * ps for w, ps in rows)
    dur = (psum + DUR_SHRINK * DUR_PRIOR) / (wsum + DUR_SHRINK)
    return max(DUR_FLOOR, min(1.0, dur))


def _same_club_el_minutes(p: dict, hz: Horizon) -> List[Tuple[float, float, float]]:
    """(mpg, gp, recency weight) for the EuroLeague seasons the player spent at his club."""
    club = p.get("club")
    if not club:
        return []
    decay = WEIGHTS.season_decay
    rows = []
    for i, s in enumerate(hz.el):
        b = (p.get("el") or {}).get(s)
        if b and b.get("team") == club and b.get("min", 0) > 0 and b.get("mpg"):
            rows.append((float(b["mpg"]), float(b.get("gp", 0)), decay[min(i, len(decay) - 1)]))
    return rows


def _career_el_minutes(p: dict, hz: Horizon) -> List[Tuple[float, float, float]]:
    """(mpg, gp, recency weight) for every EuroLeague season, regardless of club."""
    decay = WEIGHTS.season_decay
    rows = []
    for i, s in enumerate(hz.el):
        b = (p.get("el") or {}).get(s)
        if b and b.get("min", 0) > 0 and b.get("mpg"):
            rows.append((float(b["mpg"]), float(b.get("gp", 0)), decay[min(i, len(decay) - 1)]))
    return rows


def _proven_minutes(p: dict, hz: Horizon) -> float:
    """Recency-weighted minutes the player has held - at this club, or his career if a mover."""
    rows = _same_club_el_minutes(p, hz) or _career_el_minutes(p, hz)
    wsum = sum(w for _, _, w in rows)
    return sum(m * w for m, _, w in rows) / wsum if wsum else 0.0


def _career_minute_stability(p: dict, hz: Horizon) -> float:
    """0..1: how proven and consistent a player's minutes are across his EuroLeague career.

    Unlike _role_stability this ignores which club he was at - it measures whether he is an
    established high-minute player whose role travels, so a mover like Mike James (three
    straight ~30-mpg seasons) is not dampened toward the positional mean as if his workload
    were a guess.
    """
    decay = WEIGHTS.season_decay
    rows = []
    for i, s in enumerate(hz.el):
        b = (p.get("el") or {}).get(s)
        if b and b.get("min", 0) > 0 and b.get("mpg"):
            rows.append((float(b["mpg"]), decay[min(i, len(decay) - 1)]))
    if len(rows) < 2:
        return 0.0
    mpgs = [m for m, _ in rows]
    mean = sum(mpgs) / len(mpgs)
    if mean <= 0:
        return 0.0
    var = sum((m - mean) ** 2 for m in mpgs) / len(mpgs)
    cv = (var ** 0.5) / mean
    consistency = max(0.0, min(1.0, 1.0 - cv / 0.30))
    tenure = min(1.0, (len(rows) - 1) / 2.0)
    established = max(0.0, min(1.0, (mean - 14.0) / 12.0))   # a real high-minute role
    return consistency * tenure * established


def _role_stability(p: dict, hz: Horizon) -> float:
    """0..STABILITY_W_MAX: how much to trust a player's proven minutes over the re-allocation.

    Highest for a proven starter staying put (several consistent, high-minute seasons at
    THIS club).  A proven starter who changed clubs keeps a reduced share of that trust
    (MOVER_TRUST): his workload travels with him, but his exact new role is less certain.
    """
    if p.get("new_to_team"):
        return STABILITY_W_MAX * MOVER_TRUST * _career_minute_stability(p, hz)
    rows = _same_club_el_minutes(p, hz)
    if len(rows) < 2:
        return 0.0
    mpgs = [m for m, _, _ in rows]
    mean = sum(mpgs) / len(mpgs)
    if mean <= 0:
        return 0.0
    var = sum((m - mean) ** 2 for m in mpgs) / len(mpgs)
    cv = (var ** 0.5) / mean
    consistency = max(0.0, min(1.0, 1.0 - cv / 0.30))      # cv 0 -> 1, cv >= 0.30 -> 0
    tenure = min(1.0, (len(rows) - 1) / 2.0)               # 2 seasons -> 0.5, 3+ -> 1
    established = max(0.0, min(1.0, (mean - 12.0) / 12.0))  # <=12 mpg -> 0, >=24 -> 1
    return STABILITY_W_MAX * consistency * tenure * established


def allocate_minutes(players: List[dict], priors: Dict[str, Tuple[float, float]], hz: Horizon = None) -> None:
    """Distribute each club's 200 minutes per game across its actual roster, in place.

    This is where "fit" lives.  Minutes are zero-sum, so a club that signed three
    guards cannot give all of them their old workload - somebody's projection has to
    fall, and the model says whose.

    The shape of the allocation is not invented.  ROTATION_CURVE is measured from last
    season's box scores: EuroLeague sides play about eleven men, the top eight take
    ~87% of the minutes, and the twelfth man is a garbage-time cameo.  Spreading 200
    minutes evenly over a 17-man roster - the obvious implementation - would have
    projected Mike James at 15 minutes a night instead of 30.

    Allocation runs in three steps: rank the club by talent-weighted claim and read
    minutes off the empirical curve; nudge each position group toward its normal share
    of the floor; then cap individual workloads and re-normalise back to 200.
    """
    by_club: Dict[str, List[dict]] = {}
    for p in players:
        p["_mpg_prior"] = _raw_minute_prior(p, priors, hz)
        p["_proven_mpg"] = _proven_minutes(p, hz)
        p["_role_stability"] = _role_stability(p, hz)
        # Better players earn a larger share of the pool than their raw prior implies.
        talent = max(0.35, min(2.0, p["_rate"]["rate"] / max(1e-6, priors[p["position"]][0])))
        p["_claim"] = p["_mpg_prior"] * (talent ** 0.85)
        by_club.setdefault(p.get("club") or "FA", []).append(p)

    for club, squad in by_club.items():
        squad = sorted(squad, key=lambda x: -x["_claim"])
        n = len(squad)

        # 1. rotation shape ------------------------------------------------------------
        # Two views of the same thing, blended.  The measured curve knows that the ninth
        # man plays 11 minutes and the twelfth plays none; the proportional split knows
        # that this particular first option is miles clear of the second.
        claims = [max(1e-6, p["_claim"]) ** MINUTE_CONCENTRATION for p in squad]
        ctotal = sum(claims) or 1.0
        alloc = {}
        for rank, (p, w) in enumerate(zip(squad, claims)):
            curve = _curve_minutes(rank, n)
            prop = CLUB_MINUTES * w / ctotal
            alloc[id(p)] = CURVE_BLEND * curve + (1.0 - CURVE_BLEND) * prop

        # 2. positional balance --------------------------------------------------------
        # Teams do go big or small, so correct only partway toward the nominal share.
        for _ in range(3):
            for pos, share in POSITION_MINUTE_SHARE.items():
                group = [p for p in squad if p["position"] == pos]
                if not group:
                    continue
                actual = sum(alloc[id(p)] for p in group)
                target = CLUB_MINUTES * share
                if actual <= 1e-6:
                    continue
                factor = (target / actual) ** POSITION_CORRECTION
                for p in group:
                    alloc[id(p)] *= factor
            total = sum(alloc.values()) or 1.0
            for p in squad:
                alloc[id(p)] *= CLUB_MINUTES / total

        # 2.5 trust proven minutes -----------------------------------------------------
        # A stable, established starter has already earned his minutes at this club; lift
        # him back toward them where the re-allocation squeezed him below, then re-normalise
        # so the minutes come out of the uncertain depth around him (zero-sum preserved).
        lifted = False
        for p in squad:
            w = p.get("_role_stability", 0.0)
            proven = p.get("_proven_mpg", 0.0)
            if w > 0 and proven > alloc[id(p)]:
                alloc[id(p)] += w * (proven - alloc[id(p)])
                lifted = True
        if lifted:
            total = sum(alloc.values()) or 1.0
            for p in squad:
                alloc[id(p)] *= CLUB_MINUTES / total

        # 3. workload ceiling ----------------------------------------------------------
        # Compress the top of the rotation toward a realistic ceiling while keeping the
        # club on exactly 200 minutes.  Solving once for a single scale factor avoids
        # the trap of applying the squash repeatedly, which would drag the whole squad
        # down to the knee.
        vals = [alloc[id(p)] for p in squad]

        def total_at(scale: float) -> float:
            return sum(_soft_cap(v * scale) for v in vals)

        lo, hi = 1.0, 1.0
        if total_at(1.0) < CLUB_MINUTES:
            while total_at(hi) < CLUB_MINUTES and hi < 64.0:
                hi *= 2.0
        else:
            while total_at(lo) > CLUB_MINUTES and lo > 1e-3:
                lo /= 2.0
            hi = lo * 2.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if total_at(mid) < CLUB_MINUTES:
                lo = mid
            else:
                hi = mid
        scale = (lo + hi) / 2.0
        for p, v in zip(squad, vals):
            alloc[id(p)] = _soft_cap(v * scale)

        for p in squad:
            a = alloc[id(p)]
            p["mpg_proj"] = round(max(MPG_FLOOR, a), 2)
            p["mpg_prior"] = round(p["_mpg_prior"], 2)
            p["rotation_rank"] = squad.index(p) + 1
            # How much of his own prior workload the depth chart actually leaves him:
            # below 1.0 means the players around him are squeezing him out.
            p["role_squeeze"] = round(a / p["_mpg_prior"], 3) if p["_mpg_prior"] > 0.5 else None


# ======================================================================================
# Game-log derived form, variance and availability (step 5 inputs)
# ======================================================================================
def gamelog_profile(p: dict, hz: Horizon = None) -> dict:
    """Consistency, floor/ceiling and second-half form from last season's box scores."""
    hz = hz or DEFAULT_HORIZON
    logs: List[dict] = []
    for s in hz.logs:
        logs.extend((p.get("logs") or {}).get(s, []))
    played = [g for g in logs if not g["dnp"] and g["min"] > 0]
    out = {
        "games_listed": len(logs), "games_played": len(played),
        "availability": (len(played) / len(logs)) if logs else None,
        "pir_sd": None, "pir_mean": None, "floor": None, "ceiling": None,
        "consistency": None, "form_delta": None, "start_rate": None,
        "boom_rate": None, "bust_rate": None, "series": [],
    }
    if not played:
        return out
    pirs = [g["pir"] for g in played]
    out["pir_mean"] = round(sum(pirs) / len(pirs), 2)
    out["pir_sd"] = round(statistics.pstdev(pirs), 2) if len(pirs) > 1 else 0.0
    srt = sorted(pirs)

    def pct(q: float) -> float:
        if not srt:
            return 0.0
        i = min(len(srt) - 1, max(0, int(round(q * (len(srt) - 1)))))
        return srt[i]

    out["floor"] = round(pct(0.20), 1)
    out["ceiling"] = round(pct(0.80), 1)
    out["median"] = round(pct(0.50), 1)
    # Coefficient of variation, inverted: 1.0 = metronome, 0.0 = coin flip.
    if out["pir_mean"] and out["pir_mean"] > 0:
        cv = out["pir_sd"] / out["pir_mean"]
        out["consistency"] = round(max(0.0, min(1.0, 1.0 - cv / 1.6)), 3)
    out["start_rate"] = round(sum(1 for g in played if g["start"]) / len(played), 3)
    out["boom_rate"] = round(sum(1 for v in pirs if v >= 20) / len(pirs), 3)
    out["bust_rate"] = round(sum(1 for v in pirs if v <= 4) / len(pirs), 3)
    if len(played) >= 10:
        half = len(played) // 2
        first = sum(g["pir"] for g in played[:half]) / half
        second = sum(g["pir"] for g in played[half:]) / (len(played) - half)
        out["form_delta"] = round(second - first, 2)
    out["series"] = [
        {"r": g["round"], "pir": g["pir"], "min": round(g["min"], 1), "start": g["start"]}
        for g in logs
    ]
    return out


# ======================================================================================
# Team context
# ======================================================================================
def team_context(players: List[dict]) -> Dict[str, float]:
    """A small multiplier for club quality - good teams create more PIR to go around."""
    strength: Dict[str, float] = {}
    for p in players:
        club = p.get("club")
        if not club:
            continue
        strength.setdefault(club, 0.0)
        strength[club] += p["_rate"]["rate"] * p.get("mpg_proj", 0.0) / 40.0
    if not strength:
        return {}
    vals = sorted(strength.values())
    med = vals[len(vals) // 2] or 1.0
    # Cap the swing at +-5%: club quality matters, but far less than role does.
    return {c: max(0.95, min(1.05, 1.0 + 0.28 * (v / med - 1.0))) for c, v in strength.items()}


# ======================================================================================
# Top level
# ======================================================================================
def project_all(players: List[dict], hz: Horizon = None) -> List[dict]:
    hz = hz or DEFAULT_HORIZON
    priors = _positional_priors(players, hz)
    for p in players:
        p["_rate"] = project_rate(p, priors, hz)

    allocate_minutes(players, priors, hz)
    ctx = team_context(players)

    for p in players:
        rate = p["_rate"]
        prof = gamelog_profile(p, hz)
        p["profile"] = prof

        # mpg_proj is now minutes WHEN FIT (the depth-chart role); durability carries the
        # games he is expected to miss and is applied explicitly, exactly once.
        mpg_fit = p.get("mpg_proj", 0.0)
        tctx = ctx.get(p.get("club"), 1.0)
        avail = _durability(p, hz)
        base = rate["rate"] * mpg_fit / 40.0 * tctx           # points on a night he plays

        p["fp_raw"] = round(base, 3)
        p["fp"] = round(base * avail, 3)                      # expected points per team game
        p["durability"] = round(avail, 3)
        p["availability"] = round(avail, 3)                   # kept for existing consumers
        # Minutes on the night he actually suits up - the number a reader recognises.
        p["mpg_when_playing"] = round(mpg_fit, 2)
        p["rate_p40"] = round(rate["rate"], 2)
        p["sample_minutes"] = round(rate["eff_minutes"], 1)
        p["sample_leagues"] = rate["sample_leagues"]
        p["unknown"] = rate["unknown"]

        # ---- uncertainty -------------------------------------------------------------
        # Two independent sources: how noisy the player is game to game, and how little
        # we know about his true level (thin sample, league change, new club).
        n_games = max(1, prof.get("games_played") or 1)
        game_sd = prof.get("pir_sd")
        if game_sd is None or game_sd <= 0:
            game_sd = max(2.0, 0.75 * max(1.0, p["fp"]))
        sem = game_sd / math.sqrt(n_games)

        info = rate["eff_minutes"]
        thin = 1.0 / math.sqrt(1.0 + info / 300.0)      # -> 0 as the sample grows
        proj_cv = SIGMA_BASE + SIGMA_THIN_SLOPE * thin
        if p.get("new_to_team"):
            proj_cv += 0.06
        if "EL" not in (rate["sample_leagues"] or []):
            proj_cv += WEIGHTS.transition_sigma_bump
        if rate["unknown"]:
            proj_cv += 0.35

        sigma = math.sqrt((proj_cv * max(0.5, p["fp"])) ** 2 + sem ** 2)
        p["fp_sigma"] = round(sigma, 3)
        p["fp_floor"] = round(max(0.0, p["fp"] - 1.0 * sigma), 2)
        p["fp_ceiling"] = round(p["fp"] + 1.0 * sigma, 2)
        p["team_context"] = round(tctx, 3)

    for p in players:
        for k in ("_rate", "_claim", "_mpg_prior"):
            p.pop(k, None)
    return players
