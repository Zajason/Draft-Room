"""Game rules, data sources and tunable model weights for the EuroLeague Fantasy draft engine."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
OUT = os.path.join(ROOT, "out")
for _d in (DATA, RAW, OUT):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------------------
# Seasons
# --------------------------------------------------------------------------------------
# EuroLeague competition code "E", EuroCup "U".  Season code year = starting year, so
# E2026 is the 2026-27 campaign that we are drafting for and E2025 is "last season".
TARGET_SEASON = "E2026"

# (season_code, weight_label) ordered newest -> oldest.  Recency weights are applied in
# project.py; keeping the list here makes it trivial to extend the history window.
EL_HISTORY: List[str] = ["E2025", "E2024", "E2023"]
EC_HISTORY: List[str] = ["U2025", "U2024", "U2023"]

# ESPN encodes an NBA season by its *ending* year: 2026 == the 2025-26 season.
NBA_HISTORY: List[int] = [2026, 2025, 2024]

# Season used for per-game box score logs (variance / consistency / form modelling).
GAMELOG_SEASONS: List[str] = ["E2025"]

# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
FEEDS = "https://feeds.incrowdsports.com/provider/euroleague-feeds"
APIV2 = "https://api-live.euroleague.net/v2"
APIV3 = "https://api-live.euroleague.net/v3"
ESPN = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# --------------------------------------------------------------------------------------
# Fantasy game rules  (EuroLeague Fantasy Challenge, Classic mode)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GameRules:
    budget: float = 100.0
    slots: Dict[str, int] = field(default_factory=lambda: {"G": 4, "F": 4, "C": 2})
    squad_size: int = 10
    # Starting five + sixth man score 100%; the remaining four bench slots score 50%.
    full_credit_slots: int = 6
    bench_multiplier: float = 0.5
    # The official app has advertised both 1.5x and 2x for the captain across seasons.
    # 2x matches the current rulebook; override on the CLI if the app says otherwise.
    captain_multiplier: float = 2.0
    price_min: float = 4.0
    price_max: float = 16.0
    # Trades available between rounds (4 players keeping the coach, or 3 + a coach swap).
    trades_per_round: int = 4

    @property
    def bench_slots(self) -> int:
        return self.squad_size - self.full_credit_slots

    def position_order(self) -> List[str]:
        return ["G", "F", "C"]


RULES = GameRules()

# --------------------------------------------------------------------------------------
# Projection model weights
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelWeights:
    # Recency decay applied to season n-back (index 0 == most recent completed season).
    season_decay: Tuple[float, ...] = (1.0, 0.55, 0.28)

    # How much a minute of competition X is worth relative to a EuroLeague minute, and
    # how much we trust the *rate* stats produced there.  EuroCup is a clear step down;
    # the NBA is a stronger league but the role/usage translation is noisy, so its
    # reliability is discounted even though its difficulty multiplier is above 1.
    league_strength: Dict[str, float] = field(
        default_factory=lambda: {"EL": 1.00, "EC": 0.78, "NBA": 1.06}
    )
    league_reliability: Dict[str, float] = field(
        default_factory=lambda: {"EL": 1.00, "EC": 0.70, "NBA": 0.55}
    )

    # Empirical-Bayes shrinkage: prior strength expressed in "equivalent minutes played".
    # A player with 600 EuroLeague minutes is weighted 600/(600+K) toward his own rate.
    shrink_minutes_k: float = 420.0
    # Same idea for the minutes-per-game projection.
    shrink_games_k: float = 12.0

    # Age curve: peak age and the penalty per year away from it.
    peak_age: float = 27.0
    age_rise_per_year: float = 0.022   # improvement per year below peak
    age_decline_per_year: float = 0.030  # decline per year above peak

    # Monte-Carlo settings for the robust optimiser.
    n_sims: int = 4000
    # Extra variance applied to players changing league / with thin samples.
    transition_sigma_bump: float = 0.28


WEIGHTS = ModelWeights()

# --------------------------------------------------------------------------------------
# Position handling
# --------------------------------------------------------------------------------------
# The feeds API exposes positionName in {Guard, Forward, Center, ...}; fantasy uses G/F/C.
POSITION_MAP = {
    "Guard": "G",
    "Forward": "F",
    "Center": "C",
    "Guard-Forward": "G",
    "Forward-Guard": "F",
    "Forward-Center": "F",
    "Center-Forward": "C",
    "Point Guard": "G",
    "Shooting Guard": "G",
    "Small Forward": "F",
    "Power Forward": "F",
    "PG": "G", "SG": "G", "SF": "F", "PF": "F", "G": "G", "F": "F", "C": "C",
    # The head coach occupies his own roster slot, so he needs his own code.
    "H": "H", "Head Coach": "H", "Coach": "H", "HC": "H",
}


def norm_position(raw) -> str:
    if raw is None:
        return "F"
    s = str(raw).strip()
    return POSITION_MAP.get(s, POSITION_MAP.get(s.title(), "F"))
