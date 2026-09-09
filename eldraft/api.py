"""FastAPI service exposing the projection, optimisation and draft engines over HTTP.

Run locally:   uvicorn eldraft.api:app --reload   (after `eldraft build`)
Or in Docker:  see the Dockerfile — the image bakes a fresh board at build time.

The board (projections + prices) is loaded once at startup and treated as read-only;
every request builds its own pool from per-request availability/roster, so the service is
stateless and safe under concurrency.
"""
from __future__ import annotations

import copy
import json
import os
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - only hit without the [api] extra
    raise SystemExit("The API needs the 'api' extra: pip install -e '.[api]'") from exc

from .config import OUT, RULES
from .optimize import Slots, robust_draft

app = FastAPI(
    title="Draft Room API",
    version="0.1.0",
    description="Projections, an exact squad optimiser, and a draft-pick engine for the "
                "EuroLeague Fantasy Challenge.",
)

_BOARD = None


def board():
    """Lazy-load the modelled board once; 503 if it hasn't been built yet."""
    global _BOARD
    if _BOARD is None:
        from .pipeline import load_board
        try:
            _BOARD = load_board()
        except FileNotFoundError:
            raise HTTPException(503, "Board not built. Run `eldraft build` first.")
    return _BOARD


def _by_code():
    b = board()
    return {p["code"]: p for p in b["players"] + b.get("coaches", [])}


def _slim(p: dict) -> dict:
    r = p.get("ratings") or {}
    return {
        "code": p["code"], "name": p["name"], "position": p.get("position", "H"),
        "club": p.get("club_name") or p.get("club"),
        "price": p.get("price"), "projection": p.get("fp"), "sigma": p.get("fp_sigma"),
        "minutes": p.get("mpg_proj"), "rate_per40": p.get("rate_p40"),
        "overall": r.get("overall"), "value": r.get("value"),
        "unknown": bool(p.get("unknown")),
    }


# ======================================================================================
# Request models
# ======================================================================================
class OptimizeReq(BaseModel):
    budget: float = Field(default=RULES.budget, ge=40, le=200)
    available: Optional[List[str]] = Field(default=None, description="codes on the board; null = all")
    mine: List[str] = Field(default_factory=list, description="codes already owned (forced in)")
    coach: bool = True
    top: int = Field(default=25, ge=1, le=200)


class DraftReq(BaseModel):
    taken: List[str] = Field(default_factory=list)
    mine: List[str] = Field(default_factory=list)
    budget: float = Field(default=RULES.budget, ge=40, le=200)
    engine: str = Field(default="greedy", pattern="^(greedy|mcts)$")
    n_teams: int = Field(default=8, ge=2, le=16)
    my_slot: int = Field(default=1, ge=1, le=16)
    sims: int = Field(default=140, ge=30, le=600)
    top: int = Field(default=10, ge=1, le=50)


class WeeklyReq(BaseModel):
    squad: List[str] = Field(..., min_length=1, description="your current 10 outfield codes")
    cr: float = Field(default=RULES.budget, ge=40, le=200)
    round: int = Field(default=1, ge=1, le=45)
    max_transfers: int = Field(default=4, ge=0, le=6)
    excluded: List[str] = Field(default_factory=list, description="injured / unavailable codes")


# ======================================================================================
# Meta
# ======================================================================================
@app.get("/health")
def health():
    ok = os.path.exists(os.path.join(os.path.dirname(OUT), "data", "board.json")) or _BOARD is not None
    return {"status": "ok", "board_built": bool(ok)}


@app.get("/")
def root():
    b = board()
    return {
        "service": "Draft Room API", "version": "0.1.0",
        "season": b["meta"].get("target_season"),
        "players": len(b["players"]), "clubs": len(b.get("clubs", {})),
        "endpoints": ["/players", "/players/{code}", "/clubs", "/optimize",
                      "/draft/pick", "/weekly/transfers",
                      "/research/bakeoff", "/research/validation", "/research/strategy"],
    }


# ======================================================================================
# Data
# ======================================================================================
@app.get("/players")
def players(pos: Optional[str] = None, club: Optional[str] = None,
            q: Optional[str] = None, limit: int = 50, sort: str = "projection"):
    rows = [_slim(p) for p in board()["players"]]
    if pos:
        rows = [r for r in rows if r["position"] == pos.upper()]
    if club:
        rows = [r for r in rows if (r["club"] or "").lower() == club.lower()]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in r["name"].lower()]
    key = sort if sort in ("projection", "price", "overall", "value", "minutes") else "projection"
    rows.sort(key=lambda r: (r.get(key) is None, -(r.get(key) or 0)))
    return {"count": len(rows), "players": rows[:limit]}


@app.get("/players/{code}")
def player(code: str):
    p = _by_code().get(code)
    if not p:
        raise HTTPException(404, "unknown player code")
    out = _slim(p)
    out["ratings"] = p.get("ratings", {})
    out["history"] = {k: p.get(k) for k in ("el", "ec", "nba") if p.get(k)}
    out["profile"] = p.get("profile", {})
    return out


@app.get("/clubs")
def clubs():
    b = board()
    out = []
    for code, c in (b.get("clubs") or {}).items():
        squad = [p for p in b["players"] if p.get("club") == code]
        out.append({
            "code": code, "name": c.get("abbreviatedName") or c.get("name"),
            "strength": round(sum(p.get("fp") or 0 for p in squad), 1),
            "depth": [{"name": p["name"], "pos": p["position"], "minutes": p.get("mpg_proj"),
                       "projection": p.get("fp")}
                      for p in sorted(squad, key=lambda p: -(p.get("mpg_proj") or 0))[:13]],
        })
    out.sort(key=lambda x: -x["strength"])
    return {"clubs": out}


# ======================================================================================
# Optimisation
# ======================================================================================
def _pool_for(available, mine, coach):
    b = board()
    mine_set = set(mine or [])
    avail_set = set(available) if available is not None else None
    src = list(b["players"]) + (list(b.get("coaches", [])) if coach else [])
    pool = []
    for p in src:
        code = p["code"]
        is_mine = code in mine_set
        if not is_mine and avail_set is not None and code not in avail_set:
            continue
        q = copy.copy(p)
        if is_mine:
            q["mandatory"] = True
        pool.append(q)
    return pool


@app.post("/optimize")
def optimize(req: OptimizeReq):
    """Best legal salary-cap squad (structured scoring), plus the ranked draft board."""
    pool = _pool_for(req.available, req.mine, req.coach)
    if not pool:
        raise HTTPException(400, "empty pool — check `available`/`mine`")
    res = robust_draft(pool, req.budget, Slots(coach=req.coach), n_sims=250, top_k=req.top)
    return {
        "expected_points": res["squad_expected_value"],
        "cost": res["squad_cost"], "budget": req.budget,
        "squad": [_slim(p) for p in res["squad"]],
        "board": [{"name": r["name"], "position": r["position"], "club": r["club"],
                   "price": r["price"], "projection": r["fp"],
                   "draft_value": r["draft_value"], "in_squad": r["in_optimal"]}
                  for r in res["ranking"]],
    }


@app.post("/draft/pick")
def draft_pick(req: DraftReq):
    """Recommend the next pick given who's off the board — greedy (exact) or MCTS lookahead."""
    b = board()
    taken, mine = set(req.taken), set(req.mine)
    if req.engine == "greedy":
        pool = _pool_for([p["code"] for p in b["players"]
                          if p["code"] not in taken or p["code"] in mine],
                         list(mine), coach=False)
        res = robust_draft(pool, None, Slots(coach=False), n_sims=200, top_k=req.top)  # snake draft: no cap
        board_rows = [r for r in res["ranking"] if not r["mandatory"]]
        return {"engine": "greedy",
                "picks": [{"name": r["name"], "position": r["position"], "club": r["club"],
                           "price": r["price"], "projection": r["fp"],
                           "draft_value": r["draft_value"], "replaces": r.get("replaces")}
                         for r in board_rows[:req.top]]}
    # MCTS lookahead
    from .draft_sim import DraftUniverse
    from .mcts import MCTS
    players = [p for p in b["players"] if p.get("position") != "H"]
    # Snake draft: no salary cap, so zero prices and budget — only roster slots constrain.
    uni_players = [{"code": p["code"], "name": p["name"], "position": p["position"],
                    "price": 0.0, "fp": p["fp"], "club": p.get("club"),
                    "club_name": p.get("club_name"), "fp_sigma": p.get("fp_sigma")} for p in players]
    from .draft_sim import state_at_my_turn
    uni = DraftUniverse(uni_players, coach=False)
    seat = req.my_slot - 1
    taken_idx = [uni.by_code[c] for c in taken if c in uni.by_code]
    mine_idx = [uni.by_code[c] for c in mine if c in uni.by_code]
    state = state_at_my_turn(uni, req.n_teams, seat, mine_idx, taken_idx, budget=0.0)
    if state.done or state.teams[seat].needs.sum() == 0:
        return {"engine": "mcts", "picks": [], "note": "squad already complete"}
    rank = MCTS(uni, seat, req.n_teams, sims=req.sims).run(state)["ranking"]
    return {"engine": "mcts",
            "picks": [{"name": r["name"], "position": r["pos"], "club": r["club"],
                       "price": r["price"], "projection": r["fp"], "visits": r["visits"]}
                      for r in rank[:req.top]]}


@app.post("/weekly/transfers")
def weekly_transfers(req: WeeklyReq):
    """Best ≤N transfers for a round, matchup-adjusted, with the transfer-value frontier."""
    from . import weekly as W
    from .season_backtest import transfer_solve
    b = board()
    by = {p["code"]: p for p in b["players"]}
    for c in req.squad:
        if c not in by:
            raise HTTPException(400, "unknown squad code: " + c)
    wk = W.build_weekly()
    rnd_games = wk["schedule"].get(str(req.round), [])
    defense, lg = wk["defense"], wk["leagueAvg"]

    def wproj(p):
        return W.weekly_fp(p.get("fp") or 0.0, p.get("club"), rnd_games, defense, lg)

    excluded = set(req.excluded)
    cur = [c for c in req.squad if c not in excluded]
    seen, pool = set(), []

    def add(code):
        if code in seen or code not in by or code in excluded:
            return
        p = by[code]
        pool.append({"code": code, "pos": p["position"], "price": p["price"], "score": wproj(p)})
        seen.add(code)
    for c in cur:
        add(c)
    for pos in ("G", "F", "C"):
        cand = [p for p in b["players"] if p["position"] == pos and p["code"] not in excluded]
        for p in sorted(cand, key=lambda p: -wproj(p))[:22]:
            add(p["code"])
        for p in sorted(cand, key=lambda p: p["price"])[:10]:
            add(p["code"])
    frontier, squads = transfer_solve(pool, req.cr, set(cur), req.max_transfers, (4, 4, 2, 0), RULES)
    out = []
    base = frontier[0]
    for t, (v, sq) in enumerate(zip(frontier, squads)):
        moves = None
        if sq is not None:
            ins = [c for c in sq if c not in set(req.squad)]
            outs = [c for c in req.squad if c not in set(sq)]
            moves = {"out": [by[c]["name"] for c in outs if c in by],
                     "in": [by[c]["name"] for c in ins]}
        out.append({"transfers": t, "value": (round(v, 2) if v is not None else None),
                    "gain": (round(v - base, 2) if (v is not None and base is not None) else None),
                    "moves": moves})
    return {"round": req.round, "frontier": out}


# ======================================================================================
# Research results
# ======================================================================================
def _serve(name: str):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise HTTPException(404, name + " not generated yet")
    with open(path) as fh:
        return json.load(fh)


@app.get("/research/bakeoff")
def research_bakeoff():
    return _serve("bakeoff.json")


@app.get("/research/validation")
def research_validation():
    return _serve("validation.json")


@app.get("/research/strategy")
def research_strategy():
    return _serve("strategy_backtest.json")
