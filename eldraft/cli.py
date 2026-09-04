"""Command line interface.

    python -m eldraft.cli build                     # fetch + model everything
    python -m eldraft.cli draft --available a.xlsx  # who to pick next
    python -m eldraft.cli dashboard                 # write the HTML dashboard
    python -m eldraft.cli backtest                  # out-of-sample validation
    python -m eldraft.cli template                  # write a blank availability sheet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import fetch
from .config import OUT, RULES
from .optimize import Slots, robust_draft


def _fmt(rows: List[dict], cols: List[tuple]) -> str:
    head = "  ".join(name.ljust(w) for name, w, _ in cols)
    line = "-" * len(head)
    out = [head, line]
    for r in rows:
        out.append("  ".join(str(fn(r))[:w].ljust(w) for _, w, fn in cols))
    return "\n".join(out)


def _num(v, nd=2, dash="-"):
    return dash if v is None else "{:.{}f}".format(v, nd)


# ======================================================================================
def cmd_build(args) -> int:
    from .pipeline import build_board
    fetch.REFRESH = bool(args.refresh)
    if args.refresh_boxscores:
        from . import build as B
        B._collect_gamelogs.__defaults__ = (False,)
    board = build_board(rebuild=True)
    print("built {} players and {} coaches for {}".format(
        len(board["players"]), len(board["coaches"]), board["meta"]["target_season"]))
    print("-> data/board.json")
    return 0


def cmd_draft(args) -> int:
    from .pipeline import apply_availability, apply_roster, draft_pool, load_board
    board = load_board()
    report = apply_availability(board, args.available)
    if args.available:
        print("sheet: matched {} of {} rows, {} priced from the sheet".format(
            report["matched"], report["matched"] + len(report["unmatched"]), report["priced"]))
        if report["unmatched"]:
            print("  unmatched names (ignored): " + ", ".join(report["unmatched"][:12])
                  + (" ..." if len(report["unmatched"]) > 12 else ""))
    mine = apply_roster(board, args.roster)
    pool, mine, _ = draft_pool(board, include_coach=not args.no_coach)

    slots = Slots(coach=not args.no_coach)
    budget = args.budget if args.budget is not None else RULES.budget

    if mine:
        spent = sum(p["price"] for p in mine)
        print("\nalready on your roster ({} players, {:.1f} credits spent):".format(len(mine), spent))
        for p in sorted(mine, key=lambda x: -x["fp"]):
            print("  {:<26} {:<2} {:<16} {:>5.1f} cr   proj {:>5.2f}".format(
                p["name"], p["position"], (p.get("club_name") or "")[:16], p["price"], p["fp"]))
        need = dict(slots.caps)
        for p in mine:
            need[p["position"]] = max(0, need.get(p["position"], 0) - 1)
        print("  still to fill: " + ", ".join(
            "{}x{}".format(v, k) for k, v in need.items() if v) + "   |   {:.1f} credits left".format(budget - spent))

    res = robust_draft(pool, budget, slots, n_sims=args.sims, top_k=args.top)

    print("\nRECOMMENDED SQUAD  (expected {:.1f} fantasy points per round, {:.1f} credits)".format(
        res["squad_expected_value"], res["squad_cost"]))
    squad = sorted(res["squad"], key=lambda p: (p["position"] == "H", -p["fp"]))
    k = RULES.full_credit_slots
    outfield = [p for p in squad if p["position"] != "H"]
    for n, p in enumerate(squad):
        if p["position"] == "H":
            role = "coach"
        else:
            rank = outfield.index(p)
            role = "CAPTAIN" if rank == 0 else ("starter" if rank < k else "bench 50%")
        print("  {:<26} {:<2} {:<16} {:>5.1f} cr   proj {:>5.2f} +-{:<5.2f} {}".format(
            p["name"], p["position"], (p.get("club_name") or "")[:16], p["price"],
            p["fp"], p["fp_sigma"], role))

    print("\nDRAFT BOARD  (draft value = fantasy points per round gained by having him)")
    print(_fmt(res["ranking"], [
        ("player", 26, lambda r: r["name"]),
        ("pos", 3, lambda r: r["position"]),
        ("club", 16, lambda r: r["club"] or ""),
        ("cred", 5, lambda r: _num(r["price"], 1)),
        ("proj", 6, lambda r: _num(r["fp"])),
        ("+-", 5, lambda r: _num(r["fp_sigma"])),
        ("draft val", 9, lambda r: _num(r["draft_value"], 3)),
        ("pick%", 6, lambda r: _num((r["pick_rate"] or 0) * 100, 0)),
        ("in squad", 8, lambda r: "yes" if r["in_optimal"] else ""),
        ("would replace", 22, lambda r: r["replaces"] or ""),
    ]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({k: v for k, v in res.items() if k != "all_rows"}, fh, indent=1, default=str)
        print("\n-> {}".format(args.json))
    return 0


def cmd_dashboard(args) -> int:
    from .dashboard import build_dashboard
    from .pipeline import apply_availability, apply_roster, load_board
    board = load_board()
    apply_availability(board, args.available)
    apply_roster(board, args.roster)
    path = build_dashboard(board, out_path=args.out, sims=args.sims)
    print("-> {}".format(path))
    return 0


def cmd_live(args) -> int:
    from .live import build_live
    from .pipeline import apply_availability, apply_roster, load_board
    board = load_board()
    apply_availability(board, args.available)
    apply_roster(board, args.roster)
    path = build_live(board, out_path=args.out)
    print("-> {}".format(path))
    print("Open it in a browser (or: python -m http.server --directory out) and draft:")
    print("  click + Mine when you draft a player, Taken when a rival does; the engine "
          "recommends your next pick. Toggle Value engine / Lookahead (MCTS) on the right.")
    return 0


def cmd_strategy_backtest(args) -> int:
    from .strategy_backtest import run_report
    r = run_report(args.season)
    c, d = r["classic"], r["draft"]
    print("\nStrategy backtests - {} (uses last season's realised ability)".format(r["season_label"]))
    print("\nClassic (weekly salary-cap, credits drift each round):")
    print("  {:<30}{:>7}".format("Perfect hindsight (ceiling)", c["hindsight"]))
    print("  {:<30}{:>7}  {}% of ceiling, beats {}% of field".format("Engine", c["engine"], c["pct_of_ceiling"], c["beat_field_pct"]))
    print("  {:<30}{:>7}  transfers add {:+}".format("Hold (no transfers)", c["hold"], c["gain_transfers"]))
    print("  {:<30}{:>7}  injury handling adds {:+}".format("Injury-blind engine", c["no_injury"], c["gain_injury"]))
    print("  team value grew 100 -> {} credits over the season".format(c["worth_end"]))
    print("\nDraft (10-team snake, hold + injury waivers):")
    print("  engine finishes {} of 10 on average, wins {}%, top-3 {}%".format(d["avg_rank"], d["win_pct"], d["top3_pct"]))
    print("  engine {} vs field {}".format(d["engine_mean"], d["field_mean"]))
    print("\n-> out/strategy_backtest.json")
    return 0


def cmd_season_backtest(args) -> int:
    from .season_backtest import run_report
    r = run_report(args.season, n_field=args.field)
    T = r["totals"]
    print("\nSeason backtest — {} ({} regular-season rounds)".format(r["season_label"], r["rounds"]))
    print("  {:<34}{:>8}  {}".format("Perfect hindsight (ceiling)", T["hindsight"], "—"))
    print("  {:<34}{:>8}  engine captured {}% of the ceiling".format("Engine (update + transfers)", T["engine"], r["pct_of_ceiling"]))
    print("  {:<34}{:>8}  {} beats {}% of a {}-manager field".format("  field best", T["field_best"], "", r["field_beat_pct"], r["n_field"]))
    print("  {:<34}{:>8}".format("  field mean", T["field_mean"]))
    print("  {:<34}{:>8}  active transfers add {:+} pts ({:+}%)".format("Hold (no transfers)", T["hold"], r["gain_vs_hold"], r["gain_vs_hold_pct"]))
    print("  {:<34}{:>8}  updating beliefs adds {:+} pts".format("Belief never updates", T["no_update"], r["gain_vs_no_update"]))
    print("\n-> out/season_backtest.json")
    return 0


def cmd_backtest(args) -> int:
    from .backtest import run
    for season in args.seasons:
        r = run(season)
        m, b = r["metrics"], r["baselines"]["last season PIR/game"]
        print("\n{}  (n={}, model saw only {})".format(season, r["n"], ", ".join(r["history"])))
        print("  model     spearman {:.3f}  top-decile {:.3f}  MAE {:.3f}  1sd coverage {:.3f}"
              "  minutes-curve RMSE {:.2f}".format(
                  m["fp_spearman"], m["top_decile_hit_rate"], m["fp_mae"],
                  m["sigma_coverage_1sd"], m["minutes_curve_rmse"]))
        print("  baseline  spearman {:.3f}  top-decile {:.3f}  MAE {:.3f}"
              "   (predict last season's PIR/game)".format(
                  b["spearman"], b["top_decile_hit_rate"], b["mae"]))
    return 0


def cmd_template(args) -> int:
    """Write a spreadsheet pre-filled with every player, ready to edit."""
    from .pipeline import load_board
    from openpyxl import Workbook
    board = load_board()
    wb = Workbook()
    ws = wb.active
    ws.title = "availability"
    ws.append(["Player", "Position", "Team", "Credits", "Available", "Mine"])
    rows = sorted(board["players"], key=lambda p: -p.get("fp", 0))
    for p in rows:
        ws.append([p["name"], p["position"], p.get("club_name"), p.get("price"), "yes", "no"])
    for c in board.get("coaches", []):
        ws.append([c["name"], "H", c.get("club_name"), c.get("price"), "yes", "no"])
    for col, w in zip("ABCDEF", (28, 10, 22, 10, 11, 8)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    wb.save(args.out)
    print("-> {}  ({} players + {} coaches)".format(args.out, len(rows), len(board.get("coaches", []))))
    print("Edit the Available column (yes/no) and Credits to match the app, then run:")
    print("  python -m eldraft.cli draft --available {}".format(args.out))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="eldraft", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="fetch data and rebuild the model")
    b.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    b.add_argument("--refresh-boxscores", action="store_true",
                   help="also download any missing box scores (slow, rate limited)")
    b.set_defaults(fn=cmd_build)

    d = sub.add_parser("draft", help="recommend a squad and rank the board")
    d.add_argument("--available", help="spreadsheet of available players (.xlsx or .csv)")
    d.add_argument("--roster", help="players you already own (.json or one name per line)")
    d.add_argument("--budget", type=float, help="credits available (default {})".format(RULES.budget))
    d.add_argument("--sims", type=int, default=400, help="simulated seasons (default 400)")
    d.add_argument("--top", type=int, default=30, help="rows to show (default 30)")
    d.add_argument("--no-coach", action="store_true", help="ignore the head-coach slot")
    d.add_argument("--json", help="also write the result to this JSON file")
    d.set_defaults(fn=cmd_draft)

    s = sub.add_parser("dashboard", help="write the interactive HTML dashboard")
    s.add_argument("--available", help="spreadsheet of available players")
    s.add_argument("--roster", help="players you already own")
    s.add_argument("--sims", type=int, default=400)
    s.add_argument("--out", default=os.path.join(OUT, "dashboard.html"))
    s.set_defaults(fn=cmd_dashboard)

    lv = sub.add_parser("live", help="write the interactive live-draft room")
    lv.add_argument("--available", help="spreadsheet of available players")
    lv.add_argument("--roster", help="players you already own")
    lv.add_argument("--out", default=os.path.join(OUT, "live_draft.html"))
    lv.set_defaults(fn=cmd_live)

    dv = sub.add_parser("draft-eval", help="measure MCTS vs the greedy engine in simulated drafts")
    dv.add_argument("--drafts", type=int, default=20)
    dv.add_argument("--teams", type=int, default=8)
    dv.add_argument("--sims", type=int, default=150)
    dv.set_defaults(fn=lambda a: __import__("eldraft.evaluate_drafters", fromlist=["main"]).main(
        ["--drafts", str(a.drafts), "--teams", str(a.teams), "--sims", str(a.sims)]))

    sb = sub.add_parser("season-backtest", help="replay a season, out-of-sample projection")
    sb.add_argument("--season", default="E2025")
    sb.add_argument("--field", type=int, default=40)
    sb.set_defaults(fn=cmd_season_backtest)

    stb = sub.add_parser("strategy-backtest",
                         help="classic + draft backtests using last season's realised stats")
    stb.add_argument("--season", default="E2025")
    stb.set_defaults(fn=cmd_strategy_backtest)

    t = sub.add_parser("backtest", help="validate the projection out of sample")
    t.add_argument("--seasons", nargs="+", default=["E2025", "E2024"])
    t.set_defaults(fn=cmd_backtest)

    tm = sub.add_parser("template", help="write a blank availability spreadsheet")
    tm.add_argument("--out", default=os.path.join(OUT, "availability_template.xlsx"))
    tm.set_defaults(fn=cmd_template)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
