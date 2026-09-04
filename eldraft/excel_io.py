"""Reads the availability spreadsheet the app exports (or that you keep by hand).

Column names are not standardised across exports, so headers are matched fuzzily: any
sheet with a recognisable player-name column will load.  Everything else - price,
position, club, availability flag, whether the player is already on your roster - is
optional and filled in from the model when absent.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Tuple

from .names import NameIndex, clean, split_el_name

# candidate header substrings, most specific first
FIELD_PATTERNS: Dict[str, List[str]] = {
    "name":      ["player name", "full name", "player", "name", "jugador", "giocatore"],
    "price":     ["credit", "price", "cost", "value", "salary", "valore", "precio"],
    "position":  ["position", "pos", "role", "puesto", "ruolo"],
    "club":      ["team", "club", "squad", "equipo", "squadra"],
    "available": ["available", "availability", "free", "status", "drafted", "taken", "owned"],
    "mine":      ["mine", "my team", "my squad", "roster", "owned by me", "selected"],
}


def _match_field(header: str) -> Optional[str]:
    h = clean(header)
    if not h:
        return None
    for field, pats in FIELD_PATTERNS.items():
        for pat in pats:
            if pat in h:
                return field
    return None


def _rows_from_xlsx(path: str) -> List[List]:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    return [list(r) for r in ws.iter_rows(values_only=True)]


def _rows_from_csv(path: str) -> List[List]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        return [row for row in csv.reader(fh, dialect)]


def _find_header(rows: List[List]) -> Tuple[int, Dict[str, int]]:
    """Locate the header row - exports often carry a title banner above it."""
    best_i, best_map, best_score = 0, {}, -1
    for i, row in enumerate(rows[:15]):
        mapping: Dict[str, int] = {}
        for j, cell in enumerate(row):
            f = _match_field(str(cell or ""))
            if f and f not in mapping:
                mapping[f] = j
        score = len(mapping) + (3 if "name" in mapping else 0)
        if score > best_score:
            best_i, best_map, best_score = i, mapping, score
    return best_i, best_map


_TRUEISH = {"y", "yes", "true", "1", "available", "free", "ok", "si", "x"}
_FALSEISH = {"n", "no", "false", "0", "drafted", "taken", "unavailable", "owned", "out"}


def _as_bool(v, default: Optional[bool] = None) -> Optional[bool]:
    if v is None:
        return default
    s = clean(str(v))
    if not s:
        return default
    if s in _TRUEISH:
        return True
    if s in _FALSEISH:
        return False
    return default


def _as_price(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def read_availability(path: str) -> List[dict]:
    """-> [{name, price, position, club, available, mine}] with unknown fields as None."""
    ext = os.path.splitext(path)[1].lower()
    rows = _rows_from_xlsx(path) if ext in (".xlsx", ".xlsm", ".xltx") else _rows_from_csv(path)
    rows = [r for r in rows if any(c not in (None, "") for c in r)]
    if not rows:
        raise ValueError("{}: no data rows found".format(path))

    hdr_i, cols = _find_header(rows)
    if "name" not in cols:
        raise ValueError(
            "{}: could not find a player-name column. Headers seen: {}".format(
                path, [str(c) for c in rows[hdr_i]][:12]
            )
        )

    out: List[dict] = []
    for row in rows[hdr_i + 1:]:
        def cell(field):
            j = cols.get(field)
            return row[j] if (j is not None and j < len(row)) else None

        raw_name = cell("name")
        if raw_name is None or not str(raw_name).strip():
            continue
        name = str(raw_name).strip()
        if clean(name) in {"total", "totals"}:
            continue
        avail = _as_bool(cell("available"), default=None)
        # A column headed "drafted"/"taken" means the opposite of "available".
        hdr_txt = clean(str(rows[hdr_i][cols["available"]])) if "available" in cols else ""
        if avail is not None and any(w in hdr_txt for w in ("drafted", "taken", "owned")):
            avail = not avail
        out.append({
            "sheet_name": name,
            "price": _as_price(cell("price")),
            "position": (str(cell("position")).strip() if cell("position") else None),
            "club": (str(cell("club")).strip() if cell("club") else None),
            "available": avail,
            "mine": _as_bool(cell("mine"), default=False),
        })
    return out


def link_to_universe(sheet_rows: List[dict], players: List[dict],
                     coaches: Optional[List[dict]] = None) -> dict:
    """Fuzzy-join sheet rows onto real roster people.

    Returns matched pairs plus the rows we could not resolve, so the CLI can tell you
    exactly which names to fix rather than silently dropping them.
    """
    idx = NameIndex()
    for p in players:
        idx.add(p.get("first") or "", p.get("last") or "", p)
    for c in (coaches or []):
        idx.add(c.get("first") or "", c.get("last") or "", c)

    matched, unmatched = [], []
    for row in sheet_rows:
        first, last = split_el_name(row["sheet_name"])
        hit = idx.lookup(first, last, max_tier=2)
        if hit is None:
            unmatched.append(row)
        else:
            matched.append((row, hit))
    return {"matched": matched, "unmatched": unmatched}
