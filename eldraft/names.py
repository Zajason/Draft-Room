"""Name normalisation and cross-league matching.

EuroLeague feeds use "SURNAME, FIRSTNAME" (sometimes with accents or multi-part
surnames); ESPN uses "Firstname Lastname".  Matching them is the only genuinely fuzzy
join in the pipeline, so it is isolated here and kept conservative: we would much rather
miss an NBA link than attach the wrong player's numbers to a EuroLeague roster spot.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def clean(s: str) -> str:
    s = strip_accents(str(s or "")).lower()
    s = re.sub(r"[^a-z\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_el_name(raw: str) -> Tuple[str, str]:
    """'CORDINIER, ISAIA' -> ('isaia', 'cordinier').  Falls back gracefully.

    Without a comma we have to guess where the surname starts, and a generational
    suffix is the classic trap: "Wade Baldwin IV" would otherwise be read as a player
    whose surname is "IV".  Suffixes are therefore dropped before the split.
    """
    s = clean(raw)
    if "," in str(raw):
        surname, _, first = str(raw).partition(",")
        return clean(first), clean(surname)
    parts = [t for t in s.split() if t not in _SUFFIXES] or s.split()
    if len(parts) >= 2:
        return " ".join(parts[:-1]), parts[-1]
    return "", " ".join(parts)


def display_name(raw: str) -> str:
    """Human-friendly 'Isaia Cordinier' from the feed's 'CORDINIER, ISAIA'."""
    first, last = split_el_name(raw)
    def cap(x: str) -> str:
        return " ".join(
            "-".join(w.capitalize() for w in part.split("-")) for part in x.split()
        )
    return (cap(first) + " " + cap(last)).strip()


def _tokens(first: str, last: str) -> Tuple[str, List[str]]:
    lt = [t for t in last.replace("-", " ").split() if t not in _SUFFIXES]
    ft = [t for t in first.replace("-", " ").split() if t not in _SUFFIXES]
    return (" ".join(lt), ft)


def key_variants(first: str, last: str) -> List[str]:
    """Join keys ordered from strictest to loosest."""
    last_j, ft = _tokens(first, last)
    out = []
    if ft and last_j:
        out.append("{}|{}".format(ft[0], last_j))                 # first + full surname
        if len(last_j.split()) > 1:                               # last word of surname
            out.append("{}|{}".format(ft[0], last_j.split()[-1]))
        out.append("{}|{}".format(ft[0][0], last_j))              # initial + surname
    if last_j:
        out.append("|{}".format(last_j))                          # surname only (loosest)
    return out


class NameIndex:
    """Name lookup that tries strict keys before loose ones.

    Both sides of the join generate the same family of keys, and every key a record
    produces goes into one flat map.  Matching the *keys* rather than the tier they
    happened to be generated at is what lets "Fodzo Dada, Marc-Owen" find "Marc-Owen
    Fodzo Dada": the two spellings disagree about where the surname starts, so their
    strict keys differ, but their looser keys still meet.
    """

    def __init__(self):
        self.map: Dict[str, List[dict]] = {}

    def add(self, first: str, last: str, payload: dict) -> None:
        for k in key_variants(clean(first), clean(last)):
            bucket = self.map.setdefault(k, [])
            if payload not in bucket:
                bucket.append(payload)

    def lookup(self, first: str, last: str, max_tier: int = 2) -> Optional[dict]:
        for tier, k in enumerate(key_variants(clean(first), clean(last))):
            if tier > max_tier:
                break
            hits = self.map.get(k)
            # Ambiguous hits are dropped rather than guessed at - a wrong link would
            # silently attach another player's career to this roster spot.
            if hits and len(hits) == 1:
                return hits[0]
        return None
