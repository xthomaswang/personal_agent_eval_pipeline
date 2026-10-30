"""Navigation counts from a run's Chrome History database.

    cd eval
    uv run python agent_ability/web/history.py <History file> --t0 <epoch seconds> [--t-end <epoch seconds>]

A run keeps its ``Default/History`` (see chrome.py). This module copies it first (Chrome's own file is never opened),
then counts the top-level visits between t0 and the end of the run, the ends of redirect chains only:

- ``n_nav``: all such visits;
- ``n_search``: visits to a Google results page (``google.<tld>/search``);
- ``n_pages``: visits to pages that are not on a Google host;
- ``urls``: the visited URLs, in order.

A redirect chain (a short link, a Google ``/goto?`` hop) is one navigation: only its last visit counts. Visits in
sub-frames do not count. Google's own pages add a ``zx=<time>`` parameter to their address a moment after they load,
and Chrome records that as a second visit; such a visit is dropped when the page it came from has the same address.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Chrome stores times as microseconds since 1601-01-01 UTC (WebKit time).
WEBKIT_EPOCH_OFFSET_S = 11_644_473_600

# ui::PageTransition: the low byte is the core type, the rest are qualifier bits.
CORE_MASK = 0xFF
TRANSITION_AUTO_SUBFRAME = 3
TRANSITION_MANUAL_SUBFRAME = 4
CHAIN_END = 0x20000000
SUBFRAME_TRANSITIONS = (TRANSITION_AUTO_SUBFRAME, TRANSITION_MANUAL_SUBFRAME)

_GOOGLE_HOST = re.compile(r"(^|\.)google\.[a-z]{2,3}(\.[a-z]{2})?$")
_SEARCH_HOST = re.compile(r"^(www\.)?google\.[a-z]{2,3}(\.[a-z]{2})?$")


def webkit_time(epoch_s: float) -> int:
    """Seconds since the Unix epoch to Chrome's microseconds since 1601."""
    return int((epoch_s + WEBKIT_EPOCH_OFFSET_S) * 1_000_000)


def epoch_time(webkit_us: int) -> float:
    return webkit_us / 1_000_000 - WEBKIT_EPOCH_OFFSET_S


def is_google_host(host: str) -> bool:
    return bool(_GOOGLE_HOST.search(host.lower()))


def is_search_url(url: str) -> bool:
    """A Google results page: ``google.<tld>`` or ``www.google.<tld>`` and the path ``/search`` (not help.google.com)."""
    parts = urlsplit(url)
    return bool(_SEARCH_HOST.search((parts.hostname or "").lower())) and parts.path == "/search"


@dataclass
class Visit:
    t: float  # epoch seconds
    url: str
    transition: int  # the core type (0 link, 1 typed, 8 reload ...)


@dataclass
class HistorySummary:
    n_nav: int = 0
    n_search: int = 0
    n_pages: int = 0
    urls: list[str] = field(default_factory=list)
    visits: list[Visit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def copy_history(history: Path, dest_dir: Path) -> Path:
    """Copy a History file (and its journal, if any) into ``dest_dir``; the original is never opened."""
    dest = dest_dir / "History"
    shutil.copy2(history, dest)
    for suffix in ("-journal", "-wal", "-shm"):
        side = history.with_name(history.name + suffix)
        if side.exists():
            shutil.copy2(side, dest_dir / ("History" + suffix))
    return dest


def without_zx(url: str) -> str:
    """The URL without its ``zx`` query parameter (and the ``?`` if that leaves it empty)."""
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    if not any(key == "zx" for key, _ in pairs):
        return url
    return urlunsplit(parts._replace(query=urlencode([(k, v) for k, v in pairs if k != "zx"])))


def has_zx(url: str) -> bool:
    return any(key == "zx" for key, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True))


def read_visits(history: Path, t0: float, t_end: float | None = None) -> list[Visit]:
    """Top-level chain-end visits with ``t0 <= time`` (and ``<= t_end`` if given), oldest first."""
    history = Path(history)
    with tempfile.TemporaryDirectory(prefix="web-history-") as tmp:
        copy = copy_history(history, Path(tmp))
        connection = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        try:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(visits)")}
            opener = "visits.opener_visit" if "opener_visit" in columns else "0"
            rows = connection.execute(
                f"SELECT visits.id, visits.visit_time, urls.url, visits.transition, {opener} FROM visits "
                "JOIN urls ON urls.id = visits.url ORDER BY visits.visit_time, visits.id"
            ).fetchall()
        finally:
            connection.close()
    url_of = {visit_id: url for visit_id, _, url, _, _ in rows}  # every visit, also those before t0
    low, high = webkit_time(t0), webkit_time(t_end) if t_end is not None else 2**62
    visits = []
    for _, visit_time, url, transition, opener_id in rows:
        # transition is a signed 32-bit value in the database (the SERVER_REDIRECT bit makes it negative)
        if not low <= visit_time <= high or not (transition & CHAIN_END):
            continue
        if (transition & CORE_MASK) in SUBFRAME_TRANSITIONS:
            continue
        if opener_id and has_zx(url) and without_zx(url) == without_zx(url_of.get(opener_id, "")):
            continue  # Google's own zx rewrite of the page that opened it
        visits.append(Visit(round(epoch_time(visit_time), 6), url, transition & CORE_MASK))
    return visits


def summarize(history: Path, t0: float, t_end: float | None = None) -> HistorySummary:
    """n_nav, n_search, n_pages and the visited URLs between t0 and t_end (the end of the run)."""
    visits = read_visits(history, t0, t_end)
    summary = HistorySummary(visits=visits, urls=[v.url for v in visits])
    for visit in visits:
        summary.n_nav += 1
        if is_search_url(visit.url):
            summary.n_search += 1
        if not is_google_host(urlsplit(visit.url).hostname or ""):
            summary.n_pages += 1
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history", type=Path)
    parser.add_argument("--t0", type=float, required=True, help="start of the run, epoch seconds")
    parser.add_argument("--t-end", type=float, help="end of the run, epoch seconds")
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.history, args.t0, args.t_end).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
