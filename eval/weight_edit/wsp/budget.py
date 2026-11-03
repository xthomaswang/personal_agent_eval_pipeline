"""Forward-pass budget of an editing agent: state/budget/<arm>.json (remaining count) and <arm>.log (one line per pass)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import util
from .util import Refused


def _dir() -> Path:
    return util.state_dir() / "budget"


def account(arm: str) -> str:
    """The budget account an arm's passes are charged to: WSP_BUDGET_ACCOUNT when set (one shared budget for all of an
    editing agent's arms), otherwise the arm itself."""
    return os.environ.get("WSP_BUDGET_ACCOUNT") or arm


def budget_file(arm: str) -> Path:
    return _dir() / f"{account(arm)}.json"


def log_file(arm: str) -> Path:
    return _dir() / f"{account(arm)}.log"


def read(arm: str) -> dict[str, Any] | None:
    p = budget_file(arm)
    return json.loads(p.read_text()) if p.exists() else None


def set_budget(arm: str, n: int) -> dict[str, Any]:
    if n < 0:
        raise Refused("budget must be >= 0")
    _dir().mkdir(parents=True, exist_ok=True)
    b = {"arm": arm, "initial": n, "remaining": n, "set_at": util.now_iso()}
    budget_file(arm).write_text(json.dumps(b, indent=2))
    log(arm, {"event": "set", "remaining": n})
    return b


def check(arm: str) -> dict[str, Any]:
    """Raise Refused unless the arm has a budget with at least one pass left."""
    b = read(arm)
    if b is None:
        raise Refused(f"no budget for arm {arm!r}: the owner must run `wsp budget set {arm} N`")
    if b["remaining"] <= 0:
        raise Refused(f"budget of arm {arm!r} is used up (0 of {b['initial']} passes left)")
    return b


def spend(arm: str) -> int:
    b = check(arm)
    b["remaining"] -= 1
    budget_file(arm).write_text(json.dumps(b, indent=2))
    return int(b["remaining"])


def log(arm: str, entry: dict[str, Any]) -> None:
    _dir().mkdir(parents=True, exist_ok=True)
    with open(log_file(arm), "a") as f:
        f.write(json.dumps({"time": util.now_iso(), **entry}) + "\n")
