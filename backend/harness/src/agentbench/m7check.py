"""M7 tool-call correctness checks.

Success = the expected call(s) are present with the right name(s), the arguments parse
as a JSON object and validate against the tool's JSON schema, every argument matcher
holds, there is no spurious extra call (and no call at all when none is needed), and no
tool markup leaks into the visible content.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import jsonschema

ROOT = "/workspace/demo-app/"
_KEYS = {"command": "cmd", "meta": "cmd", "⌘": "cmd", "super": "cmd", "control": "ctrl", "option": "alt",
         "opt": "alt", "return": "enter", "escape": "esc"}


def _norm(v: Any, how: str | None) -> Any:
    if not isinstance(v, str) or not how:
        return v
    if how == "strip":
        return v.replace("\r\n", "\n").strip()
    if how == "ws":
        return " ".join(v.split())
    if how == "lower":
        return v.strip().lower()
    if how == "key":
        k = v.strip().lower()
        return _KEYS.get(k, k)
    if how == "path":
        p = v.strip()
        if p.startswith(ROOT):
            p = p[len(ROOT):]
        while p.startswith("./"):
            p = p[2:]
        p = re.sub(r"/+", "/", p)
        return p.rstrip("/") if len(p) > 1 else p
    raise ValueError(f"unknown normalization {how}")


def match_value(actual: Any, spec: Any, path: str = "") -> list[str]:
    """Returns a list of failure reasons (empty = match). A non-dict spec means equality."""
    if not isinstance(spec, dict) or not any(k in spec for k in (
            "eq", "contains", "icontains", "startswith", "endswith", "range", "in", "set_eq", "subset",
            "match_each", "absent", "regex")):
        return [] if actual == spec else [f"{path}: expected {spec!r}, got {actual!r}"]
    how = spec.get("norm")
    if spec.get("absent"):
        return [] if actual is None else [f"{path}: expected absent, got {actual!r}"]
    if actual is None:
        return [f"{path}: missing"]
    errs: list[str] = []
    a = _norm(actual, how)
    if "eq" in spec:
        e = _norm(spec["eq"], how)
        if isinstance(e, bool) or isinstance(a, bool):
            ok = a is e if isinstance(e, bool) else False
        elif isinstance(e, (int, float)) and isinstance(a, (int, float)):
            ok = float(a) == float(e)
        else:
            ok = a == e
        if not ok:
            errs.append(f"{path}: expected {e!r}, got {a!r}")
    for key, fn, label in (("contains", lambda x, y: y in x, "to contain"),
                           ("icontains", lambda x, y: y.lower() in x.lower(), "to contain (case-insensitive)"),
                           ("startswith", lambda x, y: x.startswith(y), "to start with"),
                           ("endswith", lambda x, y: x.endswith(y), "to end with"),
                           ("regex", lambda x, y: re.search(y, x) is not None, "to match")):
        if key in spec:
            if not isinstance(a, str) or not fn(a, spec[key]):
                errs.append(f"{path}: expected {a!r} {label} {spec[key]!r}")
    if "range" in spec:
        lo, hi = spec["range"]
        if not isinstance(a, (int, float)) or isinstance(a, bool) or not lo <= a <= hi:
            errs.append(f"{path}: expected a number in [{lo}, {hi}], got {a!r}")
    if "in" in spec and a not in [_norm(x, how) for x in spec["in"]]:
        errs.append(f"{path}: expected one of {spec['in']!r}, got {a!r}")
    if "set_eq" in spec:
        if not isinstance(actual, list):
            errs.append(f"{path}: expected a list, got {actual!r}")
        else:
            got = sorted(map(repr, (_norm(x, how) for x in actual)))
            want = sorted(map(repr, (_norm(x, how) for x in spec["set_eq"])))
            if got != want:
                errs.append(f"{path}: expected the items {spec['set_eq']!r}, got {actual!r}")
    if "subset" in spec:
        if not isinstance(actual, dict):
            errs.append(f"{path}: expected an object, got {actual!r}")
        else:
            for k, sub in spec["subset"].items():
                errs += match_value(actual.get(k), sub, f"{path}.{k}")
    if "match_each" in spec:
        if not isinstance(actual, list):
            errs.append(f"{path}: expected a list, got {actual!r}")
        else:
            wanted = spec["match_each"]
            if len(actual) != len(wanted):
                errs.append(f"{path}: expected {len(wanted)} items, got {len(actual)}")
            elif not any(all(not _match_obj(item, w, path) for item, w in zip(perm, wanted))
                         for perm in itertools.permutations(actual)):
                errs.append(f"{path}: items do not match {wanted!r}")
    return errs


def _match_obj(obj: Any, fields: dict[str, Any], path: str) -> list[str]:
    if not isinstance(obj, dict):
        return [f"{path}: expected an object, got {obj!r}"]
    errs: list[str] = []
    for k, sub in fields.items():
        errs += match_value(obj.get(k), sub, f"{path}.{k}")
    return errs


def point_check(args: dict[str, Any], spec: dict[str, Any], screen=(1280, 1024)) -> tuple[list[str], str | None]:
    """(x, y) inside the box, in screenshot pixels or in Qwen-VL style 0-1000 normalized units."""
    x, y = args.get(spec["x"]), args.get(spec["y"])
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y)):
        return [f"point: x/y missing or not numbers ({x!r}, {y!r})"], None
    x0, y0, x1, y1 = spec["box"]
    if x0 <= x <= x1 and y0 <= y <= y1:
        return [], "pixels"
    px, py = x / 1000 * screen[0], y / 1000 * screen[1]
    if x0 <= px <= x1 and y0 <= py <= y1:
        return [], "normalized_1000"
    return [f"point ({x}, {y}) is outside {spec['box']} (also as 0-1000 normalized: ({px:.0f}, {py:.0f}))"], None


def _check_call(actual: dict[str, Any], spec: dict[str, Any], schemas: dict[str, dict]) -> tuple[list[str], dict]:
    info: dict[str, Any] = {}
    errs: list[str] = []
    if actual.get("name") != spec["name"]:
        return [f"called {actual.get('name')!r} instead of {spec['name']!r}"], info
    args = actual.get("arguments_parsed")
    if not isinstance(args, dict):
        return [f"{spec['name']}: arguments are not a JSON object ({actual.get('arguments_error') or type(args).__name__})"], info
    schema = schemas.get(spec["name"])
    if schema is not None:
        v = jsonschema.Draft202012Validator(schema)
        for e in sorted(v.iter_errors(args), key=lambda e: list(e.path)):
            errs.append(f"{spec['name']}: schema: {'/'.join(map(str, e.path)) or '(root)'}: {e.message}")
    for k, sub in (spec.get("args") or {}).items():
        errs += match_value(args.get(k), sub, f"{spec['name']}.{k}")
    if spec.get("point_in_box"):
        perrs, conv = point_check(args, spec["point_in_box"])
        errs += perrs
        info["coordinates"] = conv
    return errs, info


def _match_set(calls: list[dict], specs: list[dict], schemas: dict) -> tuple[list[str], dict]:
    """Best one-to-one assignment of actual calls to expected specs."""
    best: tuple[list[str], dict] | None = None
    if len(calls) < len(specs):
        # still report per-spec reasons against whatever was called
        reasons = [f"expected {len(specs)} call(s) ({', '.join(s['name'] for s in specs)}), got {len(calls)}"]
        for s in specs:
            same = [c for c in calls if c.get("name") == s["name"]]
            if same:
                reasons += _check_call(same[0], s, schemas)[0]
        return reasons, {}
    for perm in itertools.permutations(range(len(calls)), len(specs)):
        errs: list[str] = []
        info: dict[str, Any] = {}
        for ci, s in zip(perm, specs):
            e, i = _check_call(calls[ci], s, schemas)
            errs += e
            info.update(i)
        if best is None or len(errs) < len(best[0]):
            best = (errs, info)
        if not errs:
            break
    assert best is not None
    return best


def check_case(case: dict[str, Any], tool_calls: list[dict[str, Any]], content: str,
               leaks: list[str], schemas: dict[str, dict]) -> dict[str, Any]:
    exp = case["expect"]
    reasons: list[str] = []
    info: dict[str, Any] = {}
    if leaks:
        reasons.append(f"tool markup leaked into content: {', '.join(leaks)}")
    if exp.get("no_call"):
        if tool_calls:
            reasons.append(f"spurious call(s) when none was needed: {', '.join(c.get('name', '?') for c in tool_calls)}")
        if not content.strip():
            reasons.append("no answer text")
    else:
        alternatives = exp.get("alternatives") or [exp["calls"]]
        results = [_match_set(tool_calls, specs, schemas) + (specs,) for specs in alternatives]
        errs, info, specs = min(results, key=lambda r: len(r[0]))
        reasons += errs
        if not tool_calls:
            reasons = [r for r in reasons if not r.startswith("expected ")] + ["no tool call"]
        elif len(tool_calls) > len(specs) and not exp.get("allow_extra_calls"):
            reasons.append(f"spurious extra call(s): expected {len(specs)}, got {len(tool_calls)}")
    return {"pass": not reasons, "reasons": reasons, **info}
