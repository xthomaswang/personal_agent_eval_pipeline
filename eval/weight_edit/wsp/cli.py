"""`wsp` command line. See README.md for what each command does and the rules for an editing agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import util
from .util import Refused


def agent_mode() -> bool:
    """WSP_AGENT_MODE=1 is set for an editing agent: no test suite, no unbudgeted model runs."""
    return os.environ.get("WSP_AGENT_MODE") == "1"


# --- commands ---


def cmd_copy(a: argparse.Namespace) -> int:
    from . import edits

    d = edits.copy_arm(a.arm)
    print(f"copied the original model to {d} (edit it with `wsp edit apply {a.arm} <edit.py>`)")
    return 0


def cmd_verify_original(a: argparse.Namespace) -> int:
    rec_path = util.state_dir() / "original_sha256.json"
    src = util.original_dir()
    current = {}
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_file() and rel.parts[0] != ".cache" and p.name != ".DS_Store":
            current[str(rel)] = {"sha256": util.sha256_file(p), "size": p.stat().st_size}
    if not rec_path.exists() or a.record:
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps({"recorded_at": util.now_iso(), "model_dir": str(src), "files": current}, indent=2))
        print(f"recorded {len(current)} file hashes in {rec_path}")
        print(f"model.safetensors sha256 {current['model.safetensors']['sha256']}")
        return 0
    rec = json.loads(rec_path.read_text())["files"]
    bad = [f for f in sorted(set(rec) | set(current)) if rec.get(f, {}).get("sha256") != current.get(f, {}).get("sha256")]
    print(f"model.safetensors sha256 now      {current['model.safetensors']['sha256']}")
    print(f"model.safetensors sha256 recorded {rec['model.safetensors']['sha256']}")
    print(f"{len(current)} files checked against {rec_path}")
    if bad:
        print(f"MISMATCH in: {bad}")
        return 1
    print("OK: the original model is unchanged")
    return 0


def _print_wall(t0: float) -> None:
    print(f"wall time: {time.time() - t0:.1f}s")


def cmd_eval(a: argparse.Namespace) -> int:
    from . import run

    t0 = time.time()
    if agent_mode() and (a.suite != "dev" or not a.post):
        raise Refused("agent mode: use `wsp pass eval` for model runs; plain `wsp eval` only does --suite dev --post")
    result = run.run_eval(a.arm, a.suite, device=a.device, limit=a.limit, dev_lme=a.dev_lme, post=Path(a.post) if a.post else None, label=a.label)
    print(run.format_table(result))
    _print_wall(t0)
    return 0


def cmd_postmap_fit(a: argparse.Namespace) -> int:
    from . import run

    path = run.fit_postmap(a.arm, a.kind, a.k, a.shrink, a.limit, a.dev_lme, Path(a.out) if a.out else None)
    print(f"post-map saved: {path}")
    print(f"use: wsp eval --arm {a.arm} --suite dev --post {path}")
    return 0


def cmd_acts(a: argparse.Namespace) -> int:
    from . import acts

    t0 = time.time()
    if agent_mode():
        raise Refused("agent mode: use `wsp pass acts`")
    util.check_device(a.device, a.limit)
    meta = acts.dump_acts(a.arm, acts.parse_layers(a.layers), a.limit, a.kind, a.device, a.dev_lme, a.suite)
    print(f"acts: arm={a.arm} texts={meta['n_texts']} depths={meta['layers']} -> {acts.acts_dir(a.arm)}")
    _print_wall(t0)
    return 0


def cmd_edit_apply(a: argparse.Namespace) -> int:
    from . import edits

    m = edits.apply_edit(a.arm, Path(a.edit_file), a.arg, replace=a.replace, acts_arm=a.acts_arm)
    print(f"edit applied to arm {a.arm}: {m['n_tensors_changed']} tensors, {m['n_entries_changed']} entries changed")
    for name, v in list(m["tensors"].items())[:6]:
        print(f"  {name} {v['shape']} changed {v['n_changed']}/{v['n_entries']}")
    if m["n_tensors_changed"] > 6:
        print(f"  ... {m['n_tensors_changed'] - 6} more (see edit_manifest.json)")
    print(f"edited model.safetensors sha256 {m['edited_safetensors_sha256']}")
    return 0


def cmd_budget(a: argparse.Namespace) -> int:
    from . import budget

    if a.action == "set":
        if a.n is None:
            raise Refused("usage: wsp budget set ARM N")
        b = budget.set_budget(a.arm, a.n)
        print(f"budget of {a.arm}: {b['remaining']} passes")
    else:
        b = budget.read(a.arm)
        print(json.dumps(b, indent=2) if b else f"no budget for {a.arm}")
        if b and budget.log_file(a.arm).exists():
            print(f"log: {budget.log_file(a.arm)}")
    return 0


def cmd_pass(a: argparse.Namespace) -> int:
    """A budgeted forward pass: dev suite only, one pass off the arm's budget, logged."""
    from . import acts, budget, edits, run

    t0 = time.time()
    if a.suite != "dev":
        raise Refused("`wsp pass` never runs the test suite")
    if not edits.arm_dir(a.arm).exists():
        raise Refused(f"unknown arm {a.arm!r}")
    budget.check(a.arm)
    util.check_device(a.device, a.limit)  # refused before a pass is spent
    if a.what == "acts" and not a.layers:
        raise Refused("`wsp pass acts` needs --layers, e.g. --layers 12,16,20")
    layers = acts.parse_layers(a.layers) if a.what == "acts" else None
    remaining = budget.spend(a.arm)
    entry = {"event": "pass", "what": a.what, "arm": a.arm, "suite": a.suite, "device": a.device, "limit": a.limit, "dev_lme": a.dev_lme, "remaining_after": remaining}
    try:
        if a.what == "eval":
            result = run.run_eval(a.arm, "dev", device=a.device, limit=a.limit, dev_lme=a.dev_lme)
            print(run.format_table(result))
            entry["metrics"] = {n: t["metrics"] for n, t in result["tasks"].items()}
        else:
            meta = acts.dump_acts(a.arm, layers or [], a.limit if a.limit is not None else 256, a.kind, a.device, a.dev_lme)
            print(f"acts: arm={a.arm} texts={meta['n_texts']} depths={meta['layers']} -> {acts.acts_dir(a.arm)}")
            entry["layers"] = layers
        entry["status"] = "ok"
    except BaseException as e:  # a failed pass still counts, and is logged
        entry["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        entry["wall_time_s"] = round(time.time() - t0, 2)
        budget.log(a.arm, entry)
        print(f"passes left for {a.arm}: {remaining}")
        _print_wall(t0)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from . import run

    if agent_mode() and a.suite != "dev":
        raise Refused("agent mode: compare only works on the dev suite")
    text, _ = run.compare(a.arm_a, a.arm_b, a.suite, a.limit, a.dev_lme, a.metric, a.boot)
    print(text)
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    from . import run

    t0 = time.time()
    st = run.suite_stats(a.suite, a.dev_lme)
    out = util.results_dir() / f"suite_stats.{a.suite}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st, indent=2))
    print(run.format_stats(st))
    print(f"saved {out}")
    _print_wall(t0)
    return 0


def cmd_check_split(a: argparse.Namespace) -> int:
    from . import suites

    suites.assert_split_disjoint(a.dev_lme)
    print("OK: dev and test LongMemEval ids are disjoint; dev avoids stage/pilot sets; SciFact train/test queries are disjoint")
    return 0


# --- parser ---


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wsp", description="Weight-space programming harness for Qwen3-Embedding-0.6B: edit a copy, measure retrieval.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_run_opts(sp: argparse.ArgumentParser, device_default: str = "mps") -> None:
        sp.add_argument("--device", choices=["mps", "cpu"], default=device_default, help="mps needs the booked GPU slot (or NIGHT_DEADLINE); cpu needs --limit <= 64")
        sp.add_argument("--limit", type=int, default=None, help="smoke mode: N queries per task, small corpora (gold + a few distractors)")
        sp.add_argument("--dev-lme", type=int, default=util.DEFAULT_DEV_LME, help="number of LongMemEval questions in the dev suite (default %(default)s)")

    s = sub.add_parser("copy", help="copy the original model to edits/<arm>/ (refuses an existing arm)")
    s.add_argument("arm")
    s.set_defaults(fn=cmd_copy)

    s = sub.add_parser("verify-original", help="print the original model's sha256 and compare with the recorded value")
    s.add_argument("--record", action="store_true", help="(re)record the hashes instead of comparing")
    s.set_defaults(fn=cmd_verify_original)

    s = sub.add_parser("eval", help="evaluate an arm on the dev or test suite; writes results/<arm>[+post]/<suite>.json")
    s.add_argument("--arm", required=True, help="base or an arm name")
    s.add_argument("--suite", choices=["dev", "test"], required=True)
    add_run_opts(s)
    s.add_argument("--post", help="post-map .npz: apply it to cached vectors (no model run)")
    s.add_argument("--label", help="result folder name (default: <arm> or <arm>+post)")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("postmap", help="closed-form post-maps")
    ps = s.add_subparsers(dest="pm_cmd", required=True)
    f = ps.add_parser("fit", help="fit on cached dev-suite document vectors")
    f.add_argument("--kind", choices=["pca-remove", "whiten"], required=True)
    f.add_argument("--k", type=int, default=1, help="pca-remove: number of top principal directions to remove")
    f.add_argument("--shrink", type=float, default=0.1, help="whiten: shrinkage toward isotropy in [0, 1]")
    f.add_argument("--arm", default="base")
    f.add_argument("--limit", type=int, default=None, help="use the cache of a --limit run")
    f.add_argument("--dev-lme", type=int, default=util.DEFAULT_DEV_LME)
    f.add_argument("--out", help="output .npz (default state/postmaps/...)")
    f.set_defaults(fn=cmd_postmap_fit)

    s = sub.add_parser("acts", help="dump last-token residual activations of dev texts to state/acts/<arm>/")
    s.add_argument("--arm", required=True)
    s.add_argument("--suite", choices=["dev"], default="dev")
    s.add_argument("--layers", required=True, help="depths 0..28 (k blocks applied), e.g. 12,16,20 or 8-28:4")
    s.add_argument("--kind", choices=["both", "queries", "docs"], default="both")
    s.add_argument("--limit", type=int, required=True, help="number of dev texts to sample")
    s.add_argument("--device", choices=["mps", "cpu"], default="mps")
    s.add_argument("--dev-lme", type=int, default=util.DEFAULT_DEV_LME)
    s.set_defaults(fn=cmd_acts)

    s = sub.add_parser("edit", help="edit the copy")
    es = s.add_subparsers(dest="edit_cmd", required=True)
    e = es.add_parser("apply", help="run edit(tensors, read_acts) from a python file and write the result into the arm")
    e.add_argument("arm")
    e.add_argument("edit_file")
    e.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE", help="passed to the edit file as the module-level dict ARGS (strings)")
    e.add_argument("--replace", action="store_true", help="redo the edit of an already edited arm from the original weights")
    e.add_argument("--acts-arm", default="base", help="arm whose dumped activations read_acts() returns by default")
    e.set_defaults(fn=cmd_edit_apply)

    s = sub.add_parser("budget", help="forward-pass budget of an arm (for `wsp pass`)")
    s.add_argument("action", choices=["set", "show"])
    s.add_argument("arm")
    s.add_argument("n", nargs="?", type=int)
    s.set_defaults(fn=cmd_budget)

    s = sub.add_parser("pass", help="budgeted forward pass for an editing agent: dev suite only, logged, refuses at 0")
    s.add_argument("what", choices=["eval", "acts"])
    s.add_argument("--arm", required=True)
    s.add_argument("--suite", default="dev", help="only dev; test is always refused")
    add_run_opts(s)
    s.add_argument("--layers", help="acts: depths, e.g. 12,16,20")
    s.add_argument("--kind", choices=["both", "queries", "docs"], default="both", help="acts: which dev texts")
    s.set_defaults(fn=cmd_pass)

    s = sub.add_parser("compare", help="paired per-query differences with a bootstrap 95%% CI (main metric per task)")
    s.add_argument("arm_a", help="result label, e.g. base")
    s.add_argument("arm_b", help="result label, e.g. myarm or myarm+post")
    s.add_argument("--suite", choices=["dev", "test"], required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--dev-lme", type=int, default=util.DEFAULT_DEV_LME)
    s.add_argument("--metric", help="metric name instead of each task's main metric (ndcg@10)")
    s.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("stats", help="suite sizes and token counts (tokenizer only, no model)")
    s.add_argument("--suite", choices=["dev", "test"], required=True)
    s.add_argument("--dev-lme", type=int, default=util.DEFAULT_DEV_LME)
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("check-split", help="assert that dev and test are disjoint")
    s.add_argument("--dev-lme", type=int, default=util.DEFAULT_DEV_LME)
    s.set_defaults(fn=cmd_check_split)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
