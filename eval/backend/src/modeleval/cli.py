"""`modeleval` command line (DESIGN.md §9).

  models                       the registry and what each model has
  download|convert|register M  prepare a model (DESIGN §4)
  plan M                       units, time estimates, what is already done
  run M                        the orchestrator (run.py)
  report                       rebuild RESULTS.md
  publish [M [RUN_ID]]         the results page's summary.json rows under docs/benchmark
  selftest                     offline test suite + path, contract and hygiene checks
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from modeleval import paths, prepare, registry, run


def _model(name: str) -> registry.Model:
    return registry.get(registry.load_models(), name)


def cmd_models(a: argparse.Namespace) -> int:
    models = registry.load_models()
    print(f"{'key':26s} {'served id':30s} {'quant':5s} {'spec':4s} {'vis':3s} {'ref':3s} "
          f"{'ckpt':4s} {'link':9s} {'source':6s} runs")
    for m in models.values():
        st = prepare.status(m)
        print(f"{m.key:26s} {m.served_id:30s} {m.quant:5s} {m.spec:4s} {'yes' if m.vision else 'no':3s} "
              f"{'yes' if m.reference else '':3s} {'ok' if st['checkpoint'] else 'NO':4s} {st['link']:9s} "
              f"{'cached' if st['source'] else '-':6s} {len(st['runs'])}" + (f" (latest {st['runs'][-1]})" if st['runs'] else ""))
    return 0


def cmd_download(a: argparse.Namespace) -> int:
    cmd = prepare.download(_model(a.model), a.min_free_gb, a.dry_run)
    if a.dry_run:
        print(" ".join(cmd))
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    m = _model(a.model)
    if a.dry_run:
        print(" ".join(paths.tilde(x) for x in prepare.convert_command(m, a.allow_proxy)))
        print(f"output ~{prepare.convert_checks(m) / 1e9:.1f} GB at {paths.tilde(m.path)}")
        return 0
    print(f"converted: {paths.tilde(prepare.convert(m, a.allow_proxy))}")
    return 0


def cmd_register(a: argparse.Namespace) -> int:
    link = prepare.register(_model(a.model))
    print(f"registered: {paths.tilde(link)} -> {paths.tilde(os.readlink(link))}")
    return 0


def _options(a: argparse.Namespace, dry: bool) -> run.Options:
    legs = tuple(x.strip() for x in a.legs.split(",") if x.strip())
    only = tuple(x.strip() for x in (a.only or "").split(",") if x.strip())
    harnesses = tuple(x.strip() for x in (a.harnesses or "").split(",") if x.strip())
    opts = run.Options(model=a.model, legs=legs, capability=a.capability, unverified=a.unverified, resume=a.resume,
                       notice_wait=getattr(a, "notice_wait", 600), settle_s=getattr(a, "settle", 600), dry_run=dry,
                       thinking=a.thinking, only=only, harnesses=harnesses, thinking_budget=a.thinking_budget)
    if opts.thinking_budget and opts.thinking != "on":
        raise SystemExit("--thinking-budget needs --thinking on")
    if opts.exploratory and (not opts.unverified or opts.legs != ("capability",)):
        raise SystemExit("--thinking on, --only and --harnesses are exploratory arms: "
                         "use them with --legs capability --unverified")
    return opts


def cmd_plan(a: argparse.Namespace) -> int:
    for line in run.plan_lines(_options(a, True)):
        print(line)
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    return run.run(_options(a, a.dry_run))


def cmd_report(a: argparse.Namespace) -> int:
    from modeleval.report import build_report
    out = paths.REPO / "RESULTS.md"
    out.write_text(build_report(paths.RESULTS), encoding="utf-8")
    print(f"wrote {paths.tilde(out)}")
    return 0


def cmd_publish(a: argparse.Namespace) -> int:
    from modeleval import publish
    model = registry.get(registry.load_models(), a.model).key if a.model else None
    lines = publish.publish(paths.RESULTS, model, a.run_id)
    for line in lines:
        print(line)
    if any(line.startswith("wrote ") for line in lines):
        print("next: python3 scripts/build_docs.py (at the repository root)")
    return 0 if any(line.startswith("wrote ") for line in lines) else 1


# ---------------------------------------------------------------- selftest


def _git(*args: str) -> str:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        return subprocess.run(["git", "-C", str(paths.REPO), *args], capture_output=True, text=True, env=env,
                              timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def pre_commit_installed() -> bool:
    """Whether this clone's pre-commit hook is scripts/pre-commit (DESIGN §8: results are committed only
    after redact.py --check passes), through the repository's hooks directory or core.hooksPath."""
    want = (paths.SCRIPTS / "pre-commit").resolve()
    hooks = _git("rev-parse", "--git-path", "hooks").strip()   # relative to the working directory (REPO)
    if not hooks:
        return False
    try:
        return (paths.REPO / hooks / "pre-commit").resolve() == want
    except OSError:
        return False


def hygiene_checks() -> list[tuple[str, str, str]]:
    """(level, check, detail) with level ok | WARN | FAIL. Read-only; peer repos only via rev-parse/status."""
    out: list[tuple[str, str, str]] = []

    def add(ok: bool, name: str, detail: str = "", warn: bool = False) -> None:
        out.append(("ok" if ok else ("WARN" if warn else "FAIL"), name, detail))

    add(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and os.environ.get("GIT_OPTIONAL_LOCKS") == "0",
        "PYTHONDONTWRITEBYTECODE=1 and GIT_OPTIONAL_LOCKS=0 exported")
    used = [f.name for f in paths.CONFIG_FILES if f.is_file()]
    add(True, "config", f"{' + '.join(used)}; machine {paths.machine_id()}; GPU-lock protocol "
        f"{'on' if paths.lock_enabled() else 'off'} ({paths.tilde(paths.GPU_LOCK)}); results "
        f"{paths.tilde(paths.RESULTS)}; ws_root {paths.tilde(paths.WS_ROOT)}; models {paths.tilde(paths.MODELS)}")
    add(paths.PORT not in paths.FOREIGN_PORTS, f"port {paths.PORT} is not another session's")
    add(os.access(paths.OMLX_EXE, os.X_OK), "oMLX executable", paths.tilde(paths.OMLX_EXE))
    add(paths.OMLX_PY.exists(), "oMLX interpreter", paths.tilde(paths.OMLX_PY))
    g = run.git_info(paths.OMLX_REPO)
    add(bool(g["sha"]) and g["sha"].startswith("7a5a3d75") and not g["dirty"], "oMLX at 7a5a3d75, clean",
        f"{g['sha']} dirty={g['dirty']}")
    add(not paths.BASE.is_symlink() and paths.BASE.parent.is_dir(), "base path is the pipeline's own",
        paths.tilde(paths.BASE))
    try:
        models = registry.load_models()
        add(True, "models.toml", f"{len(models)} models, reference {registry.reference(models).key}")
        for m in models.values():
            st = prepare.status(m)
            add(st["checkpoint"] and st["link"] == "ok", f"{m.key} checkpoint and link",
                f"ckpt={st['checkpoint']} link={st['link']}", warn=True)
    except (OSError, registry.RegistryError) as e:
        add(False, "models.toml", str(e)[:200])
    ql = (paths.SCRIPTS / "quietlib.py").read_text(encoding="utf-8")
    add("def judge_window(" in ql and "MONITOR_COLUMNS" in ql, "quietlib has judge_window and MONITOR_COLUMNS")
    lk = (paths.SCRIPTS / "gpu_lock.sh").read_text(encoding="utf-8")
    add("update-purpose" in lk and "MODELEVAL_RUN_ID" in lk, "gpu_lock.sh has update-purpose and run ids")
    oq = (paths.SCRIPTS / "oq_convert.py").read_text(encoding="utf-8")
    add("--no-register" in oq and ".bench-home" not in oq, "oq_convert.py has --no-register and never names .bench-home",
        "modeleval convert needs --no-register; settings belong to .modeleval-home, written per launch")
    hook = os.path.relpath(paths.SCRIPTS / "pre-commit", paths.MONOREPO)
    add(pre_commit_installed(), "pre-commit hook runs scripts/redact.py --check",
        f"install it once per clone, at the repository root: ln -s ../../{hook} .git/hooks/pre-commit", warn=True)
    epoch = registry.epoch_id(registry.load_pins())
    add(bool(epoch) and "PLACEHOLDER" not in epoch, "harness epoch in pins.toml", str(epoch), warn=True)
    add(run.CAP_PYTHON.exists(), "capability interpreter", paths.tilde(run.CAP_PYTHON), warn=True)
    try:
        run.load_allowed_env()
        add(True, "capability.runner.allowed_env importable")
    except Exception as e:  # noqa: BLE001
        add(False, "capability.runner.allowed_env importable", f"{type(e).__name__}: {e}"[:160], warn=True)
    gi = (paths.REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    need = ["runs/", "vendor/", "results/_unverified/", "*.log", "config.toml"]
    add(all(x in gi for x in need), ".gitignore keeps raw data out", ", ".join(x for x in need if x not in gi))
    files = [f for f in (_git("ls-files") + _git("ls-files", "--others", "--exclude-standard")).splitlines() if f]
    # the committed runs and their rows live in the repository's docs/benchmark (paths.RESULTS)
    files += [os.path.relpath(p, paths.REPO) for p in sorted(paths.RESULTS.rglob("*")) if p.is_file() or p.is_symlink()]
    leaks = []
    home_path = re.compile(re.escape(str(paths.HOME.parent)) + r"/[A-Za-z0-9._-]+/")   # an absolute home path
    for f in files:
        p = paths.REPO / f
        if p.is_symlink():
            leaks.append(f"{f} (symlink)")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if home_path.search(text):
            leaks.append(f)
    add(not leaks, "no absolute home paths or symlinks in committable files", ", ".join(leaks[:8]))
    return out


def cmd_selftest(a: argparse.Namespace) -> int:
    bad = 0
    for level, name, detail in hygiene_checks():
        bad += level == "FAIL"
        print(f"{level:4s} {name}" + (f": {detail}" if detail and level != "ok" else (f" ({detail})" if detail else "")))
    if not a.skip_tests:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        rc = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                            cwd=paths.REPO, env=env).returncode
        print(f"{'ok' if rc == 0 else 'FAIL'} offline test suite (pytest tests, exit {rc})")
        bad += rc != 0
    return 1 if bad else 0


# ---------------------------------------------------------------- argparse


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="modeleval", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="the registry and what each model has").set_defaults(fn=cmd_models)
    d = sub.add_parser("download", help="hf download of the pinned source revision (no lock)")
    d.add_argument("model")
    d.add_argument("--min-free-gb", type=float, default=100.0)
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(fn=cmd_download)
    c = sub.add_parser("convert", help="oQ conversion under the GPU lock, then register")
    c.add_argument("model")
    c.add_argument("--allow-proxy", action="store_true", help="accept oQ's 4-bit proxy sensitivity pass")
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(fn=cmd_convert)
    r = sub.add_parser("register", help="the models/hf-bench link")
    r.add_argument("model")
    r.set_defaults(fn=cmd_register)
    for name, fn, help_ in (("plan", cmd_plan, "what a run would do"), ("run", cmd_run, "run one model")):
        x = sub.add_parser(name, help=help_)
        x.add_argument("model")
        x.add_argument("--legs", default="speed,capability")
        x.add_argument("--capability", choices=run.CAPABILITY, default="full",
                       help="gate: the gate set only; full: then the ranking set unless the gate decision is "
                       "worse; full-force: the ranking set in any case")
        x.add_argument("--unverified", action="store_true", help="mechanics run: results/_unverified, no retries")
        x.add_argument("--resume", metavar="RUN_ID")
        x.add_argument("--thinking", choices=("off", "on"), default="off",
                       help="capability with thinking on (exploratory; needs --legs capability --unverified)")
        x.add_argument("--only", metavar="PATTERNS", help="comma-separated task-id patterns, e.g. aider/transpose")
        x.add_argument("--thinking-budget", type=int, default=0, metavar="TOKENS",
                       help="with --thinking on: oMLX ends each thinking block after this many tokens")
        x.add_argument("--harnesses", metavar="LIST", help="comma-separated subset of claude_code,hermes,localmelo")
        if name == "run":
            x.add_argument("--notice-wait", type=int, default=600, help="seconds between LOCK-NOTICE and the lock")
            x.add_argument("--settle", type=int, default=600, help="seconds a pre-unit check may wait for quiet")
            x.add_argument("--dry-run", action="store_true", help="print the plan; no lock, no server, no files")
        x.set_defaults(fn=fn)
    sub.add_parser("report", help="rebuild RESULTS.md").set_defaults(fn=cmd_report)
    pb = sub.add_parser("publish", help="summary.json rows for docs/benchmark (then run scripts/build_docs.py)")
    pb.add_argument("model", nargs="?", help="registry key or served id (default: every run of this pipeline)")
    pb.add_argument("run_id", nargs="?", help="one run of that model (default: all of its runs)")
    pb.set_defaults(fn=cmd_publish)
    s = sub.add_parser("selftest", help="offline tests and hygiene checks")
    s.add_argument("--skip-tests", action="store_true")
    s.set_defaults(fn=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    sys.dont_write_bytecode = True
    a = parser().parse_args(argv)
    try:
        return a.fn(a)
    except (registry.RegistryError, prepare.PrepareError, paths.ConfigError) as e:
        print(f"modeleval {a.cmd}: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
