"""The orchestrator: one run of one model, speed leg then capability leg (DESIGN.md §5, §6.7).

Order: LOCK-NOTICE and a 10-minute wait, the GPU lock (purpose `... speed (timed)`), the
pre-launch check, one oMLX launch with memory sampler and quiet monitor, the functional block
(stops the run on a gate FAIL), the timed units (wait until no major condition holds, run, judge:
keep a clean or minor attempt, move a major one to backend/results/_unverified and repeat, at most 3
attempts; DESIGN §5.4), QUIET-WINDOW-END (lock purpose rewritten to
untimed), then the capability runner as a subprocess on the same launch: the gate set first, then
the ranking set unless the gate set is "worse" than the reference's in the same harness epoch
(DESIGN §6.3, §6.6; `--capability full-force` runs it anyway).

Every boundary applies the user's battery rule and the hold file. A battery pause gives the machine
back completely (server stopped, lock released) and comes back as a new numbered launch. Exit codes
for the driving session: 0 complete, 1 error, 3 gate FAIL, 4 stopped or incomplete (`--resume` goes
on), 5 refused before anything was measured, 130 interrupted. The run never messages other sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from modeleval import paths, prepare, progress, quiet, registry
from modeleval.registry import Model
from modeleval.server import (
    MemorySampler,
    Pidfile,
    Server,
    ServerError,
    descendants,
    kill_group,
    model_settings,
    port_listening,
    wait_port_free,
)

CAP_PYTHON = paths.REPO / "envs/capability/.venv/bin/python"   # the pipeline's Python 3.13 environment
MAX_ATTEMPTS = 3              # a unit with a major disturbance is repeated at most twice (DESIGN §5.4)
# Attempt and unit verdicts that keep the result (DESIGN §5.4, the user's instruction of 2026-10-26:
# small disturbances have little effect, skip them, no re-run): "minor" keeps it with its flags.
KEPT = ("clean", "minor")
MAX_SERVER_LOST = 3           # relaunches after a lost server (any leg) before the run gives up
COLD_CACHE_LIMIT = 256        # one cache block: a cold request must not reuse more (DESIGN §5.1)
SSD_CACHE_BYTES = 128e9       # cache.ssd_cache_max_size (DESIGN §2): free disk a launch needs, + prepare.DISK_MARGIN
CAPABILITY = ("gate", "full", "full-force")   # full: ranking after a gate that is not "worse"; full-force: always
POLL_S = 60.0                 # hold, battery and lock polling
LOCK_WAIT_S = 2 * 3600        # how long to wait for a lock another session holds
CAFFEINATE = True
EXIT = {"complete": 0, "error": 1, "gate-fail": 3, "stopped": 4, "incomplete": 4, "refused": 5, "interrupted": 130}
LEGS = ("speed", "capability")


class Interrupted(BaseException):
    """SIGINT/SIGTERM/SIGHUP: a BaseException, so no `except Exception` inside a unit swallows it."""


class Stop(Exception):
    """End the run early with a status (see EXIT) and a reason for the RUN-DONE line."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


def sleep(seconds: float) -> None:
    """Every wait of the orchestrator goes through here (tests replace it)."""
    time.sleep(seconds)


def load_speed():
    from modeleval import speed
    return speed


def load_allowed_env():
    from modeleval.capability.runner import allowed_env
    return allowed_env


def ensure_epoch() -> str:
    """vendor/<epoch> for the epoch pins.toml names. vendor/ is gitignored, so a fresh clone has none and
    the runner would refuse every run; it is recreated with `git archive` from the pinned commits (only
    the git commands DESIGN §6.1 allows). EpochError when the pinned inputs no longer hash to the
    pinned id: a new epoch is a deliberate edit of pins.toml, never a side effect of a run."""
    from modeleval.capability import epoch
    return epoch.ensure_epoch(paths.REPO / "pins.toml", paths.VENDOR)


def missing_prompts(speed, model: Model) -> list[str]:
    """Generated fixture prompts the speed leg needs that are not built (fixtures/generated/ is
    gitignored). Found before the lock, not by the functional block after a 10-minute notice and a
    launch; a speed module without `required_prompts` (a test fake) needs none."""
    need = getattr(speed, "required_prompts", None)
    if need is None:
        return []
    root = Path(getattr(speed, "FIXTURES_DIR", paths.FIXTURES)) / "generated" / "prompts"
    return [p for p in need(model) if not (root / f"{p}.json").exists()]


def results_root(model: Model, unverified: bool = False) -> Path:
    """The directory a model's run directories go into: docs/benchmark/<family>/<quant_dir>/backend for a
    verified run (committed after the redaction check), backend/results/_unverified/<model> otherwise."""
    return paths.UNVERIFIED / model.key if unverified else model.results_dir


def capability_command(spec: Path) -> list[str]:
    return [str(CAP_PYTHON), "-I", "-B", "-m", "modeleval.capability.runner", "--spec", str(spec)]


def runner_path(path: str) -> str:
    """PATH for the runner without this repo's own .venv/bin, which `uv run` puts first: the agents
    inherit the runner's PATH behind their shims and tools, and must not find modeleval, agentbench or
    the pipeline's Python 3.12 there."""
    own = str(paths.REPO / ".venv")
    return os.pathsep.join(p for p in path.split(os.pathsep) if p and not (p == own or p.startswith(own + os.sep)))


@dataclass
class Options:
    model: str
    legs: tuple[str, ...] = LEGS
    capability: str = "full"          # "gate" | "full" | "full-force" (CAPABILITY)
    unverified: bool = False
    resume: str | None = None
    notice_wait: int = 600
    settle_s: int = 600
    dry_run: bool = False
    # Exploratory capability arms (DESIGN §6.8): thinking on, and a task/harness subset. Any of these
    # makes a run exploratory: its results stay in results/_unverified and out of RESULTS.md.
    thinking: str = "off"                 # "off" | "on"
    only: tuple[str, ...] = ()            # fnmatch patterns on task ids, e.g. "aider/transpose"
    harnesses: tuple[str, ...] = ()       # subset of claude_code, hermes, localmelo
    thinking_budget: int = 0              # > 0: oMLX's per-model thinking budget, in tokens (needs thinking on)

    @property
    def exploratory(self) -> bool:
        return self.thinking != "off" or bool(self.only) or bool(self.harnesses) or self.thinking_budget > 0


# ---------------------------------------------------------------- small helpers


def _iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() if t is None else t))


def _tilde_tree(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _tilde_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tilde_tree(v) for v in obj]
    if isinstance(obj, Path):
        return paths.tilde(obj)
    if isinstance(obj, str) and str(paths.HOME) in obj:
        return obj.replace(str(paths.HOME), "~")
    return obj


def write_json(path: Path, data: object) -> None:
    """Atomic JSON write with every home path as ~ (manifest.json and units.json are committed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_tilde_tree(data), indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _plain(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _field(res: object, key: str) -> object:
    return res.get(key) if isinstance(res, dict) else getattr(res, key, None)


def _failing_detail(res: object) -> str:
    """The first failing functional item and its detail, for the RUN-DONE reason."""
    items = _field(res, "items") or {}
    for gid, it in (items.items() if isinstance(items, dict) else []):
        if isinstance(it, dict) and str(it.get("result")).upper() == "FAIL":
            return f"{gid}: {it.get('detail') or 'FAIL'}"[:200]
    return ""


def functional_passed(res: object) -> bool:
    """FunctionalResult's verdict: `passed` (or `ok`) as a bool, else `verdict == "PASS"`."""
    get = res.get if isinstance(res, dict) else (lambda k: getattr(res, k, None))
    for k in ("passed", "ok"):
        if isinstance(get(k), bool):
            return get(k)
    if isinstance(get("verdict"), str):
        return get("verdict").upper() == "PASS"
    raise TypeError("FunctionalResult has no passed/ok/verdict field")


def cold_ok(res) -> tuple[bool, str, str | None]:
    """(valid, note, kind) of one attempt. kind "cache-hit": a cold request reused >= one cache
    block, which repeating cannot fix, so the unit is final ("invalid"); kind "incomplete": a
    missing latency sample, a trial error or a lost server, which a repeat can fix, so the attempt
    is retried like a disturbed one."""
    v = (res.metrics or {}).get("cold_cached_tokens")
    vals = v if isinstance(v, list) else [v]
    bad = [x for x in vals if isinstance(x, (int, float)) and x >= COLD_CACHE_LIMIT]
    if bad:
        note = f"cold request reused the cache (cached_tokens {bad} >= {COLD_CACHE_LIMIT})"
        return False, note + (f"; {res.note}" if res.note and not res.valid else ""), "cache-hit"
    if not res.valid:
        kind = (res.metrics or {}).get("invalid_kind")
        return False, res.note or "invalid", "cache-hit" if kind == "cache-hit" else "incomplete"
    return True, res.note or "", None


def attempt_verdict(v: quiet.UnitVerdict, valid: bool, why: str | None) -> str:
    """One attempt's verdict: "invalid" (a cold request hit the cache) or "incomplete" when the
    attempt itself is not valid; else "disturbed" for a major quiet verdict (repeated), "minor" for
    minor flags only (kept with them) or "clean"."""
    if not valid:
        return "invalid" if why == "cache-hit" else "incomplete"
    return "disturbed" if v.major else "minor" if v.severity == "minor" else "clean"


def final_verdict(verdicts: list[str]) -> str:
    """A unit's verdict after MAX_ATTEMPTS attempts without a kept (clean or minor) one."""
    kinds = set(verdicts)
    what = "disturbed" if kinds == {"disturbed"} else "incomplete" if kinds == {"incomplete"} else "disturbed/incomplete"
    return f"{what} x{len(verdicts)}"


def git_info(repo: Path) -> dict:
    """SHA and dirty flag with the only git commands allowed on peer repos (DESIGN §6.1). Untracked
    files count as dirty, except the pipeline's own .modeleval-home inside the oMLX checkout
    (paths.status_pathspec), which every launch creates."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, env=env,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    sha = git("rev-parse", "HEAD")
    status = git("--no-optional-locks", "status", "--porcelain", *paths.status_pathspec(repo)) if sha else ""
    return {"path": paths.tilde(repo), "sha": sha or None, "dirty": bool(status)}


def builds(machine: dict | None) -> dict:
    """(macOS build, Xcode build) of a machine_info(): the speed comparability key (DESIGN §2)."""
    m = machine or {}
    xc = re.search(r"Build version\s+(\S+)", str(m.get("xcodebuild") or ""))
    return {"macos": (m.get("sw_vers") or {}).get("BuildVersion"), "xcode": xc.group(1) if xc else None}


def machine_info() -> dict:
    """sw_vers and the global Xcode (DEVELOPER_DIR dropped): part of the speed comparability key."""
    def sh(*cmd: str, env: dict | None = None) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    sw = dict(line.split(":", 1) for line in sh("sw_vers").splitlines() if ":" in line)
    env = {k: v for k, v in os.environ.items() if k != "DEVELOPER_DIR"}
    return {"sw_vers": {k.strip(): v.strip() for k, v in sw.items()},
            "xcodebuild": sh("xcodebuild", "-version", env=env).replace("\n", "; "),
            "hw_model": sh("sysctl", "-n", "hw.model"), "memsize": sh("sysctl", "-n", "hw.memsize")}


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------- estimates and plan


REF_WEIGHT_GB = 19.8   # Qwen3.8-27B oQ5: quiet window ~30 min, gate ~1 h, ranking ~1.5-2.5 h (DESIGN §9)


def weight_bytes(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))
    except OSError:
        return 0


def model_factor(model: Model) -> float:
    """Rough time factor against the reference: weight bytes, x1.6 without speculative decoding;
    MoE checkpoints (active weights << total) get a flat 1.5 until their own numbers exist."""
    try:
        cfg = json.loads((model.path / "config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    text = cfg.get("text_config", cfg)
    if any(k in text for k in ("num_experts", "n_routed_experts", "num_local_experts")):
        f = 1.5
    else:
        f = (weight_bytes(model.path) / 1e9 or REF_WEIGHT_GB) / REF_WEIGHT_GB
    return f * (1.6 if model.spec == "none" else 1.0)


def measured_quiet_minutes(model: Model) -> float | None:
    """Quiet-window minutes from the newest run with every unit kept (+15 % for checks and gaps)."""
    for units_json in sorted(model.results_dir.glob("*/speed/units.json"), reverse=True):
        try:
            units = json.loads(units_json.read_text())["units"]
        except (OSError, ValueError, KeyError):
            continue
        kept = [a for u in units.values() if isinstance(u, dict) and u.get("verdict") in KEPT
                for a in u["attempts"] if a.get("kept")]
        if kept and len(kept) == len(units):
            return round(sum(a["t_end"] - a["t_start"] for a in kept) / 60 * 1.15, 1)
    return None


def estimate(model: Model) -> dict:
    f = model_factor(model)
    est = {"factor": round(f, 2), "functional_min": round(4 * f), "quiet_min": round(30 * f),
           "gate_min": round(60 * f), "ranking_min": round(120 * f), "source": "registry (rough)"}
    m = measured_quiet_minutes(model)
    if m:
        est.update(quiet_min=m, source="measured quiet window of the previous run; capability rough")
    return est


def capability_minutes(est: dict, which: str) -> int:
    """Upper bound: with `full` the ranking set is skipped after a "worse" gate."""
    return int(est["gate_min"] + (est["ranking_min"] if which != "gate" else 0))


def ssd_caches(exclude: Path | None = None) -> list[tuple[Path, int]]:
    """Leftover runs/<model>/<run-id>/ssd-cache directories (a crashed run leaves one; oMLX fills each
    up to 128 GB), with their sizes."""
    out = []
    for d in sorted(paths.RUNS.glob("*/*/ssd-cache")):
        if d.is_dir() and not d.is_symlink() and d != exclude:
            out.append((d, dir_bytes(d)))
    return out


# ---------------------------------------------------------------- the run


class Run:
    def __init__(self, opts: Options, models: dict[str, Model] | None = None):
        self.opts = opts
        self.models = models or registry.load_models()
        self.model = registry.get(self.models, opts.model)
        if opts.thinking_budget:
            # oMLX ends the thinking block after this many tokens (model_settings thinking_budget_*): the
            # only way to keep a thinking-on agent step bounded on a slow model (DESIGN §6.8).
            self.model = dataclasses.replace(self.model, model_settings={
                **self.model.model_settings, "thinking_budget_enabled": True,
                "thinking_budget_tokens": int(opts.thinking_budget)})
            self.models = {**self.models, self.model.key: self.model}
        bad = [x for x in opts.legs if x not in LEGS]
        if bad or not opts.legs:
            raise Stop("refused", f"--legs must be a subset of {','.join(LEGS)}, got {opts.legs}")
        if opts.capability not in CAPABILITY:
            raise Stop("refused", f"--capability must be one of {', '.join(CAPABILITY)}")
        self.run_id = opts.resume or time.strftime("%Y%m%d-%H%M%S")
        self.results = results_root(self.model, opts.unverified) / self.run_id
        self.runs = paths.RUNS / self.model.key / self.run_id
        self.unverified_dir = paths.UNVERIFIED / self.model.key / self.run_id
        self.units_json = self.results / "speed" / "units.json"
        self.pidfile = Pidfile(self.runs / "pidfile")
        self.est = estimate(self.model)
        self.server: Server | None = None
        self.sampler: MemorySampler | None = None
        self.guard: quiet.QuietGuard | None = None
        self.cap_proc: subprocess.Popen | None = None
        self.lock_held = False
        self.phase = "speed" if "speed" in opts.legs else "capability"
        self.launches: list[dict] = []
        self.memory: list[dict] = []
        self.ssd_peak = 0
        self.started = _iso()
        self.status, self.reason = "running", ""
        self.speed_complete = False
        self.machine: dict = {}                    # the first invocation's machine_info(): never overwritten
        self.machine_now: dict = {}                # this invocation's, recorded per launch
        self.server_settings: dict | None = None   # settings.json as written for the last launch, minus auth
        self.epoch: str | None = None              # set by preflight when the capability leg runs
        self.stored_epoch: str | None = None       # the epoch an earlier invocation of this run recorded
        self.gate: dict | None = None              # the gate decision of DESIGN §6.6 before the ranking set
        self.server_lost = 0                       # relaunches after a lost server (all legs)
        self.claimed = False
        self._claim_fd: int | None = None
        self.notice_due = False

    # -- persistent state

    def load_state(self) -> None:
        """--resume: the previous units.json, launch list, machine, epoch and gate decision of this
        run id. The first invocation's machine stays the run's: a resume never rewrites it."""
        if not self.results.exists():
            raise Stop("refused", f"--resume {self.run_id}: no {paths.tilde(self.results)}"
                       + (" (pass --unverified for a mechanics run)" if not self.opts.unverified else ""))
        man = self.results / "manifest.json"
        if man.exists():
            old = json.loads(man.read_text())
            self.launches = old.get("launches", [])
            self.memory = old.get("memory", [])
            self.started = old.get("started", self.started)
            self.machine = old.get("machine") or {}
            self.stored_epoch = old.get("epoch")
            self.gate = old.get("gate_decision")

    def check_resume(self) -> None:
        """Refuse a resume that would mix macOS/Xcode builds in the speed leg, or harness epochs in the
        capability leg, inside one reported run (DESIGN §2, §7): the first invocation's values win."""
        old, now = builds(self.machine), builds(self.machine_now)
        speed_rows = any(self.unit_rec(self.read_units(), u)["attempts"] for u in self.read_units()["units"])
        if "speed" in self.opts.legs and speed_rows and old["macos"] and old != now:
            raise Stop("refused", f"--resume {self.run_id}: the speed leg ran on macOS {old['macos']} / Xcode "
                       f"{old['xcode']}, this machine is on {now['macos']} / {now['xcode']}; start a new run "
                       "(or resume with --legs capability when only the capability leg is left)")
        epoch = registry.epoch_id(registry.load_pins())
        state = self.runs / "capability" / "state.json"
        if ("capability" in self.opts.legs and self.stored_epoch and epoch != self.stored_epoch
                and state.exists() and json.loads(state.read_text()).get("records")):
            raise Stop("refused", f"--resume {self.run_id}: its capability rows are from harness epoch "
                       f"{self.stored_epoch}, pins.toml now names {epoch}; start a new run")

    # units.json has two writers (DESIGN §8, speed.py's header): speed records each attempt's metrics
    # as soon as it ends; this module adds the quiet verdict, `kept`, the launch and the unit verdict.
    # Every change is therefore a read-modify-write of the file, never a rewrite from memory.

    def read_units(self) -> dict:
        doc = json.loads(self.units_json.read_text()) if self.units_json.exists() else {}
        doc.setdefault("run_id", self.run_id)
        doc.setdefault("model", self.model.key)
        doc.setdefault("functional", None)
        doc.setdefault("units", {})
        return doc

    @staticmethod
    def unit_rec(doc: dict, unit: str) -> dict:
        rec = doc["units"].get(unit)
        if isinstance(rec, list):
            rec = {"verdict": None, "attempts": rec}
        if not isinstance(rec, dict):
            rec = {"verdict": None, "attempts": []}
        rec.setdefault("attempts", [])
        rec.setdefault("verdict", None)
        doc["units"][unit] = rec
        return rec

    def update_unit(self, unit: str, attempt: dict | None = None, verdict: object = ...) -> dict:
        """Merge one attempt (by number) and/or the unit verdict into units.json; returns the record."""
        doc = self.read_units()
        rec = self.unit_rec(doc, unit)
        if attempt is not None:
            old = next((a for a in rec["attempts"] if a.get("attempt") == attempt["attempt"]), None)
            if old is None:
                rec["attempts"].append(attempt)
            else:
                old.update(attempt)
            rec["attempts"].sort(key=lambda a: a.get("attempt", 0))
        if verdict is not ...:
            rec["verdict"] = verdict
        write_json(self.units_json, doc)
        return rec

    def rel(self, p: Path) -> str:
        """A path in units.json or the manifest: relative to the run's results dir (an attempt moved to
        backend/results/_unverified gets a path that climbs out of docs/benchmark)."""
        return os.path.relpath(Path(p).resolve(), self.results.resolve())

    def write_manifest(self) -> None:
        """<results>/<run-id>/manifest.json (docs/benchmark/<family>/<quant_dir>/backend/<run-id>/) once
        something was launched; before that (a refused run) only runs/<model>/<run-id>/manifest.json, so the
        results never hold a run without data."""
        cap_results = self.results / "capability" / "results.json"
        measured = bool(self.launches) or (self.results / "manifest.json").exists()
        write_json((self.results if measured else self.runs) / "manifest.json", {
            "protocol_version": paths.PROTOCOL_VERSION, "run_id": self.run_id, "status": self.status,
            "reason": self.reason, "started": self.started, "ended": None if self.status == "running" else _iso(),
            "model": self.model.to_dict(), "unverified": self.opts.unverified,
            "arm": {"thinking": self.opts.thinking, "only": list(self.opts.only), "harnesses": list(self.opts.harnesses),
                    "thinking_budget": self.opts.thinking_budget},
            "options": {"legs": list(self.opts.legs), "capability": self.opts.capability,
                        "resumed": self.opts.resume is not None, "notice_wait_s": self.opts.notice_wait},
            "estimate": self.est,
            "server": {"omlx": git_info(paths.OMLX_REPO), "base_path": paths.BASE, "port": paths.PORT,
                       "settings": self.server_settings,
                       "model_settings": model_settings(list(self.models.values()))},
            "launches": self.launches, "memory": self.memory,
            "ssd_cache": {"dir": self.runs / "ssd-cache", "peak_bytes": self.ssd_peak},
            "pins": registry.load_pins(), "epoch": self.epoch or self.stored_epoch or registry.epoch_id(registry.load_pins()),
            "machine": self.machine, "repo": git_info(paths.REPO),
            "capability_results": self.rel(cap_results) if cap_results.exists() else None,
            "gate_decision": self.gate,
        })

    # -- lock

    def lock_sh(self, *args: str) -> tuple[int, str]:
        """gpu_lock.sh; a no-op that succeeds when this machine runs no GPU-lock protocol (config [lock])."""
        if not paths.lock_enabled():
            return 0, "GPU-lock protocol off (config [lock] mode)"
        env = {**os.environ, **paths.lock_env(), "MODELEVAL_RUN_ID": self.run_id}
        try:
            p = subprocess.run([str(paths.SCRIPTS / "gpu_lock.sh"), *args], capture_output=True, text=True,
                               env=env, timeout=60)
            rc, out = p.returncode, (p.stdout + p.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            rc, out = 1, str(e)
        self.runs.mkdir(parents=True, exist_ok=True)
        with (self.runs / "lock.log").open("a", encoding="utf-8") as f:
            f.write(f"{_iso()} gpu_lock.sh {' '.join(args)} -> {rc}\n{out}\n")
        return rc, out

    def purpose(self) -> str:
        if self.phase == "speed" and not self.opts.unverified:
            return f"modeleval {self.model.key} speed (timed)"
        what = "mechanics check" if self.opts.unverified and self.phase == "speed" else "capability"
        return f"modeleval {self.model.key} {what}, accuracy only (untimed)"

    def eta_minutes(self) -> int:
        cap = capability_minutes(self.est, self.opts.capability) if "capability" in self.opts.legs else 0
        spd = self.est["functional_min"] + self.est["quiet_min"] if self.phase == "speed" else 0
        return max(5, int(spd + cap))

    def our_stale_lock(self) -> bool:
        if not paths.lock_enabled():
            return False
        try:
            text = Path(paths.lock_env()["GPU_LOCK_PATH"]).read_text(encoding="utf-8")
        except OSError:
            return False
        return f"run: {self.run_id}" in text

    def notice(self) -> None:
        """LOCK-NOTICE, then wait (the driving session relays it to the peers). Without the GPU-lock
        protocol there are no peers to warn: neither."""
        if not paths.lock_enabled():
            return
        now = time.time()
        start = now + self.opts.notice_wait
        timed = self.est["functional_min"] + self.est["quiet_min"] if self.phase == "speed" else 0
        progress.emit("LOCK-NOTICE", start=_iso(start), eta=_iso(start + timed * 60), timed_minutes=int(timed),
                      purpose=self.purpose(), run_id=self.run_id)
        if self.opts.notice_wait > 0:
            sleep(self.opts.notice_wait)

    def refuse(self, reason: str) -> Stop:
        """`refused` (exit 5) means nothing was measured; once a launch happened the run is `stopped`
        and resumable (a lock wait after a battery pause, port 8500 on re-acquiring, the runner's
        preflight after the speed leg)."""
        return Stop("stopped" if self.launches else "refused", reason)

    def acquire(self) -> None:
        if self.lock_held:
            if self.notice_due:   # our own stale lock was taken over before a timed leg (DESIGN §5 step 1)
                self.notice_due = False
                self.notice()
            return
        self.notice()
        t0, waiting = time.time(), False
        while True:
            rc, out = self.lock_sh("acquire", self.purpose(), str(self.eta_minutes()))
            if rc == 0:
                self.lock_held = True
                return
            if rc == 3:   # lock created, but the agent-benchmark server listens on 8500
                self.lock_held = True
                self.release()
                raise self.refuse("port 8500 is up: the agent-benchmark session must stop its oMLX first")
            if rc != 2:   # a timeout may leave the lock created with our run id: cleanup releases it
                raise self.refuse(f"gpu_lock.sh acquire exited {rc}: {out[-200:]}")
            if time.time() - t0 > LOCK_WAIT_S:
                raise self.refuse(f"the GPU lock stayed held by another session for {LOCK_WAIT_S // 60} min")
            if not waiting:
                progress.emit("UNIT-WAIT", unit="lock", reason="GPU lock held by another session")
                waiting = True
            sleep(POLL_S)

    def release(self) -> None:
        """Release only after nothing listens on our port (DESIGN §6.7). Also a lock that carries this
        run id without lock_held: an acquire interrupted between the lock's creation and its exit."""
        if not self.lock_held and not self.our_stale_lock():
            return
        if port_listening(paths.PORT) and not wait_port_free(paths.PORT, 30):
            progress.emit("ERROR", reason=f"port {paths.PORT} still listening; GPU lock kept")
            return
        rc, out = self.lock_sh("release")
        if rc == 0:
            self.lock_held = False
        else:
            progress.emit("ERROR", reason=f"gpu_lock.sh release exited {rc}: {out[-120:]}")

    # -- launches

    def prelaunch(self) -> None:
        r = quiet.prelaunch_check(timed=self.phase == "speed" and not self.opts.unverified,
                                  settle_s=self.opts.settle_s, log=self.runs / "quiet-checks.log")
        if not r.ok:
            raise Stop("stopped", "pre-launch check failed: " + "; ".join(r.reasons)[:200])

    def launch(self, reason: str) -> None:
        n = max([ln["launch"] for ln in self.launches], default=0) + 1
        self.sampler = MemorySampler(self.results, launch_no=n)        # vm_stat baseline before the load
        self.server = Server(self.model, self.runs, n, models=list(self.models.values()), pidfile=self.pidfile,
                             caffeinate=CAFFEINATE)
        pct, _ = quiet.battery()
        # the builds this launch ran on: a resume refuses a change for the speed leg (check_resume), and
        # report.py never puts a run whose kept attempts span two builds into a build's table
        rec = {"reason": reason, "battery": pct, "phase": self.phase, "builds": builds(self.machine_now)}
        try:
            self.server.start()
        except ServerError as e:
            self.launches.append({**self.server.record(), **rec, "failed": str(e)[:300]})
            raise
        self.server_settings = self.server.written_settings
        self.sampler.start(self.server.pid)
        self.sampler.mark("after-load")
        self.guard = quiet.QuietGuard(self.results, [self.server.pid], pidfile=self.pidfile)
        self.guard.start_monitor()
        self.launches.append({**self.server.record(), **rec})
        self.write_manifest()

    def server_problem(self) -> str | None:
        """Why the current server cannot serve (exited, or GET /v1/models fails); None when it can."""
        if self.server is None:
            return "no server"
        return self.server.health()

    def relaunch(self, reason: str, why: str) -> None:
        """The same configuration again after a lost server, with its warm-up (DESIGN §6.7)."""
        self.server_lost += 1
        if self.server_lost > MAX_SERVER_LOST:
            raise Stop("stopped", f"server lost {self.server_lost} times ({why}); --resume continues")
        progress.emit("ERROR", reason=f"server lost ({why}); relaunching"[:200], launch=len(self.launches) + 1)
        self.shutdown_server()
        self.prelaunch()
        self.launch(reason)

    def ensure_server(self, where: str) -> None:
        """Before a timed unit and after an incomplete attempt: the speed leg's health check."""
        why = self.server_problem()
        if why:
            self.relaunch("server-lost", f"{why}, before {where}")

    def shutdown_server(self) -> None:
        """Stop monitor, sampler and server of the current launch; record its memory summary."""
        if self.guard is not None:
            self.guard.stop_monitor()
            self.guard = None
        if self.sampler is not None:
            self.memory.append(self.sampler.stop())
            self.sampler = None
        if self.server is not None:
            self.note_cache()
            n = self.server.launch_no
            try:
                self.server.stop()
            finally:
                for ln in self.launches:
                    if ln["launch"] == n:
                        ln.update(self.server.record(), reason=ln["reason"], battery=ln["battery"], phase=ln["phase"])
                self.server = None

    def note_cache(self) -> None:
        if (self.runs / "ssd-cache").exists():
            self.ssd_peak = max(self.ssd_peak, dir_bytes(self.runs / "ssd-cache"))

    def mark(self, label: str) -> None:
        if self.sampler is not None:
            self.sampler.mark(label)

    # -- boundaries

    def hold_wait(self) -> None:
        if not quiet.hold_requested():
            return
        progress.emit("HOLD", run_id=self.run_id, file=paths.tilde(paths.HOLD_FILE))
        while quiet.hold_requested():
            sleep(POLL_S)
        progress.emit("HOLD-RELEASED", run_id=self.run_id)

    def battery_pause(self, where: str) -> None:
        """The user's rule (DESIGN §5.5): give everything back, wait for 80 %, come back as a new launch."""
        had_server = self.server is not None
        pct, _ = quiet.battery()
        self.shutdown_server()
        self.release()
        progress.emit("BATTERY-PAUSE", battery=pct, at=where, run_id=self.run_id)
        self.write_manifest()
        while True:
            pct, _ = quiet.battery()
            if pct < 0 or pct >= quiet.RESUME_PCT:
                break
            sleep(POLL_S)
        progress.emit("BATTERY-READY", battery=pct)
        self.acquire()
        if had_server:
            self.prelaunch()
            self.launch("battery-pause")
        progress.emit("RESUMED", launch=len(self.launches), run_id=self.run_id)

    def boundary(self, kind: quiet.Boundary, where: str) -> None:
        self.hold_wait()
        if quiet.boundary(kind) != "go":
            self.battery_pause(where)

    # -- speed leg

    def speed_ctx(self, speed):
        return speed.SpeedContext(model=self.model, base_url=self.server.base_url, server_pid=self.server.pid,
                                  run_dir=self.results, tokenizer_dir=self.model.path,
                                  unverified_dir=self.unverified_dir)

    def speed_call(self, fn, *args):
        """One speed coroutine, with the harness's printed tables in runs/<model>/<run-id>/speed-stdout.log:
        the driving session parses this process's stdout, which carries progress lines only."""
        log = self.runs / "speed-stdout.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            return asyncio.run(fn(*args))

    def unit_done(self, unit: str) -> bool:
        v = self.unit_rec(self.read_units(), unit)["verdict"]
        return v in KEPT or (self.opts.unverified and v is not None)

    def functional(self, speed) -> None:
        """The functional block (DESIGN §5 step 4). A failure that came from the server is not the
        model's: the server exited or stopped answering, the block raised (`stop_reason` "error: ..."),
        or a probe failed in transport, by timeout, with a 5xx or a server stream error
        (`server_error`). Then the server is relaunched and the block repeated once; a second such
        failure stops the run resumably. Only a judged FAIL on a live server is a gate failure."""
        f = self.read_units().get("functional")
        if isinstance(f, dict) and f.get("passed") is True:
            return
        for repeat in (False, True):
            self.mark("functional")
            res = self.speed_call(speed.run_functional, self.speed_ctx(speed))
            self.mark("idle")
            ok = functional_passed(res)
            stop = str(_field(res, "stop_reason") or "")
            down = None if ok else self.server_problem()
            probe = None if ok else _field(res, "server_error")
            definite = down or (stop if stop.startswith("error:") else None) \
                or (probe if str(probe or "").startswith(("transport", "timeout")) else None)
            lost = definite or (None if repeat else probe)   # a repeated 5xx on a live server is the model's
            self.record_functional(res, ok, lost)
            if ok:
                return
            if lost and not repeat:
                self.relaunch("functional-server-lost", f"functional block: {lost}")
                continue
            if lost:
                raise Stop("stopped", f"server lost in the functional block: {lost}"[:300])
            detail = _failing_detail(res)
            raise Stop("gate-fail", f"functional block FAIL: {stop or 'no stop reason'}"
                       + (f" ({detail})" if detail else "") + (f"; probe error {probe}" if probe else ""))

    def record_functional(self, res: object, ok: bool, lost: str | None) -> None:
        """units.json `functional`; after a lost server `passed` is null, so neither a resume nor the
        report reads it as a gate failure."""
        doc = self.read_units()                     # speed may have recorded its own copy meanwhile
        f = doc.get("functional") if isinstance(doc.get("functional"), dict) else {}
        f.update(passed=None if lost else ok, launch=self.server.launch_no, time=f.get("time") or _iso())
        if lost:
            f["server_lost"] = lost[:300]
        else:
            f.pop("server_lost", None)
        f.setdefault("result", _plain(res))
        doc["functional"] = f
        write_json(self.units_json, doc)

    def speed_leg(self) -> bool:
        """True when every timed unit has a final verdict; False when a pre-unit check gave up."""
        speed = load_speed()
        self.functional(speed)
        self.set_aside_unjudged()
        order = list(speed.timed_units(self.model))
        long_units = set(order[:1]) | {u for u in order if u == "m4-1"}
        long_units |= {u for i, u in enumerate(order) if i and u.startswith("m3") and not order[i - 1].startswith(("m3", "g5"))}
        for unit in order:
            if self.unit_done(unit):
                continue
            if not self.timed_unit(speed, unit, "long-unit" if unit in long_units else "unit"):
                return False
        if hasattr(speed, "finalize"):   # memory peaks into units.json, after our last write of it
            speed.finalize(self.results)
        return True

    def set_aside_unjudged(self) -> None:
        """Attempts an interrupted invocation left without a verdict: their raw files go to _unverified
        (results/ holds kept attempts only) and the attempt is marked `interrupted`."""
        doc = self.read_units()
        for unit in list(doc["units"]):
            for a in self.unit_rec(doc, unit)["attempts"]:
                if a.get("verdict") is None and not self.opts.unverified:
                    raw = self.results / str(a.get("raw_file") or "")
                    if a.get("raw_file") and raw.is_file() and raw.resolve().parent == (self.results / "speed").resolve():
                        a["raw_file"] = self.rel(self.move_unverified(raw))
                    a.update(verdict="interrupted", kept=False)
        write_json(self.units_json, doc)

    def timed_unit(self, speed, unit: str, kind: quiet.Boundary = "unit") -> bool:
        """Up to MAX_ATTEMPTS attempts in this invocation (numbering continues across resumes). Before
        every attempt: the battery and hold rule (DESIGN §5.5), the monitor and the server's health,
        then the pre-unit check (major conditions only). An attempt with a major disturbance or an
        incomplete one is repeated; a minor one is kept with its flags (DESIGN §5.4); a cache hit is
        final."""
        self.update_unit(unit, verdict=None)
        verdicts: list[str] = []
        for _ in range(MAX_ATTEMPTS):
            attempt = max([a.get("attempt", 0) for a in self.unit_rec(self.read_units(), unit)["attempts"]], default=0) + 1
            self.boundary(kind, unit)
            self.ensure_server(unit)
            if not self.guard.monitor_alive():   # a dead monitor would leave every later window uncovered
                progress.emit("ERROR", unit=unit, reason="quiet monitor was not running; restarted")
                self.guard.start_monitor()
            if not self.opts.unverified and not self.guard.wait_quiet(unit, self.opts.settle_s):
                progress.emit("ERROR", unit=unit, reason=f"not quiet after {self.opts.settle_s} s; speed leg stopped")
                return False
            progress.emit("UNIT-START", unit=unit, attempt=attempt, launch=self.server.launch_no)
            self.mark(unit)
            res = self.speed_call(speed.run_unit, self.speed_ctx(speed), unit, attempt)
            self.mark("idle")
            v = self.guard.judge(res.t_start, res.t_end)
            valid, note, why = cold_ok(res)
            verdict = attempt_verdict(v, valid, why)
            verdicts.append(verdict)
            keep = verdict in KEPT or self.opts.unverified
            raw = Path(res.raw_file) if keep else self.move_unverified(Path(res.raw_file))
            att = {"attempt": attempt, "launch": self.server.launch_no, "t_start": res.t_start, "t_end": res.t_end,
                   "verdict": verdict, "severity": v.severity, "reasons": v.reasons,
                   "major_reasons": v.major_reasons, "minor_reasons": v.minor_reasons,
                   "peak_foreign_cpu": v.peak_foreign_cpu, "samples": v.samples,
                   "disturbed_samples": v.disturbed_samples, "major_samples": v.major_samples,
                   "minor_samples": v.minor_samples, "battery_min": v.battery_min,
                   "valid": valid, "note": note, "raw_file": self.rel(raw), "kept": keep}
            doc_att = next((a for a in self.unit_rec(self.read_units(), unit)["attempts"] if a.get("attempt") == attempt), None)
            if doc_att is None:                     # speed did not record it: keep its metrics here
                att["metrics"] = _plain(res.metrics)
            why_text = (v.minor_reasons if verdict == "minor" else v.major_reasons if verdict == "disturbed"
                        else v.reasons) + ([note] if not valid else [])
            progress.emit("UNIT-DONE", unit=unit, attempt=attempt, verdict=verdict,
                          **({"reason": "; ".join(why_text)[:160]} if verdict != "clean" else {}))
            if keep:
                self.update_unit(unit, att, verdict)
                return True
            if verdict == "invalid":        # a cache hit on a cold request is systematic: repeating cannot fix it
                self.update_unit(unit, att, "invalid")
                return True
            self.update_unit(unit, att)
        self.update_unit(unit, verdict=final_verdict(verdicts))
        return True

    def move_unverified(self, raw: Path) -> Path:
        dest = self.unverified_dir / "speed"
        dest.mkdir(parents=True, exist_ok=True)
        for f in [raw] + [g for g in raw.parent.glob(raw.stem + ".*") if g != raw]:
            if f.exists():
                shutil.move(str(f), str(dest / f.name))
        return dest / raw.name

    def quiet_window_end(self) -> None:
        self.phase = "capability"
        progress.emit("QUIET-WINDOW-END", model=self.model.key, run_id=self.run_id)
        if self.lock_held:
            rc, out = self.lock_sh("update-purpose", self.purpose(), str(self.eta_minutes()))
            if rc != 0:
                progress.emit("ERROR", reason=f"gpu_lock.sh update-purpose exited {rc}: {out[-120:]}")

    # -- capability leg

    def write_spec(self, resume: bool, sets: list[str]) -> Path:
        epoch = self.epoch or registry.epoch_id(registry.load_pins())
        spec = {"model": self.model.to_dict(tilde=False), "base_url": self.server.base_url,
                "server_pid": self.server.pid, "launch": self.server.launch_no, "sets": sets,
                "run_root": str(self.runs / "capability"), "results": str(self.results / "capability" / "results.json"),
                "ws_root": str(paths.WS_ROOT), "epoch": epoch, "vendor": str(paths.VENDOR / str(epoch)),
                "wall_s": 3600, "max_steps": 30, "max_turns": 40, "thinking": self.opts.thinking, "resume": resume,
                "pidfile": str(self.pidfile.path), "monitor": str(self.results / "speed" / "monitor.tsv"),
                "server_log": str(self.server.log_path)}
        if self.opts.only:
            spec["only"] = list(self.opts.only)
        if self.opts.harnesses:
            spec["harnesses"] = list(self.opts.harnesses)
        p = self.runs / "capability" / "spec.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        return p

    def run_runner(self, spec: Path) -> int:
        environ = {**os.environ, "PATH": runner_path(os.environ.get("PATH", ""))}
        env = load_allowed_env()(environ, {"PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0"})
        self.mark("capability")
        self.cap_proc = subprocess.Popen(capability_command(spec), cwd=paths.REPO, env=env,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        self.pidfile.add(self.cap_proc.pid, "capability", "modeleval.capability.runner")
        rc = self.cap_proc.wait()   # a signal raises here with cap_proc still set, so cleanup stops it
        self.pidfile.remove(self.cap_proc.pid)
        self.cap_proc = None
        self.mark("idle")
        self.note_cache()
        return rc

    def capability_leg(self) -> None:
        """The gate set, then the ranking set unless the gate decision says "worse" (DESIGN §6.3,
        §6.6). The check lives here, not in the runner: this process owns results/, and the runner
        is a sandboxed harness driver. With `--resume`, ranking rows already in state.json mean the
        decision was taken: the leg goes on without re-deciding."""
        if not registry.epoch_id(registry.load_pins()):
            raise self.refuse("no harness epoch in pins.toml: create one first (capability epoch)")
        self.boundary("harness-block", "capability")
        started = self.ranking_started()
        self.run_capability(["gate"], resume=self.opts.resume is not None)
        if self.opts.capability == "gate":
            return
        if not started and self.opts.capability == "full-force":
            self.gate = {"decision": None, "action": "ranking", "why": "--capability full-force"}
        elif not started:
            self.gate = self.gate_decision()
            self.write_manifest()
            if self.gate.get("action") == "skip":
                return
        self.run_capability(["ranking"], resume=True)

    def ranking_started(self) -> bool:
        try:
            state = json.loads((self.runs / "capability" / "state.json").read_text())
        except (OSError, ValueError):
            return False
        return any(r.get("set") == "ranking" for r in state.get("records") or [])

    def gate_decision(self) -> dict:
        """DESIGN §6.6's gate rule against the reference's newest finished gate set in the same
        harness epoch (results/ only, never _unverified). No comparison, and so the ranking set,
        when this model is the reference or that reference run does not exist yet."""
        from modeleval import report, stats

        out: dict = {"epoch": self.epoch, "decision": None, "action": "ranking"}
        if self.model.reference:
            out["why"] = "reference model"
        else:
            ref_key = registry.reference(self.models).key
            refs = [r for r in report.discover(paths.RESULTS) if r.model_key == ref_key
                    and str(r.epoch) == str(self.epoch) and r.records and report.cap_complete(r, "gate")]
            mine = report.cap_records(report._json(self.results / "capability" / "results.json"))
            if not refs:
                out["why"] = f"no finished reference gate set in epoch {self.epoch}"
            elif not mine:
                out["why"] = "no gate rows of this run"
            else:
                ref = max(refs, key=lambda r: r.order)
                c = stats.compare(mine, ref.records, set_name="gate")
                out.update({k: c[k] for k in ("decision", "wins", "losses", "ties", "net", "p", "pairs",
                                               "excluded", "provisional", "per_harness")},
                           reference_run=f"{ref.model_key}/{ref.run_id}",
                           action="skip" if c["decision"] == "worse" else "ranking")
        progress.emit("GATE-DECISION", decision=out["decision"] or "-", action=out["action"],
                      **{k: out[k] for k in ("wins", "losses", "reference_run") if k in out},
                      **({"why": out["why"]} if "why" in out else {}),
                      **({"excluded_pairs": len(out["excluded"])} if out.get("excluded") else {}))
        return out

    def run_capability(self, sets: list[str], resume: bool) -> None:
        """The runner on `sets` until it is done, through hold, battery pauses and lost servers."""
        while True:
            rc = self.run_runner(self.write_spec(resume, sets))
            resume = True
            self.flag_tasks()              # before any branch: a relaunch drops this launch's monitor
            if rc == 0:
                return
            if rc == 75:
                y = self.runner_yield()
                why = y.get("reason")
                if why == "hold" or (why is None and quiet.hold_requested()):
                    self.hold_wait()
                    continue
                if why in ("battery", "battery-stop") or (why is None and quiet.boundary("harness-block") != "go"):
                    self.battery_pause("capability")
                    continue
                raise Stop("stopped", f"capability runner yielded ({why or 'no reason recorded'}: "
                           f"{str(y.get('detail', ''))[:160]}); --resume continues")
            if rc == 76:
                self.relaunch("server-lost", "the capability runner found it gone")
                continue
            if rc == 2:
                raise self.refuse("capability runner preflight refused (see its output)")
            raise Stop("error", f"capability runner exited {rc}")

    def runner_yield(self) -> dict:
        """The runner's reason for exit 75 (battery, battery-stop, hold, drift), from its yield.json."""
        try:
            y = json.loads((self.runs / "capability" / "yield.json").read_text())
        except (OSError, ValueError):
            return {}
        return y if isinstance(y, dict) else {}

    def flag_tasks(self) -> None:
        """DESIGN §6.4: every capability task records the flagged samples (severity and major/minor counts,
        §5.4), peak foreign CPU and battery minimum that overlapped it (flags only; nothing is repeated,
        whatever the severity). The runner leaves `flags` null, since the monitor belongs to this
        process, or starts them with its memory watchdog's `memory_kills` (§6.1), which are kept. They go
        into the runner's state.json, so that its later rewrites of results.json keep them, and into
        results.json now (through the exporter)."""
        try:
            self._flag_tasks()
        except Exception as e:  # noqa: BLE001 - flags never stop the leg
            progress.emit("ERROR", reason=f"capability flags: {type(e).__name__}: {e}"[:200])

    def _flag_tasks(self) -> int:
        from modeleval.capability import export
        state_p = self.runs / "capability" / "state.json"
        if not state_p.exists():
            return 0
        state = json.loads(state_p.read_text())
        guard = self.guard or quiet.QuietGuard(self.results, [])
        new, by_run = 0, {}
        for r in state.get("records", []):
            own = r.get("flags")
            if (own is None or "disturbed" not in own) and r.get("t_start") and r.get("t_end"):
                r["flags"] = {**guard.flags(float(r["t_start"]), float(r["t_end"])), **(own or {})}
                new += 1
            if r.get("run_dir") and r.get("flags") is not None:
                by_run[r["run_dir"]] = r["flags"]
        if new:
            tmp = state_p.with_name(f".{state_p.name}.tmp")
            tmp.write_text(json.dumps(state, indent=1) + "\n")
            os.replace(tmp, state_p)
        res_p = self.results / "capability" / export.RESULTS_NAME
        if res_p.exists():
            doc = json.loads(res_p.read_text())
            changed = False
            for key in ("rows", "superseded", "warmups"):
                for row in doc.get(key) or []:
                    f = by_run.get(row.get("run_dir"))
                    if f is not None and row.get("flags") != f:
                        row["flags"], changed = f, True
            if changed:
                export.write_results(res_p, doc)
        return new

    # -- whole run

    def preflight(self) -> None:
        m = self.model
        if not (m.path / "config.json").exists():
            raise Stop("refused", f"not prepared (modeleval download/convert {m.key}): no checkpoint at {paths.tilde(m.path)}")
        if not m.link.exists() or m.link.resolve() != m.path.resolve():
            raise Stop("refused", f"not registered (modeleval register {m.key}): {paths.tilde(m.link)} "
                       "does not point at the checkpoint")
        if port_listening(paths.PORT):
            raise Stop("refused", f"port {paths.PORT} is already listening")
        free = prepare.free_bytes(paths.RUNS)
        if free < SSD_CACHE_BYTES + prepare.DISK_MARGIN:   # oMLX fills runs/.../ssd-cache up to its cap
            left = ", ".join(f"{paths.tilde(d)} {b / 1e9:.0f} GB" for d, b in ssd_caches(self.runs / "ssd-cache"))
            raise Stop("refused", f"{free / 1e9:.0f} GB free under {paths.tilde(paths.RUNS)}; a launch needs "
                       f"{(SSD_CACHE_BYTES + prepare.DISK_MARGIN) / 1e9:.0f} GB for its SSD cache"
                       + (f" (leftover caches: {left})" if left else ""))
        if "speed" in self.opts.legs:
            miss = missing_prompts(load_speed(), m)
            if miss:
                raise Stop("refused", f"fixtures not built ({', '.join(miss)} missing): "
                           "run `uv run agentbench fixtures build` in harness/ (needs the Qwen3.8 oQ5 tokenizer)")
        if "capability" in self.opts.legs:
            if not CAP_PYTHON.exists():
                raise Stop("refused", f"no capability interpreter {paths.tilde(CAP_PYTHON)}: "
                           "uv sync --project envs/capability && uv sync --project envs/agent-tools")
            try:
                self.epoch = ensure_epoch()
            except Exception as e:  # noqa: BLE001 - EpochError, or git failing on a peer repo
                raise Stop("refused", f"harness epoch: {e}") from None

    def claim(self) -> None:
        """One orchestrator per run id: an exclusive flock on runs/<model>/<run-id>/orchestrator.pid,
        held (fd open, not inherited by children) for the life of this process. The kernel drops it
        when the process dies, even by SIGKILL, so neither a reused PID nor `uv run`'s process group
        can fool it, and two instances can never both pass. The PID in the file is for people."""
        self.runs.mkdir(parents=True, exist_ok=True)
        orch = self.runs / "orchestrator.pid"
        fd = os.open(orch, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            try:
                holder = orch.read_text(encoding="utf-8").split()[0]
            except (OSError, IndexError):
                holder = "?"
            raise Stop("refused", f"run {self.run_id} is already running (PID {holder})") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._claim_fd = fd
        self.claimed = True

    def execute(self) -> None:
        self.machine_now = machine_info()
        if self.opts.resume:
            self.load_state()
            self.check_resume()                    # before the claim: a refusal leaves the run's files alone
        self.machine = self.machine or self.machine_now
        self.claim()
        progress.LOG_PATH = self.runs / "progress.log"
        if self.opts.resume:
            for line in self.pidfile.kill_all():   # orphans of the crashed attempt
                if "gone" not in line:
                    progress.emit("ERROR", reason=f"stale child from the previous attempt: {line}")
            self.resume_phase()
            if self.our_stale_lock():              # its lock (same run id): adopt it, so a pause releases it
                self.lock_held = True
                self.notice_due = self.phase == "speed" and not self.opts.unverified
                self.lock_sh("update-purpose", self.purpose(), str(self.eta_minutes()))
        self.preflight()
        self.write_manifest()
        self.boundary("long-unit" if self.phase == "speed" else "harness-block", "start")
        self.acquire()
        self.prelaunch()
        self.launch("initial")
        if "speed" in self.opts.legs:
            self.speed_complete = self.speed_leg()
            self.quiet_window_end()
        if "capability" in self.opts.legs:
            self.capability_leg()
        self.status = "complete" if self.all_units_clean() else "incomplete"
        self.reason = "" if self.status == "complete" else "some speed units have no kept result; --resume repeats them"

    def resume_phase(self) -> None:
        """--resume after a finished speed leg: only untimed work is left, so the lock purpose, the
        notice and the pre-launch check are the capability leg's (no timed notice, no 600 s settle)."""
        if self.phase != "speed":
            return
        f = self.read_units().get("functional")
        if isinstance(f, dict) and f.get("passed") is True \
                and all(self.unit_done(u) for u in load_speed().timed_units(self.model)):
            self.phase = "capability"

    def all_units_clean(self) -> bool:
        """Every timed unit kept: clean, or with a minor flag only (DESIGN §5.4)."""
        if "speed" not in self.opts.legs:
            return True
        if not self.speed_complete:
            return False
        doc = self.read_units()
        verdicts = [self.unit_rec(doc, u)["verdict"] for u in list(doc["units"])]
        return all(v in KEPT or (self.opts.unverified and v is not None) for v in verdicts)

    def cleanup(self) -> None:
        """Idempotent; every step on its own, so one failure never leaves the lock or a server behind.
        Nothing at all when this process never claimed the run: the pidfile, the lock and the files
        then belong to the instance that holds the claim."""
        if not self.claimed:
            return
        steps = [self._stop_runner, self.shutdown_server, self._kill_rest, self.release, self._cache, self._final]
        for step in steps:
            try:
                step()
            except BaseException as e:  # noqa: BLE001 - cleanup must reach the lock release
                with contextlib.suppress(BaseException):
                    progress.emit("ERROR", reason=f"cleanup {step.__name__}: {type(e).__name__}: {e}"[:200])

    def _stop_runner(self) -> None:
        """The runner first: its signal handler stops the harness groups without recording a verdict.
        Stopping a harness first would let the runner see a normal exit and save a pass or fail for a
        killed run. Then whatever its own cleanup missed."""
        p = self.cap_proc
        if p is None or p.poll() is not None:
            return
        ps = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True).stdout
        groups = {os.getpgid(x) for x in descendants(p.pid, ps) if _pgid_ok(x)} - {os.getpgid(0), p.pid}
        kill_group(p.pid, signal.SIGTERM, 15)
        for g in sorted(groups):
            kill_group(g, signal.SIGTERM, 10)
        self.pidfile.remove(p.pid)

    def _kill_rest(self) -> None:
        for line in self.pidfile.kill_all():
            if "gone" not in line:
                progress.emit("ERROR", reason=f"cleanup: {line}")

    def _cache(self) -> None:
        """A fresh SSD cache per run, deleted at the end with its peak size logged (DESIGN §2), whatever
        the status: oMLX fills it up to 128 GB, and nothing a resume does depends on it."""
        cache = self.runs / "ssd-cache"
        if not cache.exists():
            return
        self.note_cache()
        if self.server is None and not port_listening(paths.PORT) \
                and cache.resolve().is_relative_to(paths.RUNS.resolve()):
            shutil.rmtree(cache, ignore_errors=True)

    def _final(self) -> None:
        self.write_manifest()
        if self._claim_fd is not None:   # the file stays: unlinking it would let a waiter lock a stale inode
            with contextlib.suppress(OSError):
                os.ftruncate(self._claim_fd, 0)
            os.close(self._claim_fd)
            self._claim_fd = None


def _pgid_ok(pid: int) -> bool:
    try:
        os.getpgid(pid)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- plan and entry point


def readiness(opts: Options, m: Model) -> list[str]:
    """Read-only: what `run` would refuse before the lock, and what it would prepare itself."""
    link = m.link.exists() and m.link.resolve() == m.path.resolve()
    out = [f"ready       checkpoint {'ok' if (m.path / 'config.json').exists() else 'MISSING (modeleval download/convert)'}"
           f"; link {'ok' if link else 'MISSING (modeleval register)'}"
           + (f"; port {paths.PORT} is listening (refused)" if port_listening(paths.PORT) else "")]
    if "speed" in opts.legs:
        try:
            miss = missing_prompts(load_speed(), m)
        except ImportError:
            miss = []
        out.append("ready       fixtures " + ("ok" if not miss else f"MISSING {', '.join(miss)} (refused; run "
                                                                     "`uv run agentbench fixtures build` in harness/)"))
    if "capability" in opts.legs:
        epoch = registry.epoch_id(registry.load_pins())
        snap = bool(epoch) and (paths.VENDOR / str(epoch) / "epoch.json").exists()
        out.append(f"ready       runner interpreter {'ok' if CAP_PYTHON.exists() else 'MISSING (refused; uv sync --project envs/capability)'}"
                   f"; vendor/{epoch} {'present' if snap else 'absent (the run recreates it from pins.toml)'}")
    free = prepare.free_bytes(paths.RUNS)
    need = SSD_CACHE_BYTES + prepare.DISK_MARGIN
    out.append(f"ready       disk {free / 1e9:.0f} GB free under {paths.tilde(paths.RUNS)}"
               + ("" if free >= need else f" (refused: the SSD cache needs {need / 1e9:.0f} GB)"))
    for d, b in ssd_caches():
        out.append(f"leftover    {paths.tilde(d)} {b / 1e9:.1f} GB (a crashed run's SSD cache; delete it by hand)")
    return out


def plan_lines(opts: Options, models: dict[str, Model] | None = None) -> list[str]:
    """What a run would do, without touching the lock, the server or any file."""
    models = models or registry.load_models()
    m = registry.get(models, opts.model)
    est = estimate(m)
    run_id = opts.resume or "<new>"
    root = results_root(m, opts.unverified)
    done: dict = {}
    if opts.resume:
        uj = root / opts.resume / "speed" / "units.json"
        if uj.exists():
            done = json.loads(uj.read_text()).get("units", {})
    lines = [f"model       {m.key} ({m.display}), served as {m.served_id}, spec {m.spec}, vision {m.vision}",
             f"run id      {run_id}; results {paths.tilde(root / run_id)}",
             f"legs        {','.join(opts.legs)}; capability sets {opts.capability}; unverified {opts.unverified}",
             f"estimate    factor {est['factor']} ({est['source']}): functional ~{est['functional_min']} min, "
             f"quiet window ~{est['quiet_min']} min, gate set ~{est['gate_min']} min, ranking set ~{est['ranking_min']} min",
             (f"lock        LOCK-NOTICE, wait {opts.notice_wait} s, then gpu_lock.sh acquire "
              f"(\"modeleval {m.key} speed (timed)\" for a verified speed leg, else an untimed purpose)"
              if paths.lock_enabled() else
              "lock        none: the GPU-lock protocol is off (config [lock] mode): no lock file, no LOCK-NOTICE, no wait"),
             f"server      {paths.tilde(paths.OMLX_EXE)} serve --base-path {paths.tilde(paths.BASE)} --headless "
             f"on 127.0.0.1:{paths.PORT}, models from {paths.tilde(paths.MODELS_DIR)}",
             f"settings    hot cache 20GB, SSD cache 128GB in runs/<model>/<run-id>/ssd-cache, chunked prefill, "
             f"8 concurrent, max_tokens 32768; model: context 131072"
             + (", mtp_enabled" if m.spec == "mtp" else "") + "".join(f", {k}={v}" for k, v in m.model_settings.items()),
             f"log expect  {m.server_log_expect}"] + readiness(opts, m)
    if "speed" in opts.legs:
        try:
            units = list(load_speed().timed_units(m))
        except ImportError:
            units = ["g5+m3-64k", "m3-16k", "m3-32k", "m3-110k"] + (["m3-think-16k"] if m.spec != "none" else []) \
                + ["m1", "m4-1", "m4-2", "m4-3"]
        lines.append("functional  G0, G1, G3 text + tool_use, G4" + (" + vision probes" if m.vision else "") + ", M7")
        for u in units:
            rec = done.get(u)
            v = rec.get("verdict") if isinstance(rec, dict) else None
            lines.append(f"unit        {u:14s} {'done (' + v + ')' if v in KEPT else 'to run' + (f' (was {v})' if v else '')}")
    if "capability" in opts.legs:
        epoch = registry.epoch_id(registry.load_pins())
        lines.append(f"capability  sets {'gate' if opts.capability == 'gate' else 'gate, then ranking' + (' (always)' if opts.capability == 'full-force' else ' unless the gate decision is worse')}; "
                     f"epoch {epoch or 'MISSING (pins.toml)'}; runner {paths.tilde(CAP_PYTHON)}")
    return lines


def run(opts: Options, models: dict[str, Model] | None = None) -> int:
    """Run one model; returns the exit code (EXIT)."""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"   # no __pycache__ in peer trees from any child
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    os.environ.update(paths.lock_env())            # gpu_lock.sh, quietlib and the runner see the configured lock
    if opts.dry_run:
        for line in plan_lines(opts, models):
            print(line, flush=True)
        return 0
    try:
        r = Run(opts, models)
    except (Stop, registry.RegistryError) as e:
        status = e.status if isinstance(e, Stop) else "refused"
        progress.emit("ERROR", reason=str(e)[:200])
        progress.emit("RUN-DONE", status=status, code=EXIT[status], resume="no")
        return EXIT[status]

    # SIGHUP only when it is not ignored: under nohup, closing the terminal must not interrupt the run
    sigs = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGHUP] if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN
                                              else [])

    def on_signal(signum, _frame):
        for s in sigs:
            signal.signal(s, signal.SIG_IGN)   # a second signal must not abort the cleanup
        raise Interrupted(signum)

    old = {s: signal.signal(s, on_signal) for s in sigs}
    try:
        r.execute()
    except Stop as e:
        r.status, r.reason = e.status, e.reason
    except Interrupted as e:
        r.status, r.reason = "interrupted", f"signal {e.args[0]}"
    except ServerError as e:
        r.status, r.reason = "stopped", f"server: {e}"
    except Exception as e:  # noqa: BLE001 - report, then clean up
        r.status, r.reason = "error", f"{type(e).__name__}: {e}"
    finally:
        try:
            if r.reason:
                progress.emit("ERROR", reason=r.reason[:300])
        finally:
            r.cleanup()   # whatever happened to stdout: the server, the children and the lock go
            for s, h in old.items():
                signal.signal(s, h)
    code = EXIT[r.status]
    # resumable only when something of this run exists (a run stopped before its first launch has none)
    resumable = r.status in ("stopped", "incomplete", "interrupted", "error") and bool(r.launches or r.results.exists())
    progress.emit("RUN-DONE", status=r.status, code=code, run_id=r.run_id, resume="yes" if resumable else "no")
    progress.LOG_PATH = None
    return code
