"""Prepare a model: download its source, convert it to oQ, register it with oMLX (DESIGN.md §4).

Split by the resource each step needs:

- download: network and disk only, no lock; refused while a speed leg runs (a download is a
  disturbance by the quiet rule), allowed during a capability leg;
- convert: GPU and ~BF16-size memory, so the GPU lock with an untimed `convert` purpose and no model
  server; wraps scripts/oq_convert.py under the oMLX interpreter;
- register: the models/hf-bench link oMLX discovers the model through; refused while our port listens.

A checkpoint is never written in place after its conversion: convert refuses an existing output,
and register only ever creates a link next to the checkpoint.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from modeleval import paths, progress
from modeleval.registry import Model
from modeleval.server import kill_group, port_listening

PEER_PORT = 8500
DISK_MARGIN = 20e9          # DESIGN §4: free disk >= output size + 20 GB


class PrepareError(RuntimeError):
    pass


_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class ConvertInterrupted(PrepareError):
    """SIGINT/SIGTERM/SIGHUP during a conversion: the child is stopped before the lock is released."""


def _interrupt(signum, _frame) -> None:
    for s in _SIGNALS:
        signal.signal(s, signal.SIG_IGN)   # a second signal must not abort the cleanup
    raise ConvertInterrupted(f"conversion interrupted by signal {signum}")


def oq_out() -> Path:
    """Where scripts/oq_convert.py writes its output (<dir>/<served_id>): models/oq next to hf-bench."""
    return paths.MODELS_DIR.parent / "oq"


def snapshot_dir(model: Model) -> Path:
    """The HF cache snapshot of the model's pinned source revision."""
    org, name = model.source["hf_repo"].split("/", 1)
    return paths.HF_HOME / "hub" / f"models--{org}--{name}" / "snapshots" / model.source["revision"]


def lock_file() -> Path:
    return Path(os.environ.get("GPU_LOCK_PATH", str(paths.GPU_LOCK)))


def lock_purpose() -> str | None:
    """The current lock's purpose line, or None without a lock."""
    try:
        text = lock_file().read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^purpose: (.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def speed_leg_running() -> bool:
    """Any lock whose purpose does not say `untimed` may be a timed window (peers write free text):
    no download may disturb it."""
    p = lock_purpose()
    return p is not None and "untimed" not in p


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def hf_exe() -> str:
    local = paths.REPO / ".venv/bin/hf"
    return str(local) if local.exists() else (shutil.which("hf") or "hf")


def download(model: Model, min_free_gb: float = 100.0, dry_run: bool = False) -> list[str]:
    """`hf download <repo> --revision <sha>` into HF_HOME. Returns the command it ran (or would run)."""
    if "convert" not in model.prepare:
        raise PrepareError(f"{model.key} is used as published ({paths.tilde(model.path)}); nothing to download")
    if speed_leg_running():
        raise PrepareError(f"the GPU lock may cover a speed leg (purpose '{lock_purpose()}' is not untimed); "
                           "download during a capability leg")
    if free_bytes(paths.HF_HOME) < min_free_gb * 1e9:
        raise PrepareError(f"less than {min_free_gb:.0f} GB free under {paths.tilde(paths.HF_HOME)}")
    cmd = [hf_exe(), "download", model.source["hf_repo"], "--revision", model.source["revision"]]
    if not dry_run:
        env = {**os.environ, "HF_HOME": str(paths.HF_HOME)}
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            raise PrepareError(f"hf download exited {rc}")
        if not (snapshot_dir(model) / "config.json").exists():
            raise PrepareError(f"download finished but {paths.tilde(snapshot_dir(model))} has no config.json")
    return cmd


def convert_command(model: Model, allow_proxy: bool = False) -> list[str]:
    conv = model.prepare.get("convert")
    if not conv:
        raise PrepareError(f"{model.key} has no prepare.convert in models.toml")
    script = paths.SCRIPTS / "oq_convert.py"
    cmd = [str(paths.OMLX_PY), "-I", "-B", str(script), "--src", str(snapshot_dir(model)),
           "--name", model.served_id, "--level", f"{float(conv['level']):g}"]
    for fix in model.prepare.get("tokenizer_fixes", []):
        cmd += ["--fix", fix]
    if allow_proxy:
        cmd.append("--allow-proxy")
    if "--no-register" not in script.read_text(encoding="utf-8"):
        raise PrepareError("scripts/oq_convert.py has no --no-register option; register() below makes the link "
                           "and the server writes model_settings.json into .modeleval-home per launch")
    return cmd + ["--no-register"]


def convert_checks(model: Model) -> int:
    """Everything convert needs before taking the lock; returns the estimated output bytes."""
    src = snapshot_dir(model)
    if not (src / "config.json").exists():
        raise PrepareError(f"no source snapshot at {paths.tilde(src)} (modeleval download {model.key})")
    if model.path.resolve() != (oq_out() / model.served_id).resolve():
        raise PrepareError(f"oq_convert.py writes {paths.tilde(oq_out() / model.served_id)}, but the registry says "
                           f"{paths.tilde(model.path)}")
    if model.path.exists():
        raise PrepareError(f"{paths.tilde(model.path)} exists; a checkpoint is never rewritten after conversion")
    for port in (PEER_PORT, paths.PORT):
        if port_listening(port):
            raise PrepareError(f"port {port} is listening: no model server may run during a conversion")
    level = float(model.prepare["convert"]["level"])
    ckpt = sum(f.stat().st_size for f in src.glob("*.safetensors"))
    out = int(ckpt * (level + 0.7) / 16)         # BF16 -> level bits + oQ's boost budget (hard cap)
    if free_bytes(model.path.parent) < out + DISK_MARGIN:
        raise PrepareError(f"need {(out + DISK_MARGIN) / 1e9:.0f} GB free for the output, "
                           f"have {free_bytes(model.path.parent) / 1e9:.0f} GB")
    return out


def _lock(args: list[str], run_id: str) -> tuple[int, str]:
    env = {**os.environ, "MODELEVAL_RUN_ID": run_id}
    p = subprocess.run([str(paths.SCRIPTS / "gpu_lock.sh"), *args], capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout + p.stderr).strip()


def convert(model: Model, allow_proxy: bool = False, eta_min: int = 15) -> Path:
    """oQ conversion under the GPU lock (purpose `accuracy only (untimed), convert`), then register."""
    cmd = convert_command(model, allow_proxy)
    out_bytes = convert_checks(model)
    run_id = time.strftime("convert-%Y%m%d-%H%M%S")
    purpose = f"modeleval {model.key} accuracy only (untimed), convert"
    progress.emit("LOCK-NOTICE", start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), timed_minutes=0, purpose=purpose,
                  memory=f"source {paths.tilde(snapshot_dir(model))} resident during the sensitivity pass",
                  output_gb=f"{out_bytes / 1e9:.1f}")
    rc, out = _lock(["acquire", purpose, str(eta_min)], run_id)
    if rc == 3:
        _lock(["release"], run_id)
    if rc != 0:
        raise PrepareError(f"GPU lock not acquired (gpu_lock.sh exit {rc}): {out[-200:]}")
    log = paths.REPO / "logs" / f"oq-convert-{model.served_id}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    p: subprocess.Popen | None = None
    old = {s: signal.signal(s, _interrupt) for s in _SIGNALS if signal.getsignal(s) is not signal.SIG_IGN}
    try:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        with log.open("w", encoding="utf-8") as f:
            # its own session: a signal to us must not leave a conversion running after the release
            p = subprocess.Popen(cmd, cwd=paths.REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, start_new_session=True)
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
            rc = p.wait()
    finally:
        try:
            if p is not None and p.poll() is None:
                kill_group(p.pid, signal.SIGTERM, 30)
                p.wait(timeout=10)
        finally:
            _lock(["release"], run_id)
            for s, h in old.items():
                signal.signal(s, h)
    if rc != 0:
        raise PrepareError(f"oq_convert.py exited {rc}; see {paths.tilde(log)}")
    register(model)
    return model.path


def register(model: Model) -> Path:
    """The models/hf-bench/<served_id> link to the checkpoint (relative when it lives under models/)."""
    if port_listening(paths.PORT):
        raise PrepareError(f"port {paths.PORT} is listening; register with the server stopped")
    if not (model.path / "config.json").exists():
        raise PrepareError(f"no checkpoint at {paths.tilde(model.path)}")
    link = model.link
    if link.is_symlink():
        if link.resolve() == model.path.resolve():
            return link
        raise PrepareError(f"{paths.tilde(link)} points at {paths.tilde(link.resolve())}, not {paths.tilde(model.path)}")
    if link.exists():
        raise PrepareError(f"{paths.tilde(link)} exists and is not a link")
    link.parent.mkdir(parents=True, exist_ok=True)
    root = paths.MODELS_DIR.parent
    inside = model.path.resolve().is_relative_to(root.resolve())
    # ../oq/<name> for our own conversions (as before); an absolute link to a checkpoint elsewhere
    link.symlink_to(Path(os.path.relpath(model.path, link.parent)) if inside else model.path)
    return link


def status(model: Model) -> dict:
    """What a model has: checkpoint, link, source snapshot, conversion record, runs."""
    link = model.link
    linked = "ok" if link.is_symlink() and link.resolve() == model.path.resolve() else (
        "elsewhere" if link.is_symlink() or link.exists() else "missing")
    events = []
    rec = model.path / "agent-model-eval.json"
    if rec.exists():
        import json
        try:
            events = [e.get("step") for e in json.loads(rec.read_text())["events"]]
        except (ValueError, KeyError):
            events = ["unreadable"]
    runs = sorted(p.name for p in (paths.RESULTS / model.key).glob("*") if p.is_dir()) if (paths.RESULTS / model.key).exists() else []
    return {"checkpoint": (model.path / "config.json").exists(), "link": linked,
            "source": (snapshot_dir(model) / "config.json").exists(), "prepared": events, "runs": runs}
