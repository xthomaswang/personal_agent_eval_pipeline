"""The harness epoch (DESIGN §6.1, §6.5): a pipeline-owned snapshot of the harness code and configs,
identified by content, plus the live parts that are re-checked before every task.

Why a snapshot: personal-agent-bench and harness-smoke belong to other sessions and commit several
times a day, and the agent configs under ~/Developer/mlsys/agent are edited by hand. Importing the
live trees would change the harness under a running leg and write bytecode into them. So an epoch
copies `bench/` and `smoke/` + `shims/` at a chosen commit with `git archive` into the gitignored
`vendor/<epoch>/`, together with the Claude Code config files and the Hermes home.

Why content, not commits: the epoch id hashes the git *tree* hashes of the copied directories, the
config bytes, the pipeline's uv.lock files and the live pins, so another session's commit to its
docs or results never forces a new epoch (and with it a reference re-run).

Layout of vendor/<epoch>/:
    epoch.json                              id, hashed components, pins, per-file sha256
    personal-agent-bench/bench/...          git archive <pab_sha> bench
    harness-smoke/smoke/..., shims/...      git archive <smoke_sha> smoke shims
    configs/claude-code/claude-code.env     byte copy of agent/configs/claude-code/claude-code.env
    configs/homes/claude-code/settings.json byte copy of agent/homes/claude-code/settings.json
    configs/homes/hermes/...                agent/homes/hermes without logs, sessions and caches;
                                            tools/ and installs/ are symlinks to the live store

The only git commands run on another session's repository are `rev-parse`, `--no-optional-locks
status --porcelain`, `diff-index --quiet HEAD` and `archive`, always with GIT_OPTIONAL_LOCKS=0
(`_git` enforces this).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from modeleval import paths

EPOCH_FORMAT = 1
PINS_FILE = paths.REPO / "pins.toml"
EPOCH_FILE = "epoch.json"
ID_LENGTH = 12

# bench.harness's own lists (harness.py:51-52). The runner's selftest asserts they still match.
HERMES_SHARED = ("tools", "installs")
HERMES_SKIP = ("logs", "sessions", "cache", "audio_cache", "image_cache", "backups")
HERMES_MUST_BE_EMPTY = ("memories", "skills")
HERMES_FACTS = "tools/facts.json"

# Where each vendored part lives inside vendor/<epoch>/
PAB_DIR = "personal-agent-bench"
SMOKE_DIR = "harness-smoke"
CONFIGS_DIR = "configs"
CLAUDE_ENV_COPY = "configs/claude-code/claude-code.env"
CLAUDE_SETTINGS_COPY = "configs/homes/claude-code/settings.json"
HERMES_HOME_COPY = "configs/homes/hermes"

_GIT_ENV = {"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}


class EpochError(RuntimeError):
    """The epoch cannot be created or loaded as pinned."""


# --- small helpers ---


def _path(value: str | os.PathLike[str]) -> Path:
    """A pins path: `~` expanded; a relative path is relative to the repo."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else paths.REPO / p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tree_digest(root: Path) -> str:
    """sha256 over a directory's entries: relative path, kind, and content hash (files, with the exec
    bit) or `link` (symlinks are not followed: tools/ and installs/ are pinned by facts.json)."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for name in sorted(dirnames + filenames):
            p = base / name
            rel = p.relative_to(root).as_posix()
            mode = p.lstat().st_mode
            if stat.S_ISLNK(mode):
                entries.append(f"{rel}\0link")
            elif stat.S_ISDIR(mode):
                entries.append(f"{rel}\0dir")
            elif stat.S_ISREG(mode):
                exe = "x" if mode & stat.S_IXUSR else "-"
                entries.append(f"{rel}\0file\0{exe}\0{sha256_file(p)}")
            else:
                entries.append(f"{rel}\0other")
        dirnames[:] = [d for d in dirnames if not (base / d).is_symlink()]
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    """Run one of the git commands DESIGN §6.1 allows on another session's repository."""
    allowed = args[:1] in (("rev-parse",), ("diff-index",), ("archive",)) or args[:2] == (
        "--no-optional-locks",
        "status",
    )
    if not allowed:
        raise ValueError(f"git {' '.join(args)} is not an allowed command on a peer repository")
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, env={**os.environ, **_GIT_ENV}, timeout=120
    )
    if check and proc.returncode != 0:
        raise EpochError(f"git -C {paths.tilde(repo)} {' '.join(args)}: {proc.stderr.decode(errors='replace').strip()}")
    return proc


def _rev(repo: Path, rev: str) -> str | None:
    proc = _git(repo, "rev-parse", "--verify", "--quiet", rev, check=False)
    return proc.stdout.decode().strip() if proc.returncode == 0 else None


def repo_state(repo: Path) -> tuple[str | None, bool | None]:
    """(HEAD, dirty) without refreshing the index; (None, None) when it is not a git checkout."""
    if not (repo / ".git").exists():
        return None, None
    head = _rev(repo, "HEAD")
    # the pipeline's own .modeleval-home inside the oMLX checkout is not a change of that checkout
    status = _git(repo, "--no-optional-locks", "status", "--porcelain", *paths.status_pathspec(repo), check=False)
    if head is None or status.returncode != 0:
        return None, None
    return head, bool(status.stdout.strip())


def index_path(repo: Path) -> Path | None:
    """The repository's (or worktree's) index file, for the hygiene audit."""
    proc = _git(repo, "rev-parse", "--git-path", "index", check=False)
    if proc.returncode != 0:
        return None
    p = Path(proc.stdout.decode().strip())
    return p if p.is_absolute() else repo / p


def index_stamps(repos: list[Path]) -> dict[str, list[int] | None]:
    """mtime_ns and inode of each repository's index: a change across one of our own operations means
    we wrote into another session's repository (DESIGN §6.1, hygiene audit)."""
    out: dict[str, list[int] | None] = {}
    for repo in repos:
        index = index_path(repo) if repo.exists() else None
        try:
            st = index.stat() if index else None
        except FileNotFoundError:
            st = None
        out[str(index or repo)] = [st.st_mtime_ns, st.st_ino] if st else None
    return out


def _file_stat(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino}


def _direct_url_target(bin_path: Path, dist: str) -> Path | None:
    """The source tree an editable install in bin_path's venv points at (dist-info direct_url.json)."""
    venv = bin_path.parent.parent
    for record in sorted(venv.glob(f"lib/python*/site-packages/{dist}-*.dist-info/direct_url.json")):
        url = json.loads(record.read_text()).get("url", "")
        if url.startswith("file://"):
            return Path(url.removeprefix("file://"))
    return None


# --- pins ---


def normalize_pins(raw: dict[str, Any]) -> dict[str, Any]:
    """pins.toml as typed values: paths expanded, defaults filled."""
    snap = raw["snapshot"]
    data = raw["data"]
    root = _path(data["root"])
    return {
        "epoch_id": (raw.get("epoch") or {}).get("id"),
        "pab": {"repo": _path(snap["personal_agent_bench"]["repo"]), "sha": snap["personal_agent_bench"]["sha"]},
        "smoke": {"repo": _path(snap["harness_smoke"]["repo"]), "sha": snap["harness_smoke"]["sha"]},
        "agent_root": _path(raw["configs"]["agent_root"]),
        "claude_env": raw["configs"]["claude_env"],
        "claude_settings": raw["configs"]["claude_settings"],
        "hermes_home": raw["configs"]["hermes_home"],
        "localmelo": {
            "repo": _path(raw["localmelo"]["repo"]),
            "sha": raw["localmelo"]["sha"],
            "dist": raw["localmelo"].get("dist"),
        },
        "hermes": {
            "repo": _path(raw["hermes"]["repo"]),
            "tag": raw["hermes"]["tag"],
            "sha": raw["hermes"]["sha"],
            "bin": _path(raw["hermes"]["bin"]),
            "dist": raw["hermes"].get("dist"),
        },
        "claude_code": {"version": raw["claude_code"]["version"], "bin": _path(raw["claude_code"]["bin"])},
        "data": {
            "root": root,
            "aider_polyglot": data["aider_polyglot"],
            "mcpmark": data["mcpmark"],
            # smoke.paths' layout, so both sides agree on where the pinned data lives
            "aider_polyglot_dir": root / "aider-polyglot-benchmark" / data["aider_polyglot"][:12],
            "mcpmark_dir": root / "mcpmark" / data["mcpmark"][:12],
            "mcpmark_envs_manifest": root / "mcpmark" / "filesystem-envs" / "manifest.json",
        },
        "envs": {"capability": _path(raw["envs"]["capability"]), "agent_tools": _path(raw["envs"]["agent_tools"])},
        "shims": _path(raw["envs"]["shims"]),
        "ws_root": _path(raw["workspace"]["ws_root"]),
    }


def load_pins(path: Path = PINS_FILE) -> tuple[dict[str, Any], dict[str, Any]]:
    """(raw, normalized) pins."""
    raw = tomllib.loads(path.read_text())
    return raw, normalize_pins(raw)


# --- live checks ---


def _repo_problems(name: str, repo: Path, sha: str) -> list[str]:
    head, dirty = repo_state(repo)
    if head is None:
        return [f"{name}: no git checkout at {paths.tilde(repo)}"]
    found = []
    if head != sha:
        found.append(f"{name}: {paths.tilde(repo)} is at {head[:10]}, pinned {sha[:10]}")
    if dirty:
        found.append(f"{name}: {paths.tilde(repo)} has uncommitted changes")
    return found


def _live_problems(pins: dict[str, Any], components: dict[str, Any] | None, claude_stat: dict | None) -> list[str]:
    """What differs from the pins (and, with an epoch, from what the epoch recorded)."""
    found: list[str] = []
    hermes = pins["hermes"]
    found += _repo_problems("hermes-agent", hermes["repo"], hermes["sha"])
    if _rev(hermes["repo"], f"{hermes['tag']}^{{commit}}") != hermes["sha"]:
        found.append(f"hermes-agent: tag {hermes['tag']} does not point at {hermes['sha'][:10]}")
    if not hermes["bin"].exists():
        found.append(f"hermes: {paths.tilde(hermes['bin'])} is missing")
    elif hermes["dist"] and _direct_url_target(hermes["bin"], hermes["dist"]) != hermes["repo"]:
        found.append(f"hermes: {paths.tilde(hermes['bin'])} is not an editable install of {paths.tilde(hermes['repo'])}")

    melo = pins["localmelo"]
    found += _repo_problems("localmelo", melo["repo"], melo["sha"])
    melo_bin = melo["repo"] / ".venv" / "bin" / "melo"
    if not melo_bin.exists():
        found.append(f"localmelo: {paths.tilde(melo_bin)} is missing")
    elif melo["dist"] and _direct_url_target(melo_bin, melo["dist"]) != melo["repo"]:
        found.append(f"localmelo: {paths.tilde(melo_bin)} is not an editable install of {paths.tilde(melo['repo'])}")

    claude = pins["claude_code"]
    if not claude["bin"].is_file():
        found.append(f"claude code: {paths.tilde(claude['bin'])} is missing")
    elif claude["bin"].name != claude["version"]:
        found.append(f"claude code: {paths.tilde(claude['bin'])} is not version {claude['version']}")
    elif claude_stat is not None and _file_stat(claude["bin"]) != claude_stat:
        found.append(f"claude code: {paths.tilde(claude['bin'])} changed since the epoch was created")

    data = pins["data"]
    found += _repo_problems("aider polyglot data", data["aider_polyglot_dir"], data["aider_polyglot"])
    found += _repo_problems("mcpmark data", data["mcpmark_dir"], data["mcpmark"])
    manifest = data["mcpmark_envs_manifest"]
    if not manifest.is_file():
        found.append(f"mcpmark: {paths.tilde(manifest)} is missing")

    facts = pins["agent_root"] / pins["hermes_home"] / HERMES_FACTS
    if not facts.is_file():
        found.append(f"hermes: {paths.tilde(facts)} is missing")

    for name, env in pins["envs"].items():
        if not (env / "uv.lock").is_file():
            found.append(f"envs/{name}: uv.lock is missing")
        if not (env / ".venv" / "bin" / "python").exists():
            found.append(f"envs/{name}: .venv is missing (uv sync in {paths.tilde(env)})")

    if components is not None and not found:
        current = {
            "hermes_tools_facts": sha256_file(facts),
            "mcpmark_envs_manifest": sha256_file(manifest),
            "env_locks": {name: sha256_file(env / "uv.lock") for name, env in pins["envs"].items()},
        }
        for key, value in current.items():
            if components[key] != value:
                found.append(f"{key} changed since the epoch was created")
    return found


def verify_live(epoch: dict[str, Any]) -> list[str]:
    """Re-check the live parts of an epoch (Hermes, localmelo, the Claude Code binary, the data, the
    shared Hermes tools manifest, the pipeline's lock files). Empty list = unchanged."""
    return _live_problems(epoch["pins"], epoch["components"], epoch["claude_code_stat"])


# --- creation ---


def _archive(repo: Path, sha: str, subdirs: list[str], dest: Path) -> None:
    """`git archive <sha> <subdirs>` extracted into dest (tar's `data` filter: no links out, no devices)."""
    proc = _git(repo, "archive", "--format=tar", sha, *subdirs)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(dest, filter="data")


def _env_assignments(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _copy_hermes_home(src: Path, dest: Path) -> None:
    for name in HERMES_MUST_BE_EMPTY:
        if (src / name).is_dir() and any((src / name).iterdir()):
            raise EpochError(f"{paths.tilde(src / name)} is not empty: Hermes would start with memories or skills")
    env = src / ".env"
    if env.is_file() and _env_assignments(env.read_text()):
        raise EpochError(f"{paths.tilde(env)} assigns variables; the benchmark home must carry no secrets")
    shutil.copytree(src, dest, symlinks=True, ignore=shutil.ignore_patterns(*HERMES_SHARED, *HERMES_SKIP))
    for name in HERMES_SHARED:
        if (src / name).exists():
            (dest / name).symlink_to(src / name)


def _snapshot_files(root: Path) -> dict[str, str]:
    """sha256 of every regular file in the snapshot (symlinked live stores are not descended)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not (base / d).is_symlink())
        for name in sorted(filenames):
            p = base / name
            rel = p.relative_to(root).as_posix()
            if rel == EPOCH_FILE or p.is_symlink():
                continue
            out[rel] = sha256_file(p)
    return out


def _components(pins: dict[str, Any], pab_sha: str, smoke_sha: str, stage: Path) -> dict[str, Any]:
    data = pins["data"]
    return {
        "format": EPOCH_FORMAT,
        "trees": {
            "bench": _rev(pins["pab"]["repo"], f"{pab_sha}:bench"),
            "smoke": _rev(pins["smoke"]["repo"], f"{smoke_sha}:smoke"),
            "shims": _rev(pins["smoke"]["repo"], f"{smoke_sha}:shims"),
        },
        "configs": {
            "claude-code.env": sha256_file(stage / CLAUDE_ENV_COPY),
            "settings.json": sha256_file(stage / CLAUDE_SETTINGS_COPY),
            "hermes-home": tree_digest(stage / HERMES_HOME_COPY),
        },
        "hermes_tools_facts": sha256_file(pins["agent_root"] / pins["hermes_home"] / HERMES_FACTS),
        "env_locks": {name: sha256_file(env / "uv.lock") for name, env in pins["envs"].items()},
        "agent_shims": tree_digest(pins["shims"]),
        "localmelo": pins["localmelo"]["sha"],
        "hermes": {"tag": pins["hermes"]["tag"], "sha": pins["hermes"]["sha"]},
        "claude_code": pins["claude_code"]["version"],
        "data": {"aider_polyglot": data["aider_polyglot"], "mcpmark": data["mcpmark"]},
        "mcpmark_envs_manifest": sha256_file(data["mcpmark_envs_manifest"]),
        "ws_root": paths.tilde(pins["ws_root"]),
    }


def create_epoch(
    pab_sha: str,
    smoke_sha: str,
    *,
    pins_raw: dict[str, Any] | None = None,
    vendor_root: Path = paths.VENDOR,
) -> str:
    """Snapshot the harness code at the two commits and the configs into vendor/<epoch>/; return the id.

    Refuses (EpochError) when a live part differs from the pins. Idempotent: an intact snapshot with
    the same id is kept.
    """
    raw = pins_raw if pins_raw is not None else tomllib.loads(PINS_FILE.read_text())
    pins = normalize_pins(raw)
    problems = _live_problems(pins, None, None)
    if problems:
        raise EpochError("live parts differ from the pins: " + "; ".join(problems))
    pab_full = _rev(pins["pab"]["repo"], f"{pab_sha}^{{commit}}")
    smoke_full = _rev(pins["smoke"]["repo"], f"{smoke_sha}^{{commit}}")
    if pab_full is None or smoke_full is None:
        raise EpochError(f"unknown commit: personal-agent-bench {pab_sha} or harness-smoke {smoke_sha}")

    vendor_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=vendor_root))
    try:
        _archive(pins["pab"]["repo"], pab_full, ["bench"], stage / PAB_DIR)
        _archive(pins["smoke"]["repo"], smoke_full, ["smoke", "shims"], stage / SMOKE_DIR)
        agent = pins["agent_root"]
        (stage / CLAUDE_ENV_COPY).parent.mkdir(parents=True)
        shutil.copyfile(agent / pins["claude_env"], stage / CLAUDE_ENV_COPY)
        (stage / CLAUDE_SETTINGS_COPY).parent.mkdir(parents=True)
        shutil.copyfile(agent / pins["claude_settings"], stage / CLAUDE_SETTINGS_COPY)
        _copy_hermes_home(agent / pins["hermes_home"], stage / HERMES_HOME_COPY)

        components = _components(pins, pab_full, smoke_full, stage)
        epoch_id = _sha256_json(components)[:ID_LENGTH]
        record = {
            "id": epoch_id,
            "created_at": dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "components": components,
            "snapshot": {"personal_agent_bench": pab_full, "harness_smoke": smoke_full},
            "hermes_lists": {"shared": list(HERMES_SHARED), "skip": list(HERMES_SKIP)},
            "claude_code_sha256": sha256_file(pins["claude_code"]["bin"]),
            "claude_code_stat": _file_stat(pins["claude_code"]["bin"]),
            "pins": raw,
            "files": _snapshot_files(stage),
        }
        (stage / EPOCH_FILE).write_text(json.dumps(record, indent=2) + "\n")

        dest = vendor_root / epoch_id
        if dest.exists():
            try:
                if not snapshot_problems(load_epoch(epoch_id, vendor_root)):
                    return epoch_id
            except (EpochError, OSError, ValueError, KeyError):
                pass
            shutil.rmtree(dest)
        stage.rename(dest)
        return epoch_id
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_epoch(epoch_id: str, vendor_root: Path = paths.VENDOR) -> dict[str, Any]:
    """vendor/<id>/epoch.json, with `pins` normalized and `dir` set."""
    root = vendor_root / epoch_id
    try:
        record = json.loads((root / EPOCH_FILE).read_text())
    except FileNotFoundError:
        raise EpochError(
            f"no epoch {epoch_id} in {paths.tilde(vendor_root)}; "
            "modeleval.capability.epoch.ensure_epoch() recreates it from the commits pins.toml names"
        ) from None
    if record.get("id") != epoch_id:
        raise EpochError(f"{paths.tilde(root / EPOCH_FILE)} names epoch {record.get('id')}, not {epoch_id}")
    return {**record, "pins_raw": record["pins"], "pins": normalize_pins(record["pins"]), "dir": root}


def snapshot_problems(epoch: dict[str, Any]) -> list[str]:
    """Files of vendor/<epoch>/ that were changed, removed or added since creation (bytecode included)."""
    expected: dict[str, str] = epoch["files"]
    actual = _snapshot_files(epoch["dir"])
    found = [f"{rel}: changed" for rel in expected if rel in actual and actual[rel] != expected[rel]]
    found += [f"{rel}: missing" for rel in expected if rel not in actual]
    found += [f"{rel}: not part of the snapshot" for rel in actual if rel not in expected]
    return found


def ensure_epoch(pins_path: Path = PINS_FILE, vendor_root: Path = paths.VENDOR) -> str:
    """The epoch pins.toml names, (re)created from its snapshot commits when vendor/ lacks it.

    Refuses when the pinned commits and configs no longer hash to the pinned id: a new epoch is a
    deliberate edit of pins.toml, never a side effect.
    """
    raw, pins = load_pins(pins_path)
    pinned = pins["epoch_id"]
    if pinned:
        try:
            epoch = load_epoch(pinned, vendor_root)
            if not snapshot_problems(epoch):
                return pinned
        except EpochError:
            pass
    made = create_epoch(pins["pab"]["sha"], pins["smoke"]["sha"], pins_raw=raw, vendor_root=vendor_root)
    if pinned and made != pinned:
        raise EpochError(f"pins.toml names epoch {pinned}, but the pinned inputs now hash to {made}")
    return made
