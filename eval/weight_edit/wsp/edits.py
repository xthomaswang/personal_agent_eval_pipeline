"""Arms: full copies of the original model, edited in place of the copy only, with an edit manifest.

The original model folder is never written. `edit apply` always edits relative to the ORIGINAL weights: it reads the
original tensors, runs the user's edit function, and writes a fresh model.safetensors into the arm folder.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from . import util
from .util import Refused

MANIFEST = "edit_manifest.json"
COPY_SKIP = {".cache", ".DS_Store"}
VOLATILE_KEYS = {"created_at", "applied_at"}


def arm_dir(arm: str) -> Path:
    return util.original_dir() if arm == "base" else util.edits_dir() / arm


def manifest_path(arm: str) -> Path:
    return arm_dir(arm) / MANIFEST


def load_manifest(arm: str) -> dict[str, Any] | None:
    if arm == "base":
        return None
    p = manifest_path(arm)
    return json.loads(p.read_text()) if p.exists() else None


def is_edited(arm: str) -> bool:
    m = load_manifest(arm)
    return bool(m and m.get("edited"))


def weights_key(arm: str) -> str:
    """'base' for the original (and for an arm that was only copied); else the sha256 of the edit manifest without its
    timestamps. Any edit changes the key, so cached vectors never leak across weights."""
    m = load_manifest(arm)
    if arm == "base" or not m or not m.get("edited"):
        return "base"
    return util.sha256_json({k: v for k, v in m.items() if k not in VOLATILE_KEYS})


def weights_file(arm: str) -> Path:
    return arm_dir(arm) / "model.safetensors"


def model_hashes(arm: str) -> dict[str, str | None]:
    m = load_manifest(arm)
    return {
        "safetensors_sha256": util.sha256_file_cached(weights_file(arm)),
        "manifest_sha256": util.sha256_bytes(manifest_path(arm).read_bytes()) if m else None,
        "weights_key": weights_key(arm),
    }


def _assert_outside_original(path: Path) -> None:
    orig = util.original_dir().resolve()
    p = path.resolve()
    if p == orig or orig in p.parents:
        raise Refused(f"refusing to write inside the original model folder: {path}")


def copy_arm(arm: str) -> Path:
    util.check_arm_name(arm)
    dst = arm_dir(arm)
    _assert_outside_original(dst)
    if dst.exists():
        raise Refused(f"arm {arm!r} already exists at {dst} (pick another name)")
    src = util.original_dir()
    dst.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        if p.name in COPY_SKIP:
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name, ignore=shutil.ignore_patterns(*COPY_SKIP))
        else:
            shutil.copy2(p, dst / p.name)
    manifest = {
        "arm": arm,
        "created_at": util.now_iso(),
        "edited": False,
        "original_dir": str(src),
        "original_safetensors_sha256": util.sha256_file_cached(src / "model.safetensors"),
        "tensors": {},
    }
    manifest_path(arm).write_text(json.dumps(manifest, indent=2))
    return dst


# --- tensors ---


def _tensor_sha(t: Any) -> str:
    import torch

    return util.sha256_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())


def _parse_args(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise Refused(f"bad --arg {p!r}: use KEY=VALUE")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def load_edit_function(edit_py: Path, args: dict[str, str]):
    """Import the edit file; it must define edit(tensors, read_acts). Its module-level ARGS holds the --arg strings."""
    spec = importlib.util.spec_from_file_location("wsp_user_edit", edit_py)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot import {edit_py}")
    module = importlib.util.module_from_spec(spec)
    module.ARGS = dict(args)  # type: ignore[attr-defined]
    spec.loader.exec_module(module)
    if not callable(getattr(module, "edit", None)):
        raise Refused(f"{edit_py} must define edit(tensors, read_acts)")
    module.ARGS = dict(args)  # in case the file reassigned ARGS at import
    return module.edit


def apply_edit(arm: str, edit_py: Path, arg_pairs: list[str] | None = None, replace: bool = False, acts_arm: str = "base") -> dict[str, Any]:
    """Run the edit function on CPU and write the result into the arm's model.safetensors; returns the manifest."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .acts import read_acts as _read_acts

    util.check_arm_name(arm)
    d = arm_dir(arm)
    _assert_outside_original(d)
    m = load_manifest(arm)
    if m is None:
        raise Refused(f"arm {arm!r} does not exist: run `wsp copy {arm}` first")
    if m.get("edited") and not replace:
        raise Refused(f"arm {arm!r} already has an edit; use --replace to redo it from the original weights, or copy a new arm")
    edit_py = Path(edit_py)
    args = _parse_args(arg_pairs or [])
    fn = load_edit_function(edit_py, args)

    src = util.original_dir() / "model.safetensors"
    originals: dict[str, Any] = {}
    with safe_open(str(src), framework="pt") as f:
        meta = f.metadata()
        for name in f.keys():
            originals[name] = f.get_tensor(name)
    np_tensors = {n: t.float().numpy().copy() for n, t in originals.items()}

    def read_acts(layer: int, arm: str = acts_arm, suite: str = "dev") -> np.ndarray:
        return _read_acts(layer, arm=arm, suite=suite)

    returned = fn(np_tensors, read_acts)
    if not isinstance(returned, dict):
        raise Refused("edit() must return a dict {tensor name: array}")

    unknown = sorted(set(returned) - set(originals))
    if unknown:
        raise Refused(f"edit touches tensors that are not in the model: {unknown[:5]}")
    new_tensors: dict[str, Any] = {}
    info: dict[str, Any] = {}
    unchanged = []
    for name, arr in returned.items():
        orig = originals[name]
        arr = np.asarray(arr)
        if tuple(arr.shape) != tuple(orig.shape):
            raise Refused(f"edit changed the shape of {name}: {tuple(orig.shape)} -> {tuple(arr.shape)}")
        if not np.isfinite(arr).all():
            raise Refused(f"edit produced non-finite values in {name}")
        new = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(orig.dtype)
        n_changed = int((new != orig).sum().item())
        if n_changed == 0:
            unchanged.append(name)
            continue
        new_tensors[name] = new
        info[name] = {
            "shape": list(orig.shape),
            "dtype": str(orig.dtype).replace("torch.", ""),
            "n_entries": int(orig.numel()),
            "n_changed": n_changed,
            "sha256_base": _tensor_sha(orig),
            "sha256_edited": _tensor_sha(new),
        }
    if not new_tensors:
        raise Refused("the edit changed no entry after rounding to the model dtype (is the step too small?)")

    full = {n: new_tensors.get(n, t) for n, t in originals.items()}
    tmp = d / "model.safetensors.tmp"
    save_file(full, str(tmp), metadata=meta or {"format": "pt"})
    os.replace(tmp, d / "model.safetensors")

    text = edit_py.read_text()
    manifest = {
        "arm": arm,
        "created_at": m.get("created_at"),
        "applied_at": util.now_iso(),
        "edited": True,
        "original_dir": str(util.original_dir()),
        "original_safetensors_sha256": m.get("original_safetensors_sha256"),
        "edited_safetensors_sha256": util.sha256_file(d / "model.safetensors"),
        "edit_file": str(edit_py.resolve()),
        "edit_file_sha256": util.sha256_bytes(text.encode()),
        "edit_file_text": text,
        "edit_args": args,
        "acts_arm": acts_arm,
        "n_tensors_changed": len(new_tensors),
        "n_entries_changed": sum(v["n_changed"] for v in info.values()),
        "returned_unchanged": sorted(unchanged),
        "tensors": info,
    }
    manifest_path(arm).write_text(json.dumps(manifest, indent=2))
    return manifest
