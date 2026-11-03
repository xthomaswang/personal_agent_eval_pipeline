"""The "read" pass: dump last-token residual-stream activations of dev texts, and read them back inside an edit.

Layout: state/acts/<arm>/<suite>.meta.json (texts, kinds, layers, ...) and <suite>.L<k>.npy, one (n_texts, hidden) float32
array per depth k (k = number of blocks applied; see model.py). Only the dev suite can be dumped.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np

from . import edits, suites, util
from .util import Refused


def acts_dir(arm: str) -> Path:
    return util.state_dir() / "acts" / arm


def parse_layers(spec: str, n_layers: int = 28) -> list[int]:
    """'0,8,16' or '8-20:4' (start-end inclusive, step) or a mix, e.g. '0,12-28:8'."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", part)
        if m:
            a, b, step = int(m[1]), int(m[2]), int(m[3] or 1)
            out += list(range(a, b + 1, step))
        elif part.isdigit():
            out.append(int(part))
        else:
            raise Refused(f"bad --layers item {part!r}")
    out = sorted(set(out))
    bad = [k for k in out if not 0 <= k <= n_layers]
    if not out or bad:
        raise Refused(f"--layers must be depths in 0..{n_layers}, got {out if not bad else bad}")
    return out


def sample_texts(tasks: list[suites.Task], n: int, kind: str = "both", seed: int = 0) -> tuple[list[str], list[str]]:
    """Seeded sample of n unique dev strings (queries as the model sees them, docs as the model sees them)."""
    queries: list[str] = []
    docs: list[str] = []
    for t in tasks:
        q, d = t.strings()
        queries += q
        docs += d
    queries = sorted(set(queries))
    docs = sorted(set(docs))
    rng = random.Random(f"acts:{seed}")
    if kind == "queries":
        nq, nd = min(n, len(queries)), 0
    elif kind == "docs":
        nq, nd = 0, min(n, len(docs))
    elif kind == "both":
        nq = min(n // 4, len(queries))
        nd = min(n - nq, len(docs))
    else:
        raise Refused(f"--kind must be both, queries or docs (got {kind!r})")
    sq = rng.sample(queries, nq)
    sd = rng.sample(docs, nd)
    return sq + sd, ["query"] * len(sq) + ["doc"] * len(sd)


def dump_acts(arm: str, layers: list[int], limit: int, kind: str = "both", device: str = "cpu", dev_lme: int = util.DEFAULT_DEV_LME, suite: str = "dev") -> dict:
    if suite != "dev":
        raise Refused("activations can only be dumped for the dev suite")
    from .model import Encoder, dtype_name

    t0 = time.time()
    tasks = suites.load_suite("dev", dev_lme)
    texts, kinds = sample_texts(tasks, limit, kind)
    text_hash = util.sha256_texts(texts)
    out = acts_dir(arm)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / f"{suite}.meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else None
    keep_layers: list[int] = []
    if old and old.get("text_hash") == text_hash and old.get("weights_key") == edits.weights_key(arm):
        keep_layers = list(old.get("layers", []))
    else:
        for p in out.glob(f"{suite}.L*.npy"):
            p.unlink()
    enc = Encoder(edits.arm_dir(arm), device)
    acts = enc.acts(texts, util.MAX_DOC_TOKENS, layers)
    for k, a in acts.items():
        np.save(out / f"{suite}.L{k}.npy", a)
    meta = {
        "arm": arm,
        "suite": suite,
        "layers": sorted(set(keep_layers) | set(layers)),
        "n_texts": len(texts),
        "text_hash": text_hash,
        "kinds": kinds,
        "texts": texts,
        "weights_key": edits.weights_key(arm),
        "device": device,
        "dtype": dtype_name(device),
        "dev_lme": dev_lme,
        "hidden": int(next(iter(acts.values())).shape[1]),
        "depth_convention": "k = number of transformer blocks applied (0 = embeddings, 28 = last block output, before the final norm)",
        "wall_time_s": round(time.time() - t0, 2),
        "env": util.env_info(),
    }
    meta_path.write_text(json.dumps(meta))
    return meta


def read_acts(layer: int, arm: str = "base", suite: str = "dev") -> np.ndarray:
    """Activations (n_texts, hidden) dumped by `wsp acts` / `wsp pass acts` at depth `layer`."""
    p = acts_dir(arm) / f"{suite}.L{layer}.npy"
    if not p.exists():
        meta_path = acts_dir(arm) / f"{suite}.meta.json"
        have = json.loads(meta_path.read_text()).get("layers", []) if meta_path.exists() else []
        raise Refused(f"no activations for arm={arm} depth={layer} (dumped depths: {have}); run `wsp pass acts` or `wsp acts` first")
    return np.load(p)
