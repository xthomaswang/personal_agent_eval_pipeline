"""Example closed-form edit: remove the top principal direction of the residual stream from what later layers write.

Pipeline (see README.md):
    wsp pass acts --arm base --layers 16 --limit 256      # the "read" pass (dev texts only)
    wsp copy demo
    wsp edit apply demo edits/example_remove_direction.py --arg layer=16 --arg a=0.5
    wsp eval --arm demo --suite dev --device mps

For depth `layer` (number of blocks applied) take the top principal direction u (unit vector, size hidden) of the
last-token residual activations dumped by `wsp acts`. For every block l >= layer set, on both output projections that
write into the residual stream (self_attn.o_proj and mlp.down_proj):

    W' = W - a * u (u^T W)

so the component of what the block writes along u is scaled by (1 - a). We do not expect this to help; it only shows the
edit(tensors, read_acts) interface end to end.

Arguments (strings from `--arg KEY=VALUE`, read from the module-level ARGS dict set by the loader):
    layer   depth of the activations and first edited block (default 16)
    a       strength in [0, 1] (default 0.5)
    center  1: principal direction of mean-centered activations, 0: of the raw second moment (default 0)
"""

from __future__ import annotations

import re

import numpy as np

ARGS: dict[str, str] = {}  # the loader replaces this with the --arg values


def top_direction(x: np.ndarray, center: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[0]


def edit(tensors: dict[str, np.ndarray], read_acts) -> dict[str, np.ndarray]:
    layer = int(ARGS.get("layer", 16))
    a = float(ARGS.get("a", 0.5))
    center = ARGS.get("center", "0") == "1"
    n_blocks = 1 + max(int(m.group(1)) for k in tensors if (m := re.match(r"layers\.(\d+)\.", k)))
    u = top_direction(read_acts(layer), center).astype(np.float32)
    out: dict[str, np.ndarray] = {}
    for block in range(layer, n_blocks):
        for name in (f"layers.{block}.self_attn.o_proj.weight", f"layers.{block}.mlp.down_proj.weight"):
            w = tensors[name]  # (hidden, in): rows are residual-stream directions
            out[name] = w - a * np.outer(u, u @ w)
    return out
