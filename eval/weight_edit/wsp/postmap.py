"""Closed-form post-maps on embeddings: fitted on dev-suite corpus vectors only, applied to queries and documents.

pca-remove: all-but-the-top, v' = (v - mu) (I - V_k V_k^T) with V_k the top-k principal directions.
whiten:     v' = (v - mu) C_s^(-1/2) with C_s = (1 - s) C + s * mean(eig(C)) * I (shrinkage toward isotropy).
Both are followed by L2 re-normalization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import util


def fit(kind: str, x: np.ndarray, k: int = 1, shrink: float = 0.1) -> dict[str, np.ndarray | str | float | int]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("need a 2-D array with at least two vectors to fit a post-map")
    mu = x.mean(axis=0)
    xc = x - mu
    cov = xc.T @ xc / (len(x) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(-evals)
    evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
    d = x.shape[1]
    if kind == "pca-remove":
        if not 1 <= k < d:
            raise ValueError(f"--k must be in 1..{d - 1}")
        v = evecs[:, :k]
        m = np.eye(d) - v @ v.T
    elif kind == "whiten":
        if not 0.0 <= shrink <= 1.0:
            raise ValueError("--shrink must be in [0, 1]")
        lam = (1 - shrink) * evals + shrink * evals.mean()
        m = (evecs / np.sqrt(np.maximum(lam, 1e-12))) @ evecs.T
    else:
        raise ValueError(f"unknown post-map kind {kind!r}")
    return {
        "kind": kind,
        "k": int(k),
        "shrink": float(shrink),
        "mu": mu.astype(np.float32),
        "matrix": m.astype(np.float32),
        "n_fit": int(len(x)),
    }


def save(path: Path, pm: dict, meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in pm.items()}, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    pm = {k: z[k] for k in z.files}
    for key in ("kind",):
        pm[key] = str(pm[key])
    return pm


def apply(pm: dict, x: np.ndarray) -> np.ndarray:
    """Center, map, re-normalize (float32)."""
    y = (np.asarray(x, dtype=np.float32) - pm["mu"]) @ pm["matrix"]
    return util.l2norm(y)


def describe(pm: dict) -> str:
    if pm["kind"] == "pca-remove":
        return f"pca-remove k={int(pm['k'])}"
    return f"whiten shrink={float(pm['shrink'])}"
