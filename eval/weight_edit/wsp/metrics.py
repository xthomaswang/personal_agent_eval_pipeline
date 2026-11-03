"""Retrieval metrics on rankings (pure numpy, per query).

A ranking is an int array of doc indices, best first. Relevance is a dict {doc index: grade > 0}.
nDCG uses linear gain and a 1/log2(rank+1) discount (same as trec_eval ndcg_cut, which MTEB uses).
"""

from __future__ import annotations

import numpy as np

MAIN_METRIC = "ndcg@10"

METRICS_BY_KIND = {
    "mteb": ["ndcg@10", "recall@100", "mrr@10"],
    "lme": ["recall_any@5", "recall_any@10", "ndcg@10", "session_recall@5", "session_any@5"],
    "locomo": ["recall@5", "recall@10", "ndcg@10"],
}
TOP_K = {"mteb": 100, "lme": 10, "locomo": 10}


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best scores per row, best first; ties broken by the lower index."""
    n_docs = scores.shape[1]
    k = min(k, n_docs)
    if k < n_docs:
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        part = np.tile(np.arange(n_docs), (scores.shape[0], 1))
    part_scores = np.take_along_axis(scores, part, axis=1)
    out = np.empty_like(part)
    for i in range(part.shape[0]):
        order = np.lexsort((part[i], -part_scores[i]))
        out[i] = part[i][order]
    return out


def ndcg_at_k(ranking: np.ndarray, rel: dict[int, float], k: int) -> float:
    if not rel:
        return 0.0
    gains = np.array([rel.get(int(d), 0.0) for d in ranking[:k]], dtype=np.float64)
    disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * disc).sum())
    ideal = np.sort(np.array(list(rel.values()), dtype=np.float64))[::-1][:k]
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: np.ndarray, rel: dict[int, float], k: int) -> float:
    if not rel:
        return 0.0
    hit = sum(1 for d in ranking[:k] if int(d) in rel)
    return hit / len(rel)


def recall_any_at_k(ranking: np.ndarray, rel: dict[int, float], k: int) -> float:
    return 1.0 if any(int(d) in rel for d in ranking[:k]) else 0.0


def mrr_at_k(ranking: np.ndarray, rel: dict[int, float], k: int) -> float:
    for i, d in enumerate(ranking[:k]):
        if int(d) in rel:
            return 1.0 / (i + 1)
    return 0.0


def session_recall_at_k(ranking: np.ndarray, doc_session: list[str], gold_sessions: set[str], k: int) -> tuple[float, float]:
    """(fraction of gold sessions covered, any gold session covered) by the sessions of the top-k docs."""
    if not gold_sessions:
        return 0.0, 0.0
    seen = {doc_session[int(d)] for d in ranking[:k]}
    cov = len(seen & gold_sessions)
    return cov / len(gold_sessions), 1.0 if cov else 0.0


def query_metrics(
    kind: str,
    ranking: np.ndarray,
    rel: dict[int, float],
    doc_session: list[str] | None = None,
    gold_sessions: set[str] | None = None,
) -> dict[str, float]:
    """All metrics of a task kind for one query."""
    if kind == "mteb":
        return {
            "ndcg@10": ndcg_at_k(ranking, rel, 10),
            "recall@100": recall_at_k(ranking, rel, 100),
            "mrr@10": mrr_at_k(ranking, rel, 10),
        }
    if kind == "locomo":
        return {
            "recall@5": recall_at_k(ranking, rel, 5),
            "recall@10": recall_at_k(ranking, rel, 10),
            "ndcg@10": ndcg_at_k(ranking, rel, 10),
        }
    if kind == "lme":
        frac, any_ = session_recall_at_k(ranking, doc_session or [], gold_sessions or set(), 5)
        return {
            "recall_any@5": recall_any_at_k(ranking, rel, 5),
            "recall_any@10": recall_any_at_k(ranking, rel, 10),
            "ndcg@10": ndcg_at_k(ranking, rel, 10),
            "session_recall@5": frac,
            "session_any@5": any_,
        }
    raise ValueError(f"unknown task kind {kind!r}")


def mean_metrics(per_query: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) if len(v) else float("nan") for k, v in per_query.items()}


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of paired differences."""
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
