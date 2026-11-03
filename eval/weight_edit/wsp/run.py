"""Evaluation: vector cache, scoring, result files, the compact table, comparisons and suite statistics."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import edits, metrics, postmap, suites, util
from .util import Refused

REFERENCE_FILE = util.PKG_ROOT / "reference_scores.json"


# --- vector cache ---


def cache_base(arm: str, suite: str, task: suites.Task, dtype: str) -> str:
    """Path prefix of a task's cache files: state/cache/<weights_key>/<suite>.<task>.<text hash>.<dtype>"""
    key = edits.weights_key(arm)[:16]
    return str(util.state_dir() / "cache" / key / f"{suite}.{task.name}.{task.text_hash()[:12]}.{dtype}")


def find_cached(arm: str, suite: str, task: suites.Task, dtypes: list[str]) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Cached (query vectors, doc vectors, dtype) for the first dtype that has a file, else None."""
    for dt in dtypes:
        base = cache_base(arm, suite, task, dt)
        npy, meta = Path(base + ".npy"), Path(base + ".json")
        if npy.exists() and meta.exists():
            m = json.loads(meta.read_text())
            if m["text_hash"] != task.text_hash():
                continue
            a = np.load(npy)
            return a[: m["n_queries"]], a[m["n_queries"] :], dt
    return None


def save_cached(arm: str, suite: str, task: suites.Task, dtype: str, q: np.ndarray, d: np.ndarray, extra: dict[str, Any]) -> None:
    base = cache_base(arm, suite, task, dtype)
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    npy, meta = Path(base + ".npy"), Path(base + ".json")
    np.save(npy, np.vstack([q, d]).astype(np.float32))
    meta.write_text(json.dumps({"n_queries": len(q), "n_docs": len(d), "text_hash": task.text_hash(), "dtype": dtype, "weights_key": edits.weights_key(arm), **extra}))


class LazyEncoder:
    """Loads the model only when a task has no cached vectors."""

    def __init__(self, arm: str, device: str):
        self.arm, self.device = arm, device
        self._enc = None

    def get(self):
        if self._enc is None:
            from .model import Encoder

            self._enc = Encoder(edits.arm_dir(self.arm), self.device)
        return self._enc


def task_vectors(arm: str, suite: str, task: suites.Task, device: str, lazy: LazyEncoder | None, dtype: str, accept_dtypes: list[str]) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    hit = find_cached(arm, suite, task, accept_dtypes)
    if hit is not None:
        return hit[0], hit[1], hit[2], {"cached": True}
    if lazy is None:
        raise Refused(
            f"no cached vectors for arm={arm} suite={suite} task={task.name} (limit/dev-lme must match the run that made them); "
            "run `wsp eval` without --post first"
        )
    enc = lazy.get()
    qs, ds = task.strings()
    t0 = time.time()
    base = cache_base(arm, suite, task, dtype)
    q = enc.encode(qs, util.MAX_QUERY_TOKENS, resume_path=Path(base + ".q.part.npz"), label=f"{task.name} queries")
    d = enc.encode(ds, util.MAX_DOC_TOKENS, resume_path=Path(base + ".d.part.npz"), label=f"{task.name} docs")
    dt = time.time() - t0
    save_cached(arm, suite, task, dtype, q, d, {"encode_time_s": round(dt, 2), "device": device})
    return q, d, dtype, {"cached": False, "encode_time_s": round(dt, 2)}


# --- scoring ---


def score_task(task: suites.Task, q_vec: np.ndarray, d_vec: np.ndarray, pm: dict | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    """Mean metrics and per-query metrics of a task from its (unique-string) vectors."""
    qs, ds = task.strings()
    q_row = {s: i for i, s in enumerate(qs)}
    d_row = {s: i for i, s in enumerate(ds)}
    if pm is not None:
        q_vec, d_vec = postmap.apply(pm, q_vec), postmap.apply(pm, d_vec)
    names = metrics.METRICS_BY_KIND[task.kind]
    per: dict[str, list[float]] = {n: [] for n in names}
    qids: list[str] = []
    for g in task.groups:
        qm = q_vec[[q_row[task.format_query(t)] for t in g.qtexts]]
        dm = d_vec[[d_row[task.format_doc(t)] for t in g.dtexts]]
        scores = qm @ dm.T
        if g.ignore_identical:
            index = {d: i for i, d in enumerate(g.dids)}
            for qi, qid in enumerate(g.qids):
                if qid in index:
                    scores[qi, index[qid]] = -np.inf
        top = metrics.topk(scores, metrics.TOP_K[task.kind])
        for qi, qid in enumerate(g.qids):
            r = metrics.query_metrics(
                task.kind,
                top[qi],
                g.qrels[qi],
                g.doc_session,
                g.gold_sessions[qi] if g.gold_sessions else None,
            )
            for n in names:
                per[n].append(r[n])
            qids.append(qid)
    return metrics.mean_metrics(per), {"qids": qids, "metrics": per}


def reference_for(task: suites.Task) -> dict[str, float] | None:
    if task.kind != "mteb" or task.limited or task.meta.get("split") != "test" or not REFERENCE_FILE.exists():
        return None
    ref = json.loads(REFERENCE_FILE.read_text())["tasks"].get(task.meta["mteb_task"])
    if not ref:
        return None
    return {"ndcg@10": ref["ndcg_at_10"], "recall@100": ref["recall_at_100"], "mrr@10": ref["mrr_at_10"]}


# --- result files ---


def variant(limit: int | None, dev_lme: int) -> str:
    v = ""
    if limit is not None:
        v += f".limit{limit}"
    if dev_lme != util.DEFAULT_DEV_LME:
        v += f".lme{dev_lme}"
    return v


def result_path(label: str, suite: str, limit: int | None = None, dev_lme: int = util.DEFAULT_DEV_LME) -> Path:
    return util.results_dir() / label / f"{suite}{variant(limit, dev_lme)}.json"


def run_eval(
    arm: str,
    suite: str,
    device: str = "mps",
    limit: int | None = None,
    dev_lme: int = util.DEFAULT_DEV_LME,
    post: Path | None = None,
    label: str | None = None,
    tasks: list[suites.Task] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate an arm on a suite and write results/<label>/<suite>.json. With `post` no model runs: cached vectors only."""
    t_start = time.time()
    from .model import dtype_name

    pm = postmap.load(post) if post else None
    if pm is None:
        util.check_device(device, limit)
    if tasks is None:
        tasks = suites.load_suite(suite, dev_lme)
    if limit is not None:
        tasks = [suites.limited(t, limit) for t in tasks]
    label = label or (f"{arm}+post" if pm is not None else arm)
    dtype = dtype_name(device)
    lazy = None if pm is not None else LazyEncoder(arm, device)
    accept = [dtype, "fp32", "fp16"] if pm is not None else [dtype]
    out_tasks: dict[str, Any] = {}
    encode_time = 0.0
    used_dtypes: set[str] = set()
    for task in tasks:
        q, d, used, info = task_vectors(arm, suite, task, device, lazy, dtype, accept)
        used_dtypes.add(used)
        encode_time += info.get("encode_time_s", 0.0)
        mean, per_query = score_task(task, q, d, pm)
        out_tasks[task.name] = {
            "kind": task.kind,
            "n_queries": task.n_queries(),
            "n_docs": task.n_docs(),
            "n_docs_unique": len(task.strings()[1]),
            "limited": task.limited,
            "main_metric": metrics.MAIN_METRIC,
            "metrics": mean,
            "reference": reference_for(task),
            "vectors_cached": info["cached"],
            "meta": task.meta if task.kind != "lme" else {k: v for k, v in task.meta.items() if k != "question_ids"},
            "per_query": per_query,
        }
    result = {
        "label": label,
        "arm": arm,
        "suite": suite,
        "device": "cpu-cached" if pm is not None else device,
        "dtype": "+".join(sorted(used_dtypes)),
        "limit": limit,
        "dev_lme": dev_lme,
        "post": None
        if pm is None
        else {"file": str(post), "sha256": util.sha256_file(Path(post)), "describe": postmap.describe(pm)},
        "encode_time_s": round(encode_time, 2),
        "wall_time_s": round(time.time() - t_start, 2),
        "model": edits.model_hashes(arm),
        "env": util.env_info(),
        "created_at": util.now_iso(),
        "tasks": out_tasks,
    }
    if write:
        p = result_path(label, suite, limit, dev_lme)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result))
    return result


def format_table(result: dict[str, Any]) -> str:
    env = result["env"]
    lines = [
        f"{result['label']} | suite={result['suite']} device={result['device']}({result['dtype']}) limit={result['limit']} "
        f"dev_lme={result['dev_lme']} | wall={result['wall_time_s']:.1f}s (encode {result['encode_time_s']:.1f}s) | "
        f"macOS {env['sw_vers_build']} | {env['xcodebuild']}",
    ]
    if result["post"]:
        lines.append(f"post-map: {result['post']['describe']} ({result['post']['file']})")
    rows = []
    for name, t in result["tasks"].items():
        parts = []
        for m, v in t["metrics"].items():
            ref = (t.get("reference") or {}).get(m)
            parts.append(f"{m}={v:.4f}" + (f" (ref {ref:.4f})" if ref is not None else ""))
        rows.append((name, str(t["n_queries"]), str(t["n_docs_unique"]), "  ".join(parts) + ("  [limited]" if t["limited"] else "")))
    w = [max(len(r[i]) for r in rows + [("task", "queries", "docs", "")]) for i in range(3)]
    lines.append(f"{'task':<{w[0]}}  {'queries':>{w[1]}}  {'docs':>{w[2]}}  metrics")
    for r in rows:
        lines.append(f"{r[0]:<{w[0]}}  {r[1]:>{w[1]}}  {r[2]:>{w[2]}}  {r[3]}")
    return "\n".join(lines)


# --- compare ---


def compare(label_a: str, label_b: str, suite: str, limit: int | None = None, dev_lme: int = util.DEFAULT_DEV_LME, metric: str | None = None, n_boot: int = 10000) -> tuple[str, list[dict[str, Any]]]:
    pa, pb = result_path(label_a, suite, limit, dev_lme), result_path(label_b, suite, limit, dev_lme)
    for p in (pa, pb):
        if not p.exists():
            raise Refused(f"missing result file {p}: run the eval first")
    ra, rb = json.loads(pa.read_text()), json.loads(pb.read_text())
    rows = []
    for name, ta in ra["tasks"].items():
        tb = rb["tasks"].get(name)
        if tb is None:
            continue
        m = metric or ta["main_metric"]
        if m not in ta["per_query"]["metrics"]:
            continue
        ia = {q: i for i, q in enumerate(ta["per_query"]["qids"])}
        ib = {q: i for i, q in enumerate(tb["per_query"]["qids"])}
        common = [q for q in ia if q in ib]
        a = np.array([ta["per_query"]["metrics"][m][ia[q]] for q in common])
        b = np.array([tb["per_query"]["metrics"][m][ib[q]] for q in common])
        lo, hi = metrics.bootstrap_ci(b - a, n_boot=n_boot)
        rows.append({"task": name, "metric": m, "n": len(common), "a": float(a.mean()), "b": float(b.mean()), "diff": float((b - a).mean()), "ci_lo": lo, "ci_hi": hi, "sig": bool(lo > 0 or hi < 0)})
    w = max([len(r["task"]) for r in rows] + [4])
    lines = [f"{label_a} -> {label_b} | suite={suite}{variant(limit, dev_lme)} | paired bootstrap 95% CI of (B - A), {n_boot} resamples"]
    lines.append(f"{'task':<{w}}  {'metric':<14} {'n':>5}  {'A':>7}  {'B':>7}  {'B-A':>8}  95% CI")
    for r in rows:
        lines.append(f"{r['task']:<{w}}  {r['metric']:<14} {r['n']:>5}  {r['a']:7.4f}  {r['b']:7.4f}  {r['diff']:+8.4f}  [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{' *' if r['sig'] else ''}")
    return "\n".join(lines), rows


# --- post-map fitting ---


def fit_postmap(arm: str, kind: str, k: int, shrink: float, limit: int | None, dev_lme: int, out: Path | None) -> Path:
    """Fit on the cached dev-suite document vectors of the arm (no queries, no test data)."""
    tasks = suites.load_suite("dev", dev_lme)
    if limit is not None:
        tasks = [suites.limited(t, limit) for t in tasks]
    chunks = []
    for t in tasks:
        hit = find_cached(arm, "dev", t, ["fp32", "fp16"])
        if hit is None:
            raise Refused(f"no cached dev vectors for arm={arm} task={t.name}: run `wsp eval --arm {arm} --suite dev` first (same --limit/--dev-lme)")
        chunks.append(hit[1])
    x = np.vstack(chunks)
    pm = postmap.fit(kind, x, k=k, shrink=shrink)
    tag = f"k{k}" if kind == "pca-remove" else f"s{shrink:g}"
    path = out or util.state_dir() / "postmaps" / f"{arm}.{kind}.{tag}{variant(limit, dev_lme)}.npz"
    postmap.save(path, pm, {"arm": arm, "fit_suite": "dev", "fit_on": "document vectors", "created_at": util.now_iso()})
    return path


# --- suite statistics ---


def suite_stats(suite: str, dev_lme: int = util.DEFAULT_DEV_LME) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(util.original_dir()))
    tasks = suites.load_suite(suite, dev_lme)
    out: dict[str, Any] = {"suite": suite, "dev_lme": dev_lme, "tasks": {}}

    def n_tokens(texts: list[str], max_len: int) -> int:
        total = 0
        for i in range(0, len(texts), 4096):
            total += sum(len(x) for x in tok(texts[i : i + 4096], truncation=True, max_length=max_len)["input_ids"])
        return total

    for t in tasks:
        qs, ds = t.strings()
        tq, td = n_tokens(qs, util.MAX_QUERY_TOKENS), n_tokens(ds, util.MAX_DOC_TOKENS)
        out["tasks"][t.name] = {
            "kind": t.kind,
            "queries": t.n_queries(),
            "docs": t.n_docs(),
            "docs_unique": len(ds),
            "queries_unique": len(qs),
            "tokens_queries": tq,
            "tokens_docs": td,
            "tokens_total": tq + td,
        }
    out["tokens_total"] = sum(v["tokens_total"] for v in out["tasks"].values())
    out["queries_total"] = sum(v["queries"] for v in out["tasks"].values())
    out["docs_total"] = sum(v["docs"] for v in out["tasks"].values())
    return out


def format_stats(st: dict[str, Any]) -> str:
    lines = [f"suite={st['suite']} dev_lme={st['dev_lme']}"]
    lines.append(f"{'task':<14} {'queries':>8} {'docs':>8} {'uniq docs':>10} {'tok queries':>12} {'tok docs':>12} {'tok total':>12}")
    for n, v in st["tasks"].items():
        lines.append(f"{n:<14} {v['queries']:>8} {v['docs']:>8} {v['docs_unique']:>10} {v['tokens_queries']:>12,} {v['tokens_docs']:>12,} {v['tokens_total']:>12,}")
    lines.append(f"{'TOTAL':<14} {st['queries_total']:>8} {st['docs_total']:>8} {'':>10} {'':>12} {'':>12} {st['tokens_total']:>12,}")
    return "\n".join(lines)


def eprint(*a: object) -> None:
    print(*a, file=sys.stderr)
