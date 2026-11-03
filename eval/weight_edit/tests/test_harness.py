"""Harness tests on a tiny random Qwen3 (see conftest.py). CPU only, no real model, no MPS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsp import budget, cli, edits, postmap, run, suites, util
from wsp.util import Refused

EDIT_SCALE = """
ARGS = {}
def edit(tensors, read_acts):
    name = "layers.1.mlp.down_proj.weight"
    return {name: tensors[name] * float(ARGS.get("s", "0.5"))}
"""
EXAMPLE = Path(__file__).resolve().parent.parent / "edits" / "example_remove_direction.py"


def write_edit(tmp: Path, text: str = EDIT_SCALE, name: str = "edit.py") -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def toy_task(name: str = "toy") -> suites.Task:
    docs = ["cats purr softly", "dogs bark loudly", "the sun is a star", "water boils at 100 C", "birds can fly", "fish live in water"]
    queries = ["which animal purrs", "what is hot and bright", "which animals swim"]
    qrels = [{0: 1.0}, {2: 1.0}, {5: 1.0}]
    g = suites.Group([f"q{i}" for i in range(3)], queries, [f"d{i}" for i in range(6)], docs, qrels)
    return suites.Task(name, "mteb", "Given a question, retrieve the best document", None, [g])


# --- device gate ---


def test_mps_refused_without_the_slot(monkeypatch, env):
    monkeypatch.setenv("WSP_GPU_OWNER", "bench_owner")
    monkeypatch.setattr(util, "gpu_first_line", lambda: "2026-11-02 21:02  now: bench by agent-benchmark session (x)  |  .gpu-lock: y")
    with pytest.raises(Refused, match="mps"):
        util.check_device("mps", None)
    monkeypatch.setattr(util, "gpu_first_line", lambda: "2026-11-02 23:41  now: bench by bench_owner  |  .gpu-lock: bench_owner")
    util.check_device("mps", None)


def test_mps_allowed_with_night_deadline(monkeypatch, env):
    monkeypatch.setattr(util, "gpu_first_line", lambda: "now: free")
    monkeypatch.setenv("NIGHT_DEADLINE", "2026-11-03T11:00")
    util.check_device("mps", None)


def test_mps_name_must_match_exactly(monkeypatch, env):
    monkeypatch.setenv("WSP_GPU_OWNER", "bench_owner")
    monkeypatch.setattr(util, "gpu_first_line", lambda: "now: bench by bench_owner_two")
    with pytest.raises(Refused):
        util.check_device("mps", None)


def test_mps_refused_without_any_opt_in(monkeypatch, env):
    monkeypatch.setattr(util, "gpu_first_line", lambda: "now: bench by bench_owner")
    with pytest.raises(Refused, match="WSP_ALLOW_MPS"):
        util.check_device("mps", None)


def test_mps_allowed_with_plain_opt_in(monkeypatch, env):
    monkeypatch.setattr(util, "gpu_first_line", lambda: "")
    monkeypatch.setenv("WSP_ALLOW_MPS", "1")
    util.check_device("mps", None)


def test_cpu_needs_a_small_limit(env):
    with pytest.raises(Refused):
        util.check_device("cpu", None)
    with pytest.raises(Refused):
        util.check_device("cpu", util.CPU_MAX_LIMIT + 1)
    util.check_device("cpu", util.CPU_MAX_LIMIT)


# --- arms, edits, manifests, cache keys ---


def test_copy_refuses_existing_arm_and_skips_nothing_needed(env):
    d = edits.copy_arm("a")
    assert (d / "model.safetensors").exists() and (d / "tokenizer.json").exists() and (d / "edit_manifest.json").exists()
    with pytest.raises(Refused, match="already exists"):
        edits.copy_arm("a")
    with pytest.raises(ValueError):
        util.check_arm_name("base")
    with pytest.raises(ValueError):
        util.check_arm_name("bad+name")


def test_cache_key_changes_after_an_edit(env):
    assert edits.weights_key("base") == "base"
    edits.copy_arm("a")
    assert edits.weights_key("a") == "base"  # copied, not edited: same weights
    t = toy_task()
    before = run.cache_base("a", "dev", t, "fp32")
    edits.apply_edit("a", write_edit(env), ["s=0.5"])
    key1 = edits.weights_key("a")
    assert key1 != "base" and run.cache_base("a", "dev", t, "fp32") != before
    edits.copy_arm("b")
    edits.apply_edit("b", write_edit(env), ["s=0.25"])
    assert edits.weights_key("b") not in ("base", key1)
    edits.apply_edit("a", write_edit(env), ["s=0.5"], replace=True)  # same edit again: same key
    assert edits.weights_key("a") == key1
    assert run.cache_base("base", "dev", t, "fp32") != run.cache_base("a", "dev", t, "fp32")


def test_manifest_hashes_and_original_untouched(env):
    from safetensors import safe_open

    orig_file = util.original_dir() / "model.safetensors"
    orig_sha = util.sha256_file(orig_file)
    edits.copy_arm("a")
    m = edits.apply_edit("a", write_edit(env), ["s=0.5"])
    name = "layers.1.mlp.down_proj.weight"
    assert list(m["tensors"]) == [name] and m["edited"] and m["edit_args"] == {"s": "0.5"}
    info = m["tensors"][name]
    with safe_open(str(orig_file), framework="pt") as f0, safe_open(str(edits.weights_file("a")), framework="pt") as f1:
        t0, t1 = f0.get_tensor(name), f1.get_tensor(name)
        assert str(t0.dtype) == str(t1.dtype) == "torch.bfloat16"  # original dtype kept
        assert info["sha256_base"] == edits._tensor_sha(t0) and info["sha256_edited"] == edits._tensor_sha(t1)
        assert info["n_changed"] == int((t0 != t1).sum()) > 0 and info["shape"] == list(t0.shape)
        assert set(f0.keys()) == set(f1.keys())
        other = "layers.0.mlp.down_proj.weight"
        assert edits._tensor_sha(f0.get_tensor(other)) == edits._tensor_sha(f1.get_tensor(other))
    assert m["edited_safetensors_sha256"] == util.sha256_file(edits.weights_file("a")) != orig_sha
    assert m["edit_file_text"].strip().startswith("ARGS") and m["edit_file_sha256"] == util.sha256_bytes(m["edit_file_text"].encode())
    assert json.loads(edits.manifest_path("a").read_text())["tensors"][name]["sha256_edited"] == info["sha256_edited"]
    assert util.sha256_file(orig_file) == orig_sha  # the original is unchanged


def test_edit_rejections(env):
    edits.copy_arm("a")
    with pytest.raises(Refused, match="not in the model"):
        edits.apply_edit("a", write_edit(env, "def edit(t, r):\n    return {'layers.99.nope.weight': t['norm.weight']}\n"))
    with pytest.raises(Refused, match="shape"):
        edits.apply_edit("a", write_edit(env, "import numpy as np\ndef edit(t, r):\n    return {'norm.weight': np.zeros(3)}\n"))
    with pytest.raises(Refused, match="no entry"):
        edits.apply_edit("a", write_edit(env, "def edit(t, r):\n    return {'norm.weight': t['norm.weight']}\n"))
    with pytest.raises(Refused, match="non-finite"):
        edits.apply_edit("a", write_edit(env, "def edit(t, r):\n    return {'norm.weight': t['norm.weight'] * float('nan')}\n"))
    with pytest.raises(Refused, match="define edit"):
        edits.apply_edit("a", write_edit(env, "x = 1\n"))
    with pytest.raises(Refused, match="does not exist"):
        edits.apply_edit("nope", write_edit(env))
    assert not edits.is_edited("a")
    edits.apply_edit("a", write_edit(env))
    with pytest.raises(Refused, match="already has an edit"):
        edits.apply_edit("a", write_edit(env))


def test_an_edit_changes_the_outputs(env):
    from wsp.model import Encoder

    edits.copy_arm("a")
    edits.apply_edit("a", write_edit(env), ["s=0.1"])
    texts = ["cats purr softly", "the sun is a star and it is bright"]
    base = Encoder(util.original_dir(), "cpu").encode(texts, 64)
    edited = Encoder(edits.arm_dir("a"), "cpu").encode(texts, 64)
    assert base.shape == edited.shape == (2, 32)
    assert np.allclose(np.linalg.norm(base, axis=1), 1, atol=1e-5)
    assert np.abs(base - edited).max() > 1e-3


def test_left_padding_is_invariant(env):
    from wsp.model import Encoder

    enc = Encoder(util.original_dir(), "cpu")
    texts = ["short", "a much longer text with many more words in it, so the batch needs padding"]
    batched = enc.encode(texts, 64)
    single = np.vstack([enc.encode([t], 64) for t in texts])
    assert np.abs(batched - single).max() < 1e-3


def test_example_edit_end_to_end(env):
    from wsp.model import Encoder

    enc = Encoder(util.original_dir(), "cpu")
    texts = [f"sample text number {i} about topic {i % 3}" for i in range(12)]
    acts = enc.acts(texts, 64, [1])
    out = util.state_dir() / "acts" / "base"
    out.mkdir(parents=True)
    np.save(out / "dev.L1.npy", acts[1])
    edits.copy_arm("demo")
    m = edits.apply_edit("demo", EXAMPLE, ["layer=1", "a=0.9"])
    assert sorted(m["tensors"]) == sorted(
        f"layers.{b}.{p}" for b in (1, 2) for p in ("self_attn.o_proj.weight", "mlp.down_proj.weight")
    )
    from safetensors import safe_open

    spec = importlib_spec(EXAMPLE)
    u = spec.top_direction(acts[1], center=False)
    name = "layers.2.mlp.down_proj.weight"
    with safe_open(str(util.original_dir() / "model.safetensors"), framework="pt") as f0, safe_open(str(edits.weights_file("demo")), framework="pt") as f1:
        w0, w1 = f0.get_tensor(name).float().numpy(), f1.get_tensor(name).float().numpy()
    assert np.allclose(u @ w1, 0.1 * (u @ w0), atol=0.05 * np.abs(u @ w0).max() + 1e-3)  # component along u shrunk to (1 - a)
    with pytest.raises(Refused, match="no activations"):
        from wsp.acts import read_acts

        read_acts(2)


def importlib_spec(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ex", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- budget and pass ---


def test_budget_account_is_shared_between_arms(env, monkeypatch):
    edits.copy_arm("a")
    edits.copy_arm("b")
    monkeypatch.setenv("WSP_BUDGET_ACCOUNT", "shared")
    budget.set_budget("a", 3)
    assert budget.budget_file("b") == budget.budget_file("a")
    assert budget.read("b")["remaining"] == 3
    monkeypatch.delenv("WSP_BUDGET_ACCOUNT")
    assert budget.read("b") is None


def test_budget_refuses_at_zero(env):
    with pytest.raises(Refused, match="no budget"):
        budget.check("a")
    budget.set_budget("a", 2)
    assert budget.spend("a") == 1 and budget.spend("a") == 0
    with pytest.raises(Refused, match="used up"):
        budget.spend("a")
    assert budget.read("a")["remaining"] == 0


def test_pass_refuses_test_suite_and_keeps_the_budget(env, capsys):
    edits.copy_arm("a")
    budget.set_budget("a", 3)
    rc = cli.main(["pass", "eval", "--arm", "a", "--suite", "test", "--device", "cpu", "--limit", "4"])
    assert rc == 2 and "never runs the test suite" in capsys.readouterr().err
    assert budget.read("a")["remaining"] == 3


def test_pass_decrements_logs_and_refuses_at_zero(env, monkeypatch, capsys):
    edits.copy_arm("a")
    budget.set_budget("a", 1)
    fake = {"label": "a", "tasks": {"t": {"metrics": {"ndcg@10": 0.5}}}}
    monkeypatch.setattr(run, "run_eval", lambda *a, **k: fake)
    monkeypatch.setattr(run, "format_table", lambda r: "TABLE")
    assert cli.main(["pass", "eval", "--arm", "a", "--device", "cpu", "--limit", "4"]) == 0
    out = capsys.readouterr().out
    assert "TABLE" in out and "passes left for a: 0" in out and "wall time:" in out
    lines = [json.loads(x) for x in budget.log_file("a").read_text().splitlines()]
    assert lines[-1]["event"] == "pass" and lines[-1]["status"] == "ok" and lines[-1]["remaining_after"] == 0 and "wall_time_s" in lines[-1]
    assert cli.main(["pass", "eval", "--arm", "a", "--device", "cpu", "--limit", "4"]) == 2
    assert "used up" in capsys.readouterr().err


def test_pass_mps_refused_before_spending(env, monkeypatch, capsys):
    edits.copy_arm("a")
    budget.set_budget("a", 2)
    monkeypatch.setenv("WSP_GPU_OWNER", "bench_owner")
    monkeypatch.setattr(util, "gpu_first_line", lambda: "now: bench by someone_else")
    assert cli.main(["pass", "eval", "--arm", "a", "--device", "mps"]) == 2
    assert budget.read("a")["remaining"] == 2


def test_failed_pass_still_counts_and_is_logged(env, monkeypatch):
    edits.copy_arm("a")
    budget.set_budget("a", 2)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "run_eval", boom)
    with pytest.raises(RuntimeError):
        cli.main(["pass", "eval", "--arm", "a", "--device", "cpu", "--limit", "4"])
    assert budget.read("a")["remaining"] == 1
    assert "failed: RuntimeError: boom" in budget.log_file("a").read_text()


def test_agent_mode_blocks_test_suite_and_unbudgeted_runs(env, monkeypatch):
    monkeypatch.setenv("WSP_AGENT_MODE", "1")
    assert cli.main(["eval", "--arm", "base", "--suite", "test", "--device", "cpu", "--limit", "4"]) == 2
    assert cli.main(["eval", "--arm", "base", "--suite", "dev", "--device", "cpu", "--limit", "4"]) == 2
    assert cli.main(["acts", "--arm", "base", "--layers", "1", "--limit", "4", "--device", "cpu"]) == 2
    assert cli.main(["compare", "base", "a", "--suite", "test"]) == 2


# --- post-maps ---


def test_postmap_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 16)) @ np.diag(np.linspace(3, 0.3, 16)) + 2.0
    pm = postmap.fit("pca-remove", x, k=2)
    path = tmp_path / "pm.npz"
    postmap.save(path, pm, {"arm": "base"})
    pm2 = postmap.load(path)
    y1, y2 = postmap.apply(pm, x), postmap.apply(pm2, x)
    assert np.allclose(y1, y2) and np.allclose(np.linalg.norm(y1, axis=1), 1, atol=1e-5)
    centered = (x - pm["mu"]) @ pm["matrix"]
    top_var = np.linalg.eigvalsh(np.cov(centered.T))[::-1]
    assert top_var[0] < np.linalg.eigvalsh(np.cov((x - x.mean(0)).T))[::-1][2] * 1.01  # top-2 directions removed
    w = postmap.fit("whiten", x, shrink=0.0)
    cov = np.cov(((x - w["mu"]) @ w["matrix"]).T)
    assert np.allclose(cov, np.eye(16), atol=1e-3)
    with pytest.raises(ValueError):
        postmap.fit("pca-remove", x, k=0)
    with pytest.raises(ValueError):
        postmap.fit("whiten", x, shrink=2.0)


def test_eval_post_runs_from_cached_vectors_only(env, monkeypatch):
    monkeypatch.setattr(util, "check_device", lambda device, limit: None)
    task = toy_task()
    monkeypatch.setattr(suites, "load_suite", lambda name, dev_lme=100: [toy_task()])
    r0 = run.run_eval("base", "dev", device="cpu", limit=None)
    assert not r0["tasks"]["toy"]["vectors_cached"] and (util.results_dir() / "base" / "dev.json").exists()
    path = run.fit_postmap("base", "whiten", 1, 0.5, None, 100, None)

    class NoModel:
        def __init__(self, *a, **k):
            raise AssertionError("a model was loaded for --post")

    monkeypatch.setattr(run, "LazyEncoder", NoModel)
    r1 = run.run_eval("base", "dev", post=path)
    assert r1["label"] == "base+post" and r1["tasks"]["toy"]["vectors_cached"] and r1["post"]["describe"] == "whiten shrink=0.5"
    assert (util.results_dir() / "base+post" / "dev.json").exists()
    assert set(r1["tasks"]["toy"]["metrics"]) == {"ndcg@10", "recall@100", "mrr@10"}
    text, rows = run.compare("base", "base+post", "dev", n_boot=200)
    assert rows and rows[0]["task"] == "toy" and "95% CI" in text
    # a task without cached vectors is refused, not computed
    monkeypatch.setattr(suites, "load_suite", lambda name, dev_lme=100: [toy_task("other")])
    with pytest.raises(Refused, match="no cached vectors"):
        run.run_eval("base", "dev", post=path)


def test_second_eval_hits_the_cache_and_an_edit_misses_it(env, monkeypatch):
    monkeypatch.setattr(util, "check_device", lambda device, limit: None)
    monkeypatch.setattr(suites, "load_suite", lambda name, dev_lme=100: [toy_task()])
    assert not run.run_eval("base", "dev", device="cpu")["tasks"]["toy"]["vectors_cached"]
    assert run.run_eval("base", "dev", device="cpu")["tasks"]["toy"]["vectors_cached"]
    edits.copy_arm("a")
    edits.apply_edit("a", write_edit(env), ["s=0.1"])
    r = run.run_eval("a", "dev", device="cpu")
    assert not r["tasks"]["toy"]["vectors_cached"] and r["model"]["weights_key"] == edits.weights_key("a")


# --- original model checks ---


def test_verify_original_detects_changes(env, capsys):
    assert cli.main(["verify-original", "--record"]) == 0
    assert cli.main(["verify-original"]) == 0 and "OK" in capsys.readouterr().out
    cfg = util.original_dir() / "config.json"
    cfg.write_text(cfg.read_text() + " ")
    assert cli.main(["verify-original"]) == 1 and "MISMATCH" in capsys.readouterr().out


def test_never_writes_inside_the_original(env):
    with pytest.raises(Refused):
        edits._assert_outside_original(util.original_dir() / "x")


def test_nan_batch_is_redone_without_padding(env):
    """A padded batch that comes out non-finite is redone one text at a time when the fallback is on (GPU backends)."""
    from wsp.model import Encoder

    enc = Encoder(util.original_dir(), "cpu")
    texts = ["short", "a much longer text with many more words in it, so the batch needs padding"]
    clean = enc.encode(texts, 64)
    real_forward = enc.model.forward

    def nan_if_batched(*args, **kwargs):
        out = real_forward(*args, **kwargs)
        if kwargs["input_ids"].shape[0] > 1:
            out.last_hidden_state = out.last_hidden_state * float("nan")
        return out

    enc.model.forward = nan_if_batched
    with pytest.raises(RuntimeError, match="non-finite"):
        enc.encode(texts, 64)
    enc.single_fallback = True
    again = enc.encode(texts, 64)
    assert enc.n_fallback_batches >= 1 and np.abs(again - clean).max() < 1e-3
