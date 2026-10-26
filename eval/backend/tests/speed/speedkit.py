"""Helpers for the speed, stats and report tests (imported by name; a unique module name, so other test
packages' helpers never shadow it). Offline only: temporary directories, a mock server on an ephemeral
port, a fake tokenizer; never the real oMLX venv unless explicitly opted in."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modeleval import paths


@dataclass(frozen=True)
class FakeModel:
    """Stand-in with the fields of CONTRACTS' registry.Model (used when registry.py is not there yet)."""
    key: str = "qwen-ref"
    display: str = "Qwen3.8-27B oQ5 + MTP"
    served_id: str = "Qwen3.8-27B-oQ5-mtp"
    path: Path = Path("/nonexistent/model")
    quant: str = "oQ5"
    family: str = "qwen3.8_27b"
    quant_dir: str = "oq5"
    spec: str = "mtp"
    vision: bool = True
    reference: bool = True
    sampling: dict = field(default_factory=lambda: {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    proxy_sampling: str = "inject:qwen3.8-gen-config"
    source: dict = field(default_factory=lambda: {"hf_repo": "Qwen/Qwen3.8-27B", "revision": "abc1234567890",
                                                  "made_by": "oq_convert level 5, MTP kept"})
    model_settings: dict = field(default_factory=dict)
    server_log_expect: list = field(default_factory=list)
    prepare: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def make_model(**kw: Any) -> Any:
    try:
        from modeleval.registry import Model
        base = FakeModel(**kw)
        return Model(**{k: getattr(base, k) for k in Model.__dataclass_fields__})
    except Exception:  # noqa: BLE001 - registry not built yet, or a different constructor
        return FakeModel(**kw)


NANBEIGE = dict(key="nanbeige", display="Nanbeige4.3-33B oQ5", served_id="Nanbeige4.3-33B-oQ5", quant="oQ5",
                family="nanbeige4.3_33b",
                spec="none", vision=False, reference=False,
                sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 0},
                source={"hf_repo": "Nanbeige/Nanbeige4.3-33B", "revision": "c1093c501cc6", "made_by": "oq_convert level 5"})


# ------------------------------------------------------------------ synthetic agentbench lines
def metrics(**kw: Any) -> dict[str, Any]:
    m = {"ok": True, "http_status": 200, "error": None, "skipped": None, "ttft_s": 0.5, "e2e_s": 2.0,
         "prompt_tokens": 1000, "prompt_tokens_local": None, "completion_tokens": 100,
         "completion_tokens_source": "usage", "cached_tokens": 0, "cached_source": "usage", "finish_reason": "stop",
         "decode_s": 2.0, "decode_tps": 50.0, "first_kind": "content", "reasoning_chars": 0, "content_chars": 10}
    m.update(kw)
    return m


def req(suite: str, phase: str, t_send: float = 1000.0, **kw: Any) -> dict[str, Any]:
    m = kw.pop("m", {})
    line = {"type": "request", "suite": suite, "phase": phase, "metrics": metrics(**m),
            "record": {"t_send_wall": t_send}}
    line.update(kw)
    return line


def header(build: str = "26B5101f", xcode: str = "27A9275") -> dict[str, Any]:
    return {"type": "header", "suite": "x", "system": {"sw_vers": {"ProductVersion": "27.2", "BuildVersion": build},
                                                       "xcodebuild": {"default": f"Xcode 27.1\nBuild version {xcode}"}}}


# ------------------------------------------------------------------ a small fixture tree
def mini_fixtures(root: Path) -> Path:
    """The committed fixtures plus a tiny `generated/` tree: text equivalents and stub PNGs for every
    image, and short prompts under the names the speed units use."""
    from agentbench.fixtures.build import GENERATOR_VERSION, NONCE

    root.mkdir(parents=True, exist_ok=True)
    for f in paths.FIXTURES.glob("*"):
        if f.is_file():
            shutil.copy(f, root / f.name)
    man = json.loads((root / "manifest.json").read_text())
    assert man["generator_version"] == GENERATOR_VERSION
    for img_id, meta in man["images"].items():
        (root / meta["text"]).parent.mkdir(parents=True, exist_ok=True)
        (root / meta["text"]).write_text(f"Window {img_id}: a dialog with one checkbox and an OK button.\n")
        (root / meta["png"]).parent.mkdir(parents=True, exist_ok=True)
        (root / meta["png"]).write_bytes(b"\x89PNG\r\n\x1a\nstub")
    prompts = root / "generated" / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for kind in ("m3-off", "m3-on", "cold"):
        for size in (2000, 16000, 32000, 64000, 110000):
            msgs = [{"role": "system", "content": NONCE + "You write structured plans."},
                    {"role": "user", "content": "Write a JSON array of 150 plan steps for the document below.\n"
                                                f"Document ({kind} {size}): servers, queues, deploys."}]
            (prompts / f"{kind}-{size}.json").write_text(json.dumps(
                {"id": f"{kind}-{size}", "target_tokens": size, "tokens": 60, "enable_thinking": kind == "m3-on",
                 "nonce_placeholder": NONCE, "messages": msgs}))
    return root


# ------------------------------------------------------------------ synthetic unit lines
INCOMPLETE = "stream_error: {'message': 'tool call cut by max_tokens', 'code': 'incomplete_tool_call'}"


def m3_lines(decodes=(30.0, 32.0, 31.0), ctx: int = 16000, cap: int = 150, thinking: bool = False) -> list[dict]:
    cell = {"spec": "on", "thinking": thinking, "ctx": ctx, "cap": cap}
    lines = [req("m3", "warmup", cell={**cell, "phase": "warmup"})]
    for i, d in enumerate(decodes):
        lines.append(req("m3", "trial", trial=i, cell=cell, derived={"hit_cap": True},
                         m=dict(decode_tps=d, completion_tokens=cap, finish_reason="length",
                                cached_tokens=63000 if i == 0 else 63990, prompt_tokens=64012)))
    return lines


def m4_lines(n: int = 21, over: dict[int, dict] | None = None, ttft: float = 1.0, tps: float = 50.0) -> list[dict]:
    """Steps of 100 completion tokens decoded at `tps` (decode_s = 100 / tps), step 0 cold."""
    over = over or {}
    lines = []
    for k in range(n):
        m = dict(ttft_s=ttft, e2e_s=ttft + 100 / tps, completion_tokens=100, decode_s=100 / tps, decode_tps=99 * tps / 100,
                 cached_tokens=15000 if k else 0)
        m.update(over.get(k, {}))
        lines.append(req("m4", "start" if k == 0 else "step", step=k, run=0, m=m))
    return lines


def m1_lines(warm: float = 0.3, cold: float = 9.0) -> list[dict]:
    lines = [req("m1", "warmup", m=dict(cached_tokens=0))]
    lines += [req("m1", "warm", trial=i, m=dict(ttft_s=warm, cached_tokens=14900)) for i in range(5)]
    return lines + [req("m1", "cold", trial=0, m=dict(ttft_s=cold, cached_tokens=0))]


def g5_line(ttft: float = 60.0, t_send: float = 2000.0) -> dict:
    return req("gate", "", probe="G5.m3-off-64000", t_send=t_send,
               m=dict(ttft_s=ttft, e2e_s=ttft + 1, prompt_tokens=64012, cached_tokens=0))
