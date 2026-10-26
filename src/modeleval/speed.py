"""Speed leg (DESIGN §5.1-5.3): the functional block (G0, G1, G3, G4, vision probes, M7) and the timed
units, driven through agentbench's in-process API on the run's server launch.

Why not `agentbench run-all`: the old one-process-per-suite design lost settings between suites
(the text-only bug) and judged whole suites; here every timed unit is small, judged on its own
time window by `run.py`, and repeated alone. `gate.run()` is not used either: it always sends the
image probes and writes `capabilities.md`.

Fixed setting (DESIGN §2): the model's own agent sampling in every request (OpenAI and Anthropic),
thinking off (`chat_template_kwargs.enable_thinking: false` only, no `reasoning_effort`), text-only
prompts, the cold-nonce rule `system+tool`, fallback token counts with the model's own tokenizer.

Files: `<run_dir>/speed/<unit>-<attempt>.jsonl` (agentbench raw format), `gate-<n>.jsonl` and
`m7-<n>.jsonl` for the functional block, and `units.json`, which this module merges across attempts.
`run.py` owns waiting, judging and retries; it moves a raw file to `_unverified` when the unit
was disturbed or `UnitResult.valid` is false, and records the verdict with `record_verdict()`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from . import paths

FIXTURES_DIR = paths.FIXTURES  # tests point this at a small fixture tree
RENDER_PYTHON = paths.OMLX_PY  # the oMLX venv: the same transformers that serves the model
RENDER_TIMEOUT_S = 300
COLD_BLOCK = 256  # one oMLX cache block: a cold request must reuse less than this
M3_CAP, M3_TRIALS, THINK_CAP, WARMUP_MAX_TOKENS = 150, 3, 1000, 16
M1_TRIALS, M1_COLD = 5, 1
M4_STEPS, M4_CAP, M4_FIXED_TOKENS = 20, 150, 100
M7_TRIALS, M7_FAIL_AT = 5, 3  # a case fails when it misses in >= 3 of 5 trials

# unit -> (thinking, context tokens, cap). The 64K cell shares its unit with G5, whose prompt it reuses.
M3_CELLS: dict[str, tuple[bool, int, int]] = {
    "g5+m3-64k": (False, 64000, M3_CAP),
    "m3-16k": (False, 16000, M3_CAP),
    "m3-32k": (False, 32000, M3_CAP),
    "m3-110k": (False, 110000, M3_CAP),
    "m3-think-16k": (True, 16000, THINK_CAP),
}
UNIT_ORDER = ("g5+m3-64k", "m3-16k", "m3-32k", "m3-110k", "m3-think-16k", "m1", "m4-1", "m4-2", "m4-3")
TOOL_HISTORY_NAME = "computer_click"  # the canned assistant call of the M1 trial (G0 request 3)


# ======================================================================= data
@dataclass
class SpeedContext:
    model: Any  # registry.Model
    base_url: str  # "http://127.0.0.1:8703/v1"
    server_pid: int
    run_dir: Path  # results/<model>/<run-id>
    tokenizer_dir: Path  # the model's checkpoint (its own tokenizer and template)
    unverified_dir: Path


@dataclass
class UnitResult:
    unit: str
    attempt: int
    t_start: float
    t_end: float
    raw_file: Path
    metrics: dict
    valid: bool
    note: str


@dataclass
class FunctionalResult:
    ok: bool  # False: the run stops (a FAIL on G0, G1, G3.tool_use or G4, or missing fixtures)
    stop_reason: str | None
    items: dict[str, dict[str, Any]]  # G0..G4 (+G2, G2b with vision): result, detail
    m7: dict[str, Any] | None  # per-case gate
    files: list[Path] = field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 0.0
    # the first gate request that failed on the server's side ("transport: ...", "timeout: ...",
    # "http 5xx: ...", "stream_error: ..."), so run.py can tell a lost server from a judged FAIL
    server_error: str | None = None


def timed_units(model: Any) -> list[str]:
    """Quiet-window units in order. Without speculative decoding per-token work does not depend on
    thinking, so the thinking cell would repeat the 16K decode rate (DESIGN §5.1)."""
    return [u for u in UNIT_ORDER if u != "m3-think-16k" or model.spec != "none"]


def speed_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "speed"


def server_root(base_url: str) -> str:
    """CONTRACTS' base_url ends in /v1; agentbench appends /v1/... itself."""
    u = base_url.rstrip("/")
    return u[: -len("/v1")] if u.endswith("/v1") else u


def sampling_body(model: Any) -> dict[str, Any]:
    return {k: model.sampling[k] for k in ("temperature", "top_p", "top_k") if k in (model.sampling or {})}


def required_prompts(model: Any) -> list[str]:
    names = [f"m3-{'on' if th else 'off'}-{ctx}" for u, (th, ctx, _) in M3_CELLS.items() if u in timed_units(model)]
    return sorted(set(names))


# ======================================================================= agentbench plumbing
def _checked(cls: type, kw: dict[str, Any]) -> Any:
    """Construct a harness dataclass, failing loudly when the copied harness lacks a field the
    protocol needs (a silently dropped `vision=False` would send image probes to a text model)."""
    missing = sorted(set(kw) - {f.name for f in fields(cls)})
    if missing:
        raise RuntimeError(f"harness {cls.__name__} lacks {missing}; update harness/ (DESIGN §10)")
    return cls(**kw)


def run_config(ctx: SpeedContext, unit: str, attempt: int = 0) -> Any:
    from agentbench.profiles import get_profile
    from agentbench.suites.common import RunConfig

    m = ctx.model
    return _checked(RunConfig, dict(
        profile=get_profile("omlx"), base_url=server_root(ctx.base_url), model=m.served_id, framework="_agentbench",
        tag=m.key, out_root=speed_dir(ctx.run_dir), fixtures_dir=FIXTURES_DIR,
        text_only=True, text_only_reason="protocol: the core comparison is text-only (DESIGN §2)",
        server_pid=None,  # memory comes from core's sampler (memory.tsv); no second footprint loop
        extra_body=sampling_body(m), thinking_fields="kwargs",
        args={"unit": unit, "attempt": attempt, "model": m.key, "protocol_version": paths.PROTOCOL_VERSION},
        vision=bool(m.vision), cold_nonce="system+tool", anthropic_sampling=True,
        tokenizer_dir=Path(ctx.tokenizer_dir)))


def suite_opts(**kw: Any) -> Any:
    from agentbench.suites.opts import SuiteOpts

    base = {"think_effort": None, "warmup_max_tokens": WARMUP_MAX_TOKENS}
    return _checked(SuiteOpts, {**base, **kw})


_CAPTURE: type | None = None


def capture_writer(cfg: Any, name: str) -> Any:
    """An agentbench Writer that also keeps every line, so metrics are computed from exactly what
    was written."""
    global _CAPTURE
    if _CAPTURE is None:
        from agentbench.suites.common import Writer

        class Capture(Writer):
            def __init__(self, cfg: Any, suite: str) -> None:
                self.lines: list[dict[str, Any]] = []
                super().__init__(cfg, suite)

            def write(self, obj: dict[str, Any]) -> None:
                self.lines.append(obj)
                super().write(obj)

        _CAPTURE = Capture
    return _CAPTURE(cfg, name)


def _place(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)
    for d in (src.parent, src.parent.parent):  # _agentbench/raw, _agentbench
        try:
            d.rmdir()
        except OSError:
            break
    return dest


# ======================================================================= G0 prompt parity
RENDER_SCRIPT = r"""
import json, sys
import transformers
from transformers import AutoTokenizer
req = json.load(sys.stdin)
tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
kw = dict(req.get("chat_template_kwargs") or {})
try:
    ids = tok.apply_chat_template(req["messages"], tools=req.get("tools"), add_generation_prompt=True,
                                  tokenize=True, return_dict=False, **kw)
except TypeError:
    ids = tok.apply_chat_template(req["messages"], tools=req.get("tools"), add_generation_prompt=True,
                                  tokenize=True, **kw)
if hasattr(ids, "keys"):
    ids = ids["input_ids"]
if hasattr(ids, "tolist"):
    ids = ids.tolist()
json.dump({"ids": [int(i) for i in ids], "transformers": transformers.__version__}, sys.stdout)
"""


def independent_render(tokenizer_dir: Path, messages: list[dict], tools: list[dict] | None,
                       chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
    """`apply_chat_template(tokenize=True)` with the model's own tokenizer in the oMLX venv, as a
    subprocess (`-I -B`: no user site, no bytecode written into the venv). Returns {"ids": [...]}."""
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "HF_HOME")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1")
    req = {"messages": messages, "tools": tools, "chat_template_kwargs": chat_template_kwargs}
    p = subprocess.run([str(RENDER_PYTHON), "-I", "-B", "-c", RENDER_SCRIPT, str(tokenizer_dir)],
                       input=json.dumps(req), capture_output=True, text=True, timeout=RENDER_TIMEOUT_S, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"independent render failed (exit {p.returncode}): {p.stderr.strip()[-600:]}")
    out = json.loads(p.stdout)
    if not isinstance(out.get("ids"), list):
        raise RuntimeError("independent render returned no ids")
    return out


def first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def g0_requests(fx: Any, cfg: Any) -> dict[str, tuple[list[dict], list[dict] | None]]:
    """(1) no tools; (2) exactly the G1 request (44 tools), whose usage check 4 compares; (3) 44 tools
    plus a tool-call history: the text-only M1 trial (assistant tool call, tool result, screenshot text)."""
    from agentbench.suites import m1

    base = [fx.agent_system(), {"role": "user", "content": fx.gate["g1"]["prompt"]}]
    hist, _ = m1.trial_messages(fx, cfg, 0, fx.m1["m1_images"])
    return {"no_tools": (base, None), "tools": (base, fx.tools()), "history": (hist, fx.tools())}


def thinking_off_suffix(text: str) -> tuple[bool, str]:
    if re.search(r"<think>\s*</think>\s*$", text):
        return True, "prompt ends in an empty think block (thinking off)"
    if re.search(r"<think>\s*$", text):
        return False, "prompt ends in an open <think>: thinking is on"
    if "<think>" not in text and "</think>" not in text:
        return True, "template renders no think block"
    return False, f"no thinking-off generation suffix (tail {text[-80:]!r})"


def g0_properties(kind: str, ids: list[int], text: str, fx: Any,
                  hist: list[dict] | None = None) -> list[dict[str, Any]]:
    """The properties G0 exists to catch on the tools requests (DESIGN §5.1 (3))."""
    out = []
    doubled = len(ids) >= 2 and ids[0] == ids[1]
    out.append({"check": "single BOS", "ok": not doubled, "detail": f"ids[0:2]={ids[:2]}"})
    sp = fx.system_prompt
    if sp in text:
        out.append({"check": "system prompt verbatim", "ok": True, "detail": "verbatim"})
    else:
        ok = _ws(sp) in _ws(text)
        out.append({"check": "system prompt verbatim", "ok": ok,
                    "detail": "verbatim up to whitespace" if ok else "system prompt not found in the rendered prompt"})
    ok, why = thinking_off_suffix(text)
    out.append({"check": "thinking-off suffix", "ok": ok, "detail": why})
    if kind == "history" and hist:
        norm = _ws(text)
        task = _ws(hist[1]["content"])
        call = next(m for m in hist if m.get("tool_calls"))["tool_calls"][0]["function"]
        result = _ws(next(m["content"] for m in hist if m.get("role") == "tool"))
        a = norm.find(task)
        b = norm.find(call["name"], a + 1) if a >= 0 else -1
        args = json.loads(call["arguments"])
        args_ok = b >= 0 and all(str(v) in norm[b:] for v in args.values())
        c = norm.find(result, b + 1) if b >= 0 else -1
        out.append({"check": "tool-call turn rendered", "ok": args_ok,
                    "detail": f"{call['name']} {sorted(args)} after the task" if args_ok else "assistant tool call missing"})
        out.append({"check": "tool-response turn rendered", "ok": c >= 0,
                    "detail": "tool result after the call" if c >= 0 else "tool result missing"})
    return out


async def _post(target: Any, path: str, body: dict[str, Any]) -> Any:
    r = await target.client.http.post(path, json=body, timeout=300.0)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


async def g0_tokenize(ctx: SpeedContext, target: Any, fx: Any, cfg: Any, w: Any) -> dict[str, Any]:
    """G0 parts 1-3. The ids come from oMLX's own /v1/tokenize, which renders and encodes exactly as
    /v1/chat/completions does (add_special_tokens unset keeps the engine's encoding, so a doubled
    BOS shows). Part 4 (`g0_usage`) needs the real G1 request first."""
    kwargs = {"enable_thinking": False}
    reqs = g0_requests(fx, cfg)
    res: dict[str, Any] = {"requests": {}, "checks": []}
    ids_by: dict[str, list[int]] = {}
    for kind, (msgs, tools) in reqs.items():
        body: dict[str, Any] = {"model": ctx.model.served_id, "messages": msgs, "chat_template_kwargs": kwargs}
        if tools:
            body["tools"] = tools
        js = await _post(target, "/v1/tokenize", body)
        ids_by[kind] = [int(i) for i in js["tokens"]]
        res["requests"][kind] = {"count": len(ids_by[kind]), "n_tools": len(tools or []), "head": ids_by[kind][:8]}
    # (2) the no-tools request id for id against an independent render with the model's tokenizer
    msgs, _ = reqs["no_tools"]
    try:
        ind = await asyncio.to_thread(independent_render, Path(ctx.tokenizer_dir), msgs, None, kwargs)
        mine, theirs = ids_by["no_tools"], ind["ids"]
        i = first_difference(mine, theirs)
        detail = (f"{len(mine)} ids identical" if i is None else
                  f"first difference at index {i}: server {mine[max(0, i - 2):i + 3]} vs independent "
                  f"{theirs[max(0, i - 2):i + 3]} (lengths {len(mine)} / {len(theirs)})")
        res["checks"].append({"request": "no_tools", "check": "independent render", "ok": i is None,
                              "detail": detail, "first_difference": i, "transformers": ind.get("transformers")})
    except Exception as e:  # noqa: BLE001 - a broken renderer is a G0 failure, not a crash
        res["checks"].append({"request": "no_tools", "check": "independent render", "ok": False,
                              "detail": f"{type(e).__name__}: {e}"})
    # (3) properties of the tools requests, on the server's own ids decoded by the server
    for kind in ("tools", "history"):
        text = (await _post(target, "/v1/detokenize", {"model": ctx.model.served_id, "tokens": ids_by[kind]}))["prompt"]
        for c in g0_properties(kind, ids_by[kind], text, fx, reqs[kind][0]):
            res["checks"].append({"request": kind, **c})
    res["tools_count"] = len(ids_by["tools"])
    res["note"] = ("G1 also sends top-level reasoning_effort=none (Hermes shape), which oMLX forwards as a "
                   "template kwarg; both templates ignore it with thinking off")
    w.write({"type": "g0", **res})
    return res


def g0_usage(g0: dict[str, Any], g1_line: dict[str, Any] | None) -> dict[str, Any]:
    """G0 part 4: the real G1 request's usage.prompt_tokens equals the tokenize length of the same
    request, which shows that /v1/tokenize matches generation on this build."""
    pt = ((g1_line or {}).get("metrics") or {}).get("prompt_tokens")
    n = g0["tools_count"]
    ok = pt is not None and int(pt) == n
    detail = (f"usage.prompt_tokens {pt} == tokenize {n}" if ok else
              f"usage.prompt_tokens {pt} vs tokenize {n}" if pt is not None else "G1 returned no usage")
    return {"request": "tools", "check": "G1 usage", "ok": ok, "detail": detail}


def g0_item(g0: dict[str, Any]) -> dict[str, Any]:
    bad = [c for c in g0["checks"] if not c["ok"]]
    detail = "; ".join(f"{c['request']}: {c['check']}: {c['detail']}" for c in bad) or \
        "; ".join(f"{k} {v['count']} ids" for k, v in g0["requests"].items()) + "; all checks pass"
    return {"id": "G0", "result": "FAIL" if bad else "PASS", "detail": detail, "checks": g0["checks"]}


# ======================================================================= functional block
def _probe_line(lines: list[dict], prefix: str) -> dict[str, Any] | None:
    hits = [ln for ln in lines if ln.get("type") == "request" and str(ln.get("probe", "")).startswith(prefix)]
    return hits[-1] if hits else None


def _item(gate: Any, gid: str) -> dict[str, Any]:
    it = gate.items.get(gid) or {"result": "MISSING", "detail": "probe did not run", "sub": []}
    return {"id": gid, "result": it["result"], "detail": it["detail"], "sub": it.get("sub", [])}


def server_error(lines: list[dict]) -> str | None:
    """The first gate request that failed because of the server rather than the model's output: a
    transport error, a timeout, an HTTP 5xx, or a stream error other than a cap-cut tool call
    (`incomplete_tool_call` is the model's output running into its cap)."""
    from agentbench.metrics import CAP_TRUNCATED

    for ln in lines:
        if ln.get("type") != "request":
            continue
        err = (ln.get("record") or {}).get("error") or {}
        kind, where = err.get("type"), ln.get("probe") or ln.get("suite")
        if kind in ("transport", "timeout"):
            return f"{kind}: {where}: {err.get('detail') or err.get('phase') or ''}"[:200]
        if kind == "http" and int(err.get("status") or 0) >= 500:
            return f"http {err.get('status')}: {where}: {str(err.get('body') or '')[:120]}"
        if kind == "stream_error" and not any(c in json.dumps(err.get("detail")) for c in CAP_TRUNCATED):
            return f"stream_error: {where}: {json.dumps(err.get('detail'))[:120]}"
    return None


def m7_case_gate(lines: list[dict]) -> dict[str, Any]:
    """M7 as a per-case gate (DESIGN §5.1): a case fails when it misses in >= 3 of its 5 trials, and
    passes only after all 5 ran: a case cut short by an M7 error is `incomplete`, never a pass.
    Not a quality score: its misses are borderline cases that ran opposite to precision before."""
    cases: dict[str, dict[str, Any]] = {}
    na = []
    for ln in lines:
        if ln.get("suite") != "m7":
            continue
        if ln.get("type") == "skip":
            na.append(ln["cell"]["case"])
        elif ln.get("type") == "request" and ln.get("phase") == "trial":
            c = cases.setdefault(ln["cell"]["case"], {"category": ln["cell"].get("category"), "trials": 0, "misses": 0,
                                                       "reasons": []})
            c["trials"] += 1
            j = ln.get("judgement") or {}
            if not j.get("pass"):
                c["misses"] += 1
                c["reasons"] += [r for r in j.get("reasons", [])[:1] if r not in c["reasons"]]
    for c in cases.values():
        c["pass"] = c["misses"] < M7_FAIL_AT and c["trials"] >= M7_TRIALS
        c["incomplete"] = c["trials"] < M7_TRIALS and c["misses"] < M7_FAIL_AT
    failing = sorted(k for k, c in cases.items() if c["misses"] >= M7_FAIL_AT)
    incomplete = sorted(k for k, c in cases.items() if c["incomplete"])
    return {"scored_cases": len(cases) - len(incomplete), "passed_cases": sum(1 for c in cases.values() if c["pass"]),
            "failing": failing, "incomplete": incomplete,
            "na": sorted(na), "trials": sum(c["trials"] for c in cases.values()),
            "misses": sum(c["misses"] for c in cases.values()), "cases": dict(sorted(cases.items()))}


async def run_functional(ctx: SpeedContext) -> FunctionalResult:
    """Untimed and fail-fast, before the quiet window (DESIGN §5 step 4). A FAIL on G0, G1, G3.tool_use
    or G4 stops the run; G2/G2b and M7 are recorded only."""
    from agentbench.suites import m7
    from agentbench.suites.common import Target

    cfg = run_config(ctx, "functional")
    n = 1 + len(list(speed_dir(ctx.run_dir).glob("gate-*.jsonl")))  # a relaunch repeats the block
    items: dict[str, dict[str, Any]] = {}
    stop: str | None = None
    m7res: dict[str, Any] | None = None
    files: list[Path] = []
    t0 = time.time()
    fx, problem = _fixtures(ctx)
    if problem:  # found now rather than as an on-the-fly 110K prompt build inside the quiet window
        fr = FunctionalResult(False, problem, {}, None, [], t0, time.time())
        record_functional(ctx, fr)
        return fr
    target = Target(cfg, fx)
    gw = capture_writer(cfg, "gate")
    try:
        stop = await _gate_block(ctx, cfg, target, fx, gw, items)
    except Exception as e:  # noqa: BLE001 - an unreachable or crashing server stops the run, recorded
        stop = f"error: {type(e).__name__}: {e}"
        gw.write({"type": "error", "suite": "gate", "error": stop})
    finally:
        gw.close()
        files.append(_place(gw.path, speed_dir(ctx.run_dir) / f"gate-{n}.jsonl"))
    try:
        if stop is None:
            mw = capture_writer(cfg, "m7")
            try:
                await m7.run(cfg, target, mw, suite_opts(trials=M7_TRIALS))
            except Exception as e:  # noqa: BLE001 - M7 is recorded, never a stop rule
                mw.write({"type": "error", "suite": "m7", "error": f"{type(e).__name__}: {e}"})
            finally:
                mw.close()
                files.append(_place(mw.path, speed_dir(ctx.run_dir) / f"m7-{n}.jsonl"))
            m7res = m7_case_gate(mw.lines)
            err = next((ln["error"] for ln in mw.lines if ln.get("type") == "error"), None)
            if err:
                m7res["error"] = err
    finally:
        await target.aclose()
    fr = FunctionalResult(stop is None, stop, items, m7res, files, t0, time.time(),
                          server_error=None if stop is None else server_error(gw.lines))
    record_functional(ctx, fr, _system_from_header(gw.lines))
    return fr


def _fixtures(ctx: SpeedContext) -> tuple[Any, str | None]:
    """The fixture store, or why the speed leg cannot run with it. A prompt size that was not
    pre-built would otherwise be generated on the fly with the Qwen tokenizer during a timed unit."""
    from agentbench.fixtures.store import Fixtures, FixturesMissing

    try:
        fx = Fixtures(FIXTURES_DIR)
    except FixturesMissing as e:
        return None, f"fixtures: {e}"
    prompts = Path(fx.root) / "generated" / "prompts"
    missing = [p for p in required_prompts(ctx.model) if not (prompts / f"{p}.json").exists()]
    if missing:
        return fx, (f"fixtures: generated prompts missing ({', '.join(missing)}); "
                    "rebuild them with `agentbench fixtures build`")
    return fx, None


async def _gate_block(ctx: SpeedContext, cfg: Any, target: Any, fx: Any, gw: Any,
                      items: dict[str, dict[str, Any]]) -> str | None:
    """G0 -> G1 -> G0 usage check -> G3 -> G4 (-> G2, G2b with vision). Returns the stop reason."""
    from agentbench.suites.gate import Gate

    gate = Gate(cfg, target, gw)
    g0 = await g0_tokenize(ctx, target, fx, cfg, gw)
    items["G0"] = g0_item(g0)
    try:
        if items["G0"]["result"] == "FAIL":
            return "G0 failed"
        await gate.g1()
        items["G1"] = _item(gate, "G1")
        g0["checks"].append(g0_usage(g0, _probe_line(gw.lines, "G1.hermes")))
        items["G0"] = g0_item(g0)
        if items["G1"]["result"] == "FAIL":
            return "G1 failed"
        if items["G0"]["result"] == "FAIL":
            return "G0 failed (G1 usage)"
    finally:
        gw.write({"type": "gate_item", **items["G0"]})
    await gate.g3(vision=bool(ctx.model.vision))
    items["G3"] = _item(gate, "G3")
    tool_use = next((s for s in items["G3"]["sub"] if s.get("probe") == "G3.tool_use"), None)
    if not (tool_use and tool_use.get("ok")):
        return "G3.tool_use failed"
    await gate.g4()
    items["G4"] = _item(gate, "G4")
    if items["G4"]["result"] == "FAIL":
        return "G4 failed"
    if ctx.model.vision:  # recorded only: stock oMLX fails G2b by construction (DESIGN §6.8)
        await gate.g2()
        await gate.g2b()
        items["G2"], items["G2b"] = _item(gate, "G2"), _item(gate, "G2b")
    return None


# ======================================================================= timed units
async def _g5_m3_64k(ctx: SpeedContext, cfg: Any, target: Any, w: Any) -> None:
    """G5 sends the 64K M3 prompt cold (fresh nonce at the start of the system message); the 64K cell
    then reuses that prefix (same nonce) instead of a separate cold warm-up (DESIGN §5.1)."""
    from agentbench.fixtures.store import new_digits
    from agentbench.suites import m3
    from agentbench.suites.gate import Gate

    nonce = new_digits()
    await Gate(cfg, target, w).g5(prompt="m3-off", nonce=nonce)
    _, ctx_tokens, cap = M3_CELLS["g5+m3-64k"]
    await m3.run(cfg, target, w, suite_opts(spec_label=_spec_label(ctx.model), contexts=[ctx_tokens], caps=[cap],
                                            trials=M3_TRIALS, no_thinking_bucket=True, reuse_nonce=nonce))


def _spec_label(model: Any) -> str:
    return "off" if model.spec == "none" else "on"


async def _m3_cell(ctx: SpeedContext, unit: str, cfg: Any, target: Any, w: Any) -> None:
    from agentbench.suites import m3

    thinking, ctx_tokens, cap = M3_CELLS[unit]
    if thinking:  # enable_thinking only: each template renders its own default effort (DESIGN §2)
        opts = suite_opts(spec_label=_spec_label(ctx.model), contexts=[], caps=[M3_CAP], trials=M3_TRIALS,
                          think_contexts=[ctx_tokens], think_caps=[cap])
    else:
        opts = suite_opts(spec_label=_spec_label(ctx.model), contexts=[ctx_tokens], caps=[cap], trials=M3_TRIALS,
                          no_thinking_bucket=True)
    await m3.run(cfg, target, w, opts)


async def run_unit(ctx: SpeedContext, unit: str, attempt: int) -> UnitResult:
    """One timed unit on the running server. Never moves or judges: the raw file always lands at
    speed/<unit>-<attempt>.jsonl and units.json gets the attempt's metrics."""
    from agentbench.fixtures.store import Fixtures
    from agentbench.suites import m1, m4
    from agentbench.suites.common import Target

    if unit not in timed_units(ctx.model):
        raise ValueError(f"{unit!r} is not a timed unit for {ctx.model.key} ({', '.join(timed_units(ctx.model))})")
    cfg = run_config(ctx, unit, attempt)
    fx = Fixtures(cfg.fixtures_dir)
    target = Target(cfg, fx)
    w = capture_writer(cfg, unit)
    error = None
    t_start = time.time()
    try:
        if unit == "g5+m3-64k":
            await _g5_m3_64k(ctx, cfg, target, w)
        elif unit in M3_CELLS:
            await _m3_cell(ctx, unit, cfg, target, w)
        elif unit == "m1":
            await m1.run(cfg, target, w, suite_opts(trials=M1_TRIALS, cold_trials=M1_COLD))
        else:
            await m4.run(cfg, target, w, suite_opts(runs=1, steps=M4_STEPS, m4_max_tokens=M4_CAP))
    except Exception as e:  # noqa: BLE001 - recorded; run.py decides (server health, retry)
        error = f"{type(e).__name__}: {e}"
        w.write({"type": "error", "suite": unit, "error": error})
    finally:
        t_end = time.time()
        w.close()
        await target.aclose()
    metrics, valid, notes = unit_metrics(unit, w.lines)
    if error:
        valid, notes = False, [f"error: {error}"] + notes
        metrics.setdefault("invalid_kind", "incomplete")
    raw = _place(w.path, speed_dir(ctx.run_dir) / f"{unit}-{attempt}.jsonl")
    res = UnitResult(unit, attempt, t_start, t_end, raw, metrics, valid, "; ".join(notes))
    record_unit(ctx, res, _system_from_header(w.lines))
    return res


# ======================================================================= metrics (pure)
def _requests(lines: list[dict], suite: str | None = None) -> list[dict]:
    return [ln for ln in lines if ln.get("type") == "request" and (suite is None or ln.get("suite") == suite)]


def _latency_ok(m: dict) -> bool:
    """Latency samples: successful requests plus streams cut by the cap inside a tool call (DESIGN §5.1)."""
    from agentbench.metrics import latency_sample

    return latency_sample(m)


def _mmm(xs: list[Any]) -> dict[str, Any]:
    v = [x for x in xs if x is not None]
    if not v:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}


def _cold_ok(cached: int | None) -> bool:
    return cached is None or cached < COLD_BLOCK


def m3_cell_metrics(lines: list[dict], expect_trials: int = M3_TRIALS) -> tuple[dict[str, Any], list[str]]:
    """One M3 cell: median decode tok/s over its trials (the per-request (n-1)/span rate) and seconds
    per 100 output tokens. With `reuse_nonce` the warm-up is G5's cold prefill: `warmup` records
    whether trial 0 found it cached ("reused") or primed it itself ("fallback")."""
    reqs = _requests(lines, "m3")
    trials = [ln for ln in reqs if ln.get("phase") == "trial"]
    warmups = [ln for ln in reqs if ln.get("phase") == "warmup"]
    ok = [ln["metrics"] for ln in trials if ln["metrics"]["ok"]]
    dec = _mmm([m["decode_tps"] for m in ok])
    notes = []
    if len(ok) < expect_trials:
        notes.append(f"m3: {len(ok)}/{expect_trials} trials ok")
    hit = sum(1 for ln in trials if (ln.get("derived") or {}).get("hit_cap"))
    if trials and hit < len(trials):
        notes.append(f"m3: {len(trials) - hit} trial(s) ended before the cap")
    if any(ln.get("warmup") == "fallback" for ln in warmups):
        mode, first = "fallback", next(ln for ln in warmups if ln.get("warmup") == "fallback")
        notes.append("m3: G5's prefix was not cached; trial 0 became the warm-up")
    elif trials and trials[0].get("warmup") == "reused":
        mode, first = "reused", trials[0]
    else:
        mode, first = ("normal" if warmups else "none"), (trials[0] if trials else None)
    cell = trials[0]["cell"] if trials else {}
    return {"ctx": cell.get("ctx"), "cap": cell.get("cap"), "thinking": cell.get("thinking"),
            "decode_tps": dec, "s_per_100": (100.0 / dec["median"]) if dec["median"] else None,
            "ttft_s": _mmm([m["ttft_s"] for m in ok]), "completion_tokens": _mmm([m["completion_tokens"] for m in ok]),
            "prompt_tokens": _mmm([m["prompt_tokens"] for m in ok])["median"],
            "warmup": mode, "trial0_cached_tokens": first["metrics"]["cached_tokens"] if first else None,
            "reasoning_chars": _mmm([m.get("reasoning_chars") for m in ok])["median"],
            "hit_cap": hit, "trials": len(trials), "errors": len(trials) - len(ok)}, notes


def g5_metrics(lines: list[dict]) -> tuple[dict[str, Any], list[str]]:
    ln = _probe_line(lines, "G5")
    if ln is None:
        return {"ok": False}, ["g5: no request"]
    m, rec = ln["metrics"], ln.get("record") or {}
    t0 = rec.get("t_send_wall")
    out = {"ok": bool(m["ok"]), "ttft_s": m["ttft_s"], "e2e_s": m["e2e_s"], "prompt_tokens": m["prompt_tokens"],
           "cached_tokens": m["cached_tokens"], "error": m["error"],
           "t_start": t0, "t_end": (t0 + (m["e2e_s"] or 0)) if t0 else None}
    notes = [] if m["ok"] else [f"g5: {m['error']}"]
    if not _cold_ok(m["cached_tokens"]):
        notes.append(f"g5: cached_tokens {m['cached_tokens']} >= {COLD_BLOCK}: not cold")
    return out, notes


def _ttft(m: dict) -> tuple[float | None, str | None]:
    """A request's TTFT: for a response that opens with a tool call, the server's own (oMLX streams
    the call as one delta at its end), see agentbench.metrics.ttft."""
    from agentbench.metrics import ttft

    return ttft(m)


def m1_metrics(lines: list[dict]) -> tuple[dict[str, Any], list[str]]:
    reqs = _requests(lines, "m1")
    warm = [ln["metrics"] for ln in reqs if ln.get("phase") == "warm"]
    cold = [ln["metrics"] for ln in reqs if ln.get("phase") == "cold"]
    wl, cl = [m for m in warm if _latency_ok(m)], [m for m in cold if _latency_ok(m)]
    cold_cached = cold[0]["cached_tokens"] if cold else None
    sources = [_ttft(m)[1] for m in wl + cl]
    notes = []
    if len(wl) < M1_TRIALS or len(cl) < M1_COLD:
        notes.append(f"m1: {len(wl)}/{M1_TRIALS} warm and {len(cl)}/{M1_COLD} cold latency samples")
    if not _cold_ok(cold_cached):
        notes.append(f"m1: cold trial reused {cold_cached} cached tokens (>= {COLD_BLOCK})")
    elif cold and cold_cached is None:
        notes.append("m1: cold trial has no usage; cache check not possible")
    if "client-incl-decode" in sources:
        notes.append("m1: a trial opened with a tool call and the server reported no TTFT; its TTFT includes the decode")
    return {"warm_ttft_s": _mmm([_ttft(m)[0] for m in wl]), "cold_ttft_s": _mmm([_ttft(m)[0] for m in cl]),
            "warm_e2e_s": _mmm([m["e2e_s"] for m in wl]),
            "warm_cached_tokens": _mmm([m["cached_tokens"] for m in wl])["median"],
            "warm_prompt_tokens": _mmm([m["prompt_tokens"] for m in wl])["median"],
            "ttft_sources": {k: sources.count(k) for k in sorted({x for x in sources if x})},
            "ttft_includes_decode": sources.count("client-incl-decode"),
            "cold_cached_tokens": cold_cached, "errors": len(warm) + len(cold) - len(wl) - len(cl)}, notes


def m4_metrics(lines: list[dict], fixed_tokens: int = M4_FIXED_TOKENS) -> tuple[dict[str, Any], list[str]]:
    """M4 headline (DESIGN §5.1): sum of TTFT over the steps + steps x 100 / R, R = completion tokens /
    decode time over the steps with server usage. The raw total mixes byte-identical prefill with each
    model's own output length, so a terser model would win it regardless of speed. The arithmetic is
    the harness's own `m4.run_stats`, so the raw file's summary and units.json agree."""
    from agentbench.metrics import cap_hit
    from agentbench.suites.m4 import run_stats

    steps = sorted((ln for ln in _requests(lines, "m4") if ln.get("phase") in ("start", "step")),
                   key=lambda x: x["step"])
    run = [ln["metrics"] | {"step": ln["step"], "cap_hit": cap_hit(ln["metrics"]),
                            "latency_sample": _latency_ok(ln["metrics"])} for ln in steps]
    st = run_stats(run, fixed_tokens)
    notes = []
    if st["failed_steps"]:
        notes.append(f"m4: steps {st['failed_steps']} have no latency sample")
    step0 = st["start_cached_tokens"]
    if not _cold_ok(step0):
        notes.append(f"m4: step 0 reused {step0} cached tokens (>= {COLD_BLOCK})")
    elif run and step0 is None:
        notes.append("m4: step 0 has no usage; cache check not possible")
    if st.get("ttft_includes_decode"):
        notes.append(f"m4: steps {st['ttft_includes_decode']} opened with a tool call and the server reported no "
                     "TTFT; their TTFT includes the decode")
    if run and not st["failed_steps"] and st["headline_s"] is None:
        notes.append("m4: no step has a decode time, so R and the headline are empty")
    return {"steps": st["n_steps"], "headline_s": st["headline_s"], "ttft_sum_s": st["ttft_sum_s"],
            "decode_rate_tps": st["decode_rate_tps"], "rate_steps": st["usage_steps"],
            "raw_total_s": st["e2e_total_s"] if run else None, "mean_completion_tokens": st["mean_completion_tokens"],
            "cap_hits": st["cap_hits"], "cap_hits_by_kind": st["cap_hits_by_kind"], "step0_cached_tokens": step0,
            "direct_calls": len(st.get("direct_calls") or []),
            "ttft_includes_decode": len(st.get("ttft_includes_decode") or []),
            "fixed_tokens": fixed_tokens, "hard_errors": len(st["failed_steps"])}, notes


def unit_metrics(unit: str, lines: list[dict]) -> tuple[dict[str, Any], bool, list[str]]:
    """Metrics of one unit attempt and whether it is a valid measurement (all samples present, the
    cold requests really cold). `cold_cached_tokens` lists the cached tokens of the requests that must
    be cold (G5, the M1 cold trial, M4 step 0), for run.py's own check. An invalid attempt's
    `invalid_kind` says why: "cache-hit" (systematic, final) or "incomplete" (a missing sample or an
    error, which run.py repeats)."""
    metrics, valid, notes = _unit_metrics(unit, lines)
    if not valid:
        cold = metrics.get("cold_cached_tokens") or []
        metrics["invalid_kind"] = "cache-hit" if any(not _cold_ok(c) for c in cold) else "incomplete"
    return metrics, valid, notes


def _unit_metrics(unit: str, lines: list[dict]) -> tuple[dict[str, Any], bool, list[str]]:
    if unit == "g5+m3-64k":
        g5, n1 = g5_metrics(lines)
        cell, n2 = m3_cell_metrics(lines)
        valid = g5.get("ok", False) and _cold_ok(g5.get("cached_tokens")) and cell["errors"] == 0 \
            and cell["trials"] == M3_TRIALS
        return {"g5": g5, "m3": cell, "cold_cached_tokens": [g5.get("cached_tokens")]}, bool(valid), n1 + n2
    if unit in M3_CELLS:
        cell, notes = m3_cell_metrics(lines)
        return {"m3": cell}, cell["errors"] == 0 and cell["trials"] == M3_TRIALS, notes
    if unit == "m1":
        m, notes = m1_metrics(lines)
        valid = (m["warm_ttft_s"]["n"] == M1_TRIALS and m["cold_ttft_s"]["n"] == M1_COLD
                 and _cold_ok(m["cold_cached_tokens"]))
        return {"m1": m, "cold_cached_tokens": [m["cold_cached_tokens"]]}, valid, notes
    if unit.startswith("m4-"):
        m, notes = m4_metrics(lines)
        valid = m["steps"] == M4_STEPS + 1 and m["hard_errors"] == 0 and _cold_ok(m["step0_cached_tokens"])
        return {"m4": m, "cold_cached_tokens": [m["step0_cached_tokens"]]}, valid, notes
    raise ValueError(unit)


# ======================================================================= units.json
# One schema with run.py, which also writes this file (DESIGN §8): {"run_id", "model": <key>,
# "functional": {"passed", "time", "result": FunctionalResult}, "units": {unit: {"verdict", "attempts":
# [{"attempt", "t_start", "t_end", "raw_file", "valid", "note", "metrics", "verdict", "kept", ...}]}}}.
# This module records each attempt as soon as it ends (so a crash keeps its metrics); run.py adds the
# quiet verdict ("clean" | "disturbed" | "invalid") and "kept", and the unit-level verdict.
def units_path(run_dir: Path) -> Path:
    return speed_dir(run_dir) / "units.json"


def load_units(run_dir: Path) -> dict[str, Any]:
    p = units_path(run_dir)
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    doc.setdefault("units", {})
    doc.setdefault("functional", None)
    return doc


def save_units(run_dir: Path, doc: dict[str, Any]) -> None:
    p = units_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return paths.tilde(o)
    raise TypeError(type(o).__name__)


def unit_record(doc: dict[str, Any], unit: str) -> dict[str, Any]:
    """The unit's record in run.py's shape (a bare attempt list is converted)."""
    rec = doc.setdefault("units", {}).get(unit)
    if isinstance(rec, list):
        rec = {"verdict": None, "attempts": rec}
    if not isinstance(rec, dict):
        rec = {"verdict": None, "attempts": []}
    rec.setdefault("attempts", [])
    doc["units"][unit] = rec
    return rec


def functional_result(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """The latest functional block as a FunctionalResult dict with `ok`."""
    f = (doc or {}).get("functional")
    if isinstance(f, list):
        f = f[-1] if f else None
    if isinstance(f, dict) and isinstance(f.get("result"), dict):
        if f.get("server_lost"):   # run.py: the block failed because of the server, not the model
            return {**f["result"], "ok": None, "server_lost": f["server_lost"]}
        ok = f.get("passed") if isinstance(f.get("passed"), bool) else f["result"].get("ok")
        return {**f["result"], "ok": ok}
    return f if isinstance(f, dict) else None


def model_info(model: Any) -> dict[str, Any]:
    keys = ("key", "display", "served_id", "quant", "spec", "vision", "reference", "sampling", "proxy_sampling",
            "source", "notes")
    out = {k: getattr(model, k, None) for k in keys}
    out["path"] = paths.tilde(getattr(model, "path", ""))
    return out


def _system_from_header(lines: list[dict]) -> dict[str, Any]:
    """macOS and Xcode builds from the agentbench header: part of the speed comparability key."""
    h = next((ln for ln in lines if ln.get("type") == "header"), {})
    sysi = h.get("system") or {}
    sw = sysi.get("sw_vers") or {}
    xc = (sysi.get("xcodebuild") or {}).get("default") or ""
    m = re.search(r"Build version\s+(\S+)", xc)
    return {"macos": sw.get("ProductVersion"), "macos_build": sw.get("BuildVersion"),
            "xcode": xc.splitlines()[0] if xc else None, "xcode_build": m.group(1) if m else None,
            "harness": h.get("harness")}


def _rel(run_dir: Path, p: Path) -> str:
    """Run-relative, so committed files carry no absolute paths (_unverified is a sibling tree)."""
    return os.path.relpath(Path(p).resolve(), Path(run_dir).resolve())


def _stamp(doc: dict[str, Any], ctx: SpeedContext, system: dict[str, Any] | None) -> None:
    doc.setdefault("run_id", Path(ctx.run_dir).name)
    doc.setdefault("model", ctx.model.key)
    doc["protocol_version"] = paths.PROTOCOL_VERSION
    doc["model_info"] = model_info(ctx.model)
    if system and system.get("macos_build") and not doc.get("system"):
        doc["system"] = system


def _plain(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def record_functional(ctx: SpeedContext, fr: FunctionalResult, system: dict[str, Any] | None = None) -> None:
    doc = load_units(ctx.run_dir)
    _stamp(doc, ctx, system)
    result = _plain(fr)
    result["files"] = [_rel(ctx.run_dir, f) for f in fr.files]
    doc["functional"] = {"passed": fr.ok, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "result": result}
    save_units(ctx.run_dir, doc)


def record_unit(ctx: SpeedContext, res: UnitResult, system: dict[str, Any] | None = None) -> None:
    doc = load_units(ctx.run_dir)
    _stamp(doc, ctx, system)
    rec = unit_record(doc, res.unit)
    rec["attempts"] = [a for a in rec["attempts"] if a.get("attempt") != res.attempt]
    rec["attempts"].append({"attempt": res.attempt, "t_start": res.t_start, "t_end": res.t_end,
                            "raw_file": _rel(ctx.run_dir, res.raw_file), "valid": res.valid, "note": res.note,
                            "metrics": _plain(res.metrics), "verdict": None, "kept": False})
    rec["attempts"].sort(key=lambda a: a["attempt"])
    save_units(ctx.run_dir, doc)


def record_verdict(run_dir: Path, unit: str, attempt: int, verdict: Any, raw_file: Path | None = None,
                   max_attempts: int = 3) -> None:
    """Store a quiet verdict the way run.py does (for callers other than run.py, and tests): a
    `QuietGuard.UnitVerdict`, a dict with `clean`, or a verdict string; a moved raw file's new place."""
    doc = load_units(run_dir)
    rec = unit_record(doc, unit)
    a = next((x for x in rec["attempts"] if x.get("attempt") == attempt), None)
    if a is None:
        raise KeyError(f"no attempt {attempt} of {unit} in {units_path(run_dir)}")
    if isinstance(verdict, str):
        v, extra = verdict, {}
    else:
        d = _plain(verdict) if not isinstance(verdict, dict) else dict(verdict)
        v = "clean" if d.get("clean") and a.get("valid") else ("invalid" if not a.get("valid") else "disturbed")
        extra = {k: d[k] for k in ("reasons", "peak_foreign_cpu", "samples", "battery_min") if k in d}
    a.update(extra, verdict=v, kept=v == "clean")
    if raw_file is not None:
        a["raw_file"] = _rel(run_dir, raw_file)
    if a["kept"]:
        rec["verdict"] = "clean"
    elif len(rec["attempts"]) >= max_attempts and not any(x.get("kept") for x in rec["attempts"]):
        rec["verdict"] = f"disturbed x{len(rec['attempts'])}"
    save_units(run_dir, doc)


def _in_speed_dir(run_dir: Path, raw: Any) -> bool:
    if not raw:
        return False
    p = Path(str(raw)).expanduser()
    p = p if p.is_absolute() else Path(run_dir) / p
    return p.exists() and p.resolve().parent == speed_dir(run_dir).resolve()


def kept_attempt(rec: dict | list, run_dir: Path | None = None) -> dict | None:
    """The attempt a table uses (the last one that qualifies): `kept`, or valid and judged clean, or,
    without a recorded verdict, valid with its raw file still under speed/ (run.py moves every other
    one to _unverified)."""
    attempts = rec.get("attempts", []) if isinstance(rec, dict) else (rec or [])
    kept = None
    for a in attempts:
        v = a.get("verdict")
        if a.get("kept") is True:
            kept = a
        elif not a.get("valid") or a.get("kept") is False and v is not None:
            continue
        elif isinstance(v, str):
            kept = a if v == "clean" else kept
        elif isinstance(v, dict):
            kept = a if v.get("clean") else kept
        elif run_dir is not None and _in_speed_dir(run_dir, a.get("raw_file")):
            kept = a
    return kept


def finalize(run_dir: Path) -> dict[str, Any]:
    """After the speed leg (and after run.py's last write of units.json): memory peaks from memory.tsv
    into units.json. report.py computes them from memory.tsv itself when they are missing."""
    doc = load_units(run_dir)
    doc["memory"] = memory_summary(run_dir, doc)
    save_units(run_dir, doc)
    return doc["memory"]


# ======================================================================= memory.tsv
# core's MemorySampler (server.MEMORY_COLUMNS): iso, epoch, launch, label, source, tree_bytes, wired_delta,
# compressor_delta, file_backed_delta, pressure, swapouts_delta. The deltas are against the launch's
# pre-launch baseline, so swapouts_delta is cumulative per launch. Other spellings are accepted too.
MEM_TIME = ("epoch", "t", "t_wall", "time", "ts", "unix")
MEM_FOOTPRINT = ("tree_bytes", "footprint_bytes", "server_footprint_bytes", "server_bytes", "footprint",
                 "tree_mb", "footprint_mb", "server_mb", "footprint_mib", "tree_gib", "footprint_gib")
MEM_SYSTEM = ("sys_delta_bytes", "system_delta_bytes", "system_used_delta_bytes", "used_delta_bytes",
              "sys_delta_mb", "system_delta_mb", "used_delta_mb", "system_delta_gib")
MEM_SYSTEM_PARTS = ("wired_delta", "compressor_delta", "file_backed_delta")  # bytes; summed when no total column
MEM_PRESSURE = ("pressure", "pressure_level", "vm_pressure")
MEM_SWAP = ("swapouts_delta", "swapout_delta", "swapouts")
MEM_LABEL = ("label", "mark")
MEM_LAUNCH = ("launch",)
AFTER_LOAD = ("after-load", "after_load", "loaded")
_SCALE = (("_bytes", 1), ("_b", 1), ("_kb", 1024), ("_kib", 1024), ("_mb", 2**20), ("_mib", 2**20),
          ("_gb", 2**30), ("_gib", 2**30))


def _scale(col: str) -> int:
    return next((s for suf, s in _SCALE if col.endswith(suf)), 1)


def _num(x: str | None) -> float | None:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def read_memory_tsv(path: Path) -> list[dict[str, Any]]:
    """Samples as {t, launch, footprint, system, pressure, swap, label} (bytes; None when absent)."""
    if not Path(path).exists():
        return []
    rows = [ln for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not rows:
        return []
    head = [h.strip().lower() for h in rows[0].lstrip("#").strip().split("\t")]

    def col(cands: tuple[str, ...]) -> str | None:
        return next((c for c in cands if c in head), None)

    ct, cf, cs, cp, cw, cl, cn = (col(MEM_TIME), col(MEM_FOOTPRINT), col(MEM_SYSTEM), col(MEM_PRESSURE),
                                  col(MEM_SWAP), col(MEM_LABEL), col(MEM_LAUNCH))
    parts = [c for c in MEM_SYSTEM_PARTS if c in head]
    if ct is None:
        return []
    out = []
    for ln in rows[1:]:
        if ln.startswith("#") or ln.split("\t")[0].strip().lower() == head[0]:  # comments, repeated headers
            continue
        v = dict(zip(head, ln.split("\t")))
        t = _num(v.get(ct))
        if t is None:
            continue
        f = _num(v.get(cf)) if cf else None
        if cs:
            s = _num(v.get(cs))
            s = s * _scale(cs) if s is not None else None
        else:
            vals = [_num(v.get(c)) for c in parts]
            s = sum(x for x in vals if x is not None) if parts and any(x is not None for x in vals) else None
        out.append({"t": t, "launch": v.get(cn) if cn else None,
                    "footprint": f * _scale(cf) if f is not None else None, "system": s,
                    "pressure": _num(v.get(cp)) if cp else None, "swap": _num(v.get(cw)) if cw else None,
                    "label": (v.get(cl) or "").strip() if cl else ""})
    return out


def _peak(samples: list[dict], key: str, t0: float | None = None, t1: float | None = None) -> float | None:
    vals = [s[key] for s in samples
            if s[key] is not None and (t0 is None or s["t"] >= t0) and (t1 is None or s["t"] <= t1)]
    return max(vals) if vals else None


def memory_summary(run_dir: Path, doc: dict[str, Any] | None = None) -> dict[str, Any]:
    """After load, peak during G5, peak during the 110K cell, peak overall (server footprint and the
    system-wide delta), and whether the system reached pressure or swapped (DESIGN §5.2). Windows come
    from the kept attempts' own timestamps, so G5 is told apart from the 64K cell it shares a unit with."""
    doc = doc if doc is not None else load_units(run_dir)
    samples = read_memory_tsv(speed_dir(run_dir) / "memory.tsv")
    if not samples:
        return {}
    func = functional_result(doc) or {}
    labelled = [s for s in samples if s["label"].lower() in AFTER_LOAD]
    if labelled:
        after = labelled[0]
    else:  # the server's own warm-up loads the model before the functional block starts
        before = [s for s in samples if func.get("t_start") and s["t"] <= func["t_start"]]
        after = before[-1] if before else samples[0]
    swap_by_launch: dict[Any, float] = {}
    for s in samples:  # cumulative per launch: the last (largest) value of each launch
        if s["swap"] is not None:
            swap_by_launch[s["launch"]] = max(swap_by_launch.get(s["launch"], 0.0), s["swap"])
    out: dict[str, Any] = {"after_load": after["footprint"], "after_load_system": after["system"],
                           "peak": _peak(samples, "footprint"), "peak_system": _peak(samples, "system"),
                           "max_pressure": _peak(samples, "pressure"),
                           "swapouts": sum(swap_by_launch.values()) if swap_by_launch else None,
                           "samples": len(samples)}
    units = doc.get("units", {})
    g5a = kept_attempt(units.get("g5+m3-64k", []), run_dir)
    if g5a:
        g5 = (g5a.get("metrics") or {}).get("g5", {})
        if g5.get("t_start"):
            end = (g5.get("t_end") or g5["t_start"]) + 2.5  # one sampler interval past the request
            out["g5_peak"] = _peak(samples, "footprint", g5["t_start"], end)
            out["g5_peak_system"] = _peak(samples, "system", g5["t_start"], end)
    a110 = kept_attempt(units.get("m3-110k", []), run_dir)
    if a110:
        out["m3_110k_peak"] = _peak(samples, "footprint", a110["t_start"], a110["t_end"] + 2.5)
        out["m3_110k_peak_system"] = _peak(samples, "system", a110["t_start"], a110["t_end"] + 2.5)
    return out
