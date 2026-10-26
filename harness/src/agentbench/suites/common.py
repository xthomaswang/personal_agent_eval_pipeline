"""Shared suite plumbing: run configuration, JSONL writer, request target, metrics."""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..client import RequestRecord, ServerConfig, StreamClient
from ..fixtures.store import Fixtures
from ..metrics import decode_metrics
from ..profiles import Profile
from ..provenance import header, redact, repo_root

RESERVED_PORTS = {8500, *range(8600, 8700)}  # the agent benchmark's ports; never used here


def check_port(base_url: str) -> None:
    m = re.match(r"https?://[^/:]+:(\d+)", base_url)
    if m and int(m.group(1)) in RESERVED_PORTS:
        raise SystemExit(f"refusing to talk to {base_url}: 8500 and 8600-8699 belong to the agent benchmark")


@dataclass
class RunConfig:
    profile: Profile
    base_url: str
    model: str = ""
    framework: str = ""
    framework_version: str = ""
    launch_flags: str = ""
    tag: str = "default"
    out_root: Path = field(default_factory=repo_root)
    fixtures_dir: Path | None = None
    text_only: bool = False
    text_only_reason: str = ""
    quick: bool = False
    server_pid: int | None = None
    server_log: Path | None = None
    api_key: str = ""
    extra_body: dict[str, Any] = field(default_factory=dict)
    thinking_fields: str = "hermes"  # hermes | kwargs | effort | none
    read_timeout: float = 1800.0
    total_timeout: float = 3600.0
    screenshot_placement: str = "user"  # user | tool
    context_config: int | None = None
    args: dict[str, Any] = field(default_factory=dict)
    preflight_cmd: list[str] | None = None  # e.g. [<repo>/scripts/quiet_check.sh]; run before each suite
    unverified: bool = False  # write raw files to <framework>/raw/_unverified/ (gitignored, never published)
    preflight: dict[str, Any] | None = None  # result of the last preflight, recorded in the header
    interference: dict[str, Any] | None = None  # M8 block record (scenario, parameters, allowances), in the header

    @property
    def fw_dir(self) -> Path:
        return self.out_root / (self.framework or self.profile.framework)


# ---------------------------------------------------------------- writer
def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


class Writer:
    """`<framework>/raw/<suite>-<tag>-<timestamp>.jsonl`, provenance header first."""

    def __init__(self, cfg: RunConfig, suite: str) -> None:
        raw = cfg.fw_dir / "raw" / ("_unverified" if cfg.unverified else "")
        raw.mkdir(parents=True, exist_ok=True)
        stem = f"{suite}-{cfg.tag}-{timestamp()}"
        path, i = raw / f"{stem}.jsonl", 1
        while path.exists():
            i += 1
            path = raw / f"{stem}-{i}.jsonl"
        self.path = path
        self.f = path.open("w", encoding="utf-8")
        safe_args = {k: v for k, v in cfg.args.items() if k not in ("api_key",)}
        fw = {"name": cfg.framework or cfg.profile.framework, "profile": cfg.profile.name,
              "version": cfg.framework_version, "launch_flags": cfg.launch_flags}
        server = {"base_url": cfg.base_url, "model": cfg.model, "server_pid_given": cfg.server_pid is not None,
                  "context_config": cfg.context_config}
        hdr = header(suite, cfg.tag, fw, server, safe_args)
        hdr["preflight"] = cfg.preflight
        hdr["unverified"] = cfg.unverified
        if cfg.interference is not None:
            hdr["interference"] = cfg.interference
        hdr["mode"] = {"quick": cfg.quick, "text_only": cfg.text_only, "text_only_reason": cfg.text_only_reason,
                       "thinking_fields": cfg.thinking_fields, "screenshot_placement": cfg.screenshot_placement,
                       "tool_call_mode": cfg.profile.tool_call_mode,
                       **({"structured_format": cfg.profile.structured_format, "structured_union": cfg.profile.structured_union}
                          if cfg.profile.tool_call_mode == "structured_output" else {})}
        self.write(hdr)

    def write(self, obj: dict[str, Any]) -> None:
        self.f.write(redact(json.dumps(obj, ensure_ascii=False, default=str)) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------- helpers
_ZERO_PATTERNS = [(re.compile(r"\[run \d{12}\]"), "[run 000000000000]"),
                  (re.compile(r"request_id: \d{12}"), "request_id: 000000000000")]


def prompt_hash(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """sha256 of the prompt content with nonce / request-id digits zeroed, so the same
    canned prompt hashes identically across runs and frameworks."""
    s = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, sort_keys=True)
    for pat, rep in _ZERO_PATTERNS:
        s = pat.sub(rep, s)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cached_with_source(rec: RequestRecord) -> tuple[int | None, str | None]:
    u = rec.usage_raw or {}
    for path in (("prompt_tokens_details", "cached_tokens"), ("input_tokens_details", "cached_tokens"),
                 ("cached_tokens",), ("cache_read_input_tokens",), ("prompt_cache_hit_tokens",)):
        v: Any = u
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            return int(v), "usage." + ".".join(path)
    t = rec.timings_raw or {}
    if isinstance(t.get("cache_n"), (int, float)):
        return int(t["cache_n"]), "timings.cache_n"
    ex = u.get("extra") if isinstance(u.get("extra"), dict) else {}
    if isinstance(ex.get("prompt_tokens"), (int, float)) and isinstance(ex.get("prefill_tokens"), (int, float)):
        return int(ex["prompt_tokens"] - ex["prefill_tokens"]), "usage.extra (prompt_tokens - prefill_tokens)"
    return None, None


def short_error(err: dict[str, Any] | None) -> str | None:
    if not err:
        return None
    if err.get("type") == "http":
        return f"HTTP {err.get('status')}: {str(err.get('body', ''))[:300]}"
    return f"{err.get('type')}: {str(err.get('detail') or err.get('phase') or '')[:300]}"


class LogTail:
    """Reads values (e.g. lithos `'cached_prompt_tokens': N`) appended to a server log
    after a request."""

    def __init__(self, path: Path, regex: str) -> None:
        self.path, self.re = Path(path), re.compile(regex)
        self.pos = self._size()

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def mark(self) -> None:
        self.pos = self._size()

    async def read_after(self, timeout: float = 1.5) -> int | None:
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while True:
            try:
                with self.path.open("rb") as f:
                    f.seek(self.pos)
                    text = f.read().decode("utf-8", "replace")
            except OSError:
                text = ""
            found = self.re.findall(text)
            if found:
                return int(found[-1])
            if loop.time() >= end:
                return None
            await asyncio.sleep(0.1)


@dataclass
class Outcome:
    rec: RequestRecord
    shaping: list[str]
    prompt: dict[str, Any]
    retries_429: int = 0
    cached_log: int | None = None
    skipped: str | None = None

    def metrics(self, count_tokens=None) -> dict[str, Any]:
        r = self.rec
        dm = decode_metrics(r, count_tokens)
        cached, src = cached_with_source(r)
        if cached is None and self.cached_log is not None:
            cached, src = self.cached_log, "server-log"
        return {
            "ok": r.ok and not self.skipped, "http_status": r.http_status, "error": short_error(r.error),
            "skipped": self.skipped,
            "ttft_s": r.ttft, "e2e_s": r.e2e, "t_headers_s": r.t_headers, "t_first_data_s": r.t_first_data,
            "first_kind": r.chunks[0].kind if r.chunks else None,
            "prompt_tokens": r.prompt_tokens, "prompt_tokens_local": self.prompt.get("local_tokens"),
            "completion_tokens": dm["completion_tokens"], "completion_tokens_source": dm["tokens_source"],
            "cached_tokens": cached, "cached_source": src, "finish_reason": r.finish_reason,
            "decode_tps": dm["decode_tps"], "gap_p50_ms": dm["gap_p50_ms"], "gap_p99_ms": dm["gap_p99_ms"],
            "tokens_per_chunk": dm["tokens_per_chunk"], "n_token_chunks": dm["n_token_chunks"],
            "n_tool_calls": len(r.tool_calls), "markup_leak": r.markup_leak(),
            "reasoning_chars": len(r.reasoning), "content_chars": len(r.content),
            "reasoning_fields": r.reasoning_fields, "retries_429": self.retries_429,
            "tool_call_mode": r.tool_call_mode, "structured_error": r.structured_error,
        }

    def line(self, suite: str, count_tokens=None, **fields: Any) -> dict[str, Any]:
        return {"type": "request", "suite": suite, **fields, "tool_call_mode": self.rec.tool_call_mode,
                "shaping": self.shaping, "prompt": self.prompt, "metrics": self.metrics(count_tokens),
                "record": self.rec.to_json()}


def skipped_outcome(reason: str, api: str = "openai") -> Outcome:
    return Outcome(RequestRecord(api=api, url=""), [], {}, skipped=reason)


class Target:
    """One server under test: builds request bodies (Hermes-style thinking fields by
    default), applies the profile's shaping as the last pass, and records outcomes."""

    def __init__(self, cfg: RunConfig, fixtures: Fixtures | None = None, max_connections: int = 8) -> None:
        check_port(cfg.base_url)
        self.cfg, self.profile = cfg, cfg.profile
        self.fx = fixtures
        self.client = StreamClient(ServerConfig(base_url=cfg.base_url, model=cfg.model, extra_body=cfg.extra_body,
                                                api_key=cfg.api_key, read_timeout_s=cfg.read_timeout,
                                                total_timeout_s=cfg.total_timeout), max_connections=max_connections)
        self.tail = (LogTail(cfg.server_log, self.profile.log_cached_regex)
                     if cfg.server_log and self.profile.log_cached_regex else None)
        self._prompt = None

    async def aclose(self) -> None:
        await self.client.aclose()

    @property
    def qwen(self):
        if self._prompt is None:
            from ..tokenizer import default_prompt
            self._prompt = default_prompt()
        return self._prompt

    def count_tokens(self, text: str) -> int:
        return self.qwen.count_text(text)

    # ------------------------------------------------------------ thinking
    def thinking_fields(self, thinking: bool | None, effort: str | None = None,
                        effort_in_kwargs: bool = False) -> dict[str, Any]:
        """Hermes-style by default: top-level reasoning_effort ("none" when off) plus
        chat_template_kwargs.enable_thinking. thinking=None sends no thinking fields."""
        if thinking is None or self.cfg.thinking_fields == "none":
            return {}
        eff = (effort or "medium") if thinking else "none"
        kwargs: dict[str, Any] = {"enable_thinking": bool(thinking)}
        if thinking and effort_in_kwargs:
            kwargs["reasoning_effort"] = eff
        style = self.cfg.thinking_fields
        out: dict[str, Any] = {}
        if style in ("hermes", "effort"):
            out["reasoning_effort"] = eff
        if style in ("hermes", "kwargs"):
            out["chat_template_kwargs"] = kwargs
        return out

    # ------------------------------------------------------------- prompts
    def prompt_info(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                    thinking: bool | None, count_local: bool) -> dict[str, Any]:
        n_img = sum(1 for m in messages if isinstance(m.get("content"), list)
                    for p in m["content"] if p.get("type") in ("image_url", "image"))
        info: dict[str, Any] = {"sha256": prompt_hash(messages, tools), "n_messages": len(messages), "n_images": n_img,
                                "n_tools": len(tools or [])}
        if count_local:
            try:
                info["local_tokens"] = self.qwen.count(messages, tools, enable_thinking=thinking is not False)
            except Exception as e:  # never let local accounting break a measurement
                info["local_tokens_error"] = f"{type(e).__name__}: {e}"
        return info

    # ------------------------------------------------------------- requests
    async def chat(self, messages: list[dict[str, Any]], *, max_tokens: int, tools: list[dict[str, Any]] | None = None,
                   thinking: bool | None = False, effort: str | None = None, effort_in_kwargs: bool = False,
                   extra: dict[str, Any] | None = None, raw_fields: dict[str, Any] | None = None,
                   retry_429: bool = True, count_local: bool = True, structured_format: str | None = None) -> Outcome:
        """`raw_fields` replaces the generated thinking fields (gate probes use exact shapes)."""
        if raw_fields is None:
            blocked = self.profile.thinking_blocked(thinking, effort)
            if blocked:
                return skipped_outcome(blocked)
            fields = self.thinking_fields(thinking, effort, effort_in_kwargs)
        else:
            fields = raw_fields
        body = self.client.openai_body(messages, max_tokens, tools, extra={**fields, **(extra or {})})
        profile = self.profile
        if structured_format and profile.tool_call_mode == "structured_output":
            from dataclasses import replace
            profile = replace(profile, structured_format=structured_format)
        body, notes = profile.shape_openai(body)
        # describe what the server actually receives (structured mode rewrites messages and tools)
        info = self.prompt_info(body["messages"], body.get("tools"), thinking, count_local)
        structured = profile.tool_call_mode == "structured_output" and bool(tools)
        return await self._send(lambda: self.client.chat(body), notes, info, retry_429, structured)

    async def anthropic(self, body: dict[str, Any], retry_429: bool = True) -> Outcome:
        body, notes = self.profile.shape_anthropic(dict(body, model=body.get("model", self.cfg.model)))
        info = {"sha256": prompt_hash(body.get("messages", []), body.get("tools")), "n_messages": len(body.get("messages", []))}
        return await self._send(lambda: self.client.messages(body, add_model=False), notes, info, retry_429)

    async def _send(self, fn, notes: list[str], info: dict[str, Any], retry_429: bool,
                    structured: bool = False) -> Outcome:
        retries = 0
        while True:
            if self.tail:
                self.tail.mark()
            rec = await fn()
            if rec.http_status == 429 and retry_429 and retries < 60:
                retries += 1
                await asyncio.sleep(1.0)
                continue
            break
        rec.tool_call_mode = self.profile.tool_call_mode
        if structured and rec.ok:
            from ..structured import apply_to_record
            apply_to_record(rec)
        cached_log = await self.tail.read_after() if self.tail and rec.ok else None
        return Outcome(rec, notes, info, retries, cached_log)

    async def get_json(self, path: str) -> tuple[int | None, Any]:
        try:
            r = await self.client.http.get(path, timeout=10.0)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, r.text[:500]
        except Exception as e:  # noqa: BLE001 - probing only
            return None, f"{type(e).__name__}: {e}"


def run_preflight(cfg: RunConfig) -> dict[str, Any] | None:
    """METHODOLOGY 'quiet machine' rule: run the quiet check (read-only) before a timed segment.
    A failing check routes the run to raw/_unverified/ instead of aborting it."""
    if not cfg.preflight_cmd:
        return None
    import os
    import subprocess

    cmd = list(cfg.preflight_cmd) + ["--allow-pid", str(os.getpid())]
    if cfg.server_pid:
        cmd += ["--allow-pid", str(cfg.server_pid)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        res = {"command": " ".join(Path(c).name if i == 0 else c for i, c in enumerate(cmd)), "exit_code": out.returncode,
               "output": redact(out.stdout.strip())[-4000:], "stderr": redact(out.stderr.strip())[-1000:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        res = {"command": Path(cmd[0]).name, "exit_code": None, "output": "", "stderr": f"{type(e).__name__}: {e}"}
    cfg.preflight = res
    if res["exit_code"] != 0 and not cfg.unverified:
        cfg.unverified = True
        eprint(f"preflight: machine is NOT quiet (exit {res['exit_code']}); results go to raw/_unverified/ "
               "and must be repeated before publishing")
    return res


# ------------------------------------------------------------- markdown
def upsert_section(path: Path, key: str, body: str, title: str) -> None:
    """Replace (or append) the block between `<!-- agentbench:<key> -->` markers, keeping
    everything else in the file untouched."""
    begin, end = f"<!-- agentbench:begin {key} -->", f"<!-- agentbench:end {key} -->"
    block = f"{begin}\n{body.rstrip()}\n{end}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else f"# {title}\n\n"
    if begin in text and end in text:
        a, b = text.index(begin), text.index(end) + len(end)
        text = text[:a] + block.rstrip("\n") + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n" + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(text), encoding="utf-8")


def remove_section(path: Path, key: str) -> bool:
    """Drop the block between `<!-- agentbench:begin/end <key> -->` markers, if present."""
    if not path.exists():
        return False
    begin, end = f"<!-- agentbench:begin {key} -->", f"<!-- agentbench:end {key} -->"
    text = path.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        return False
    a, b = text.index(begin), text.index(end) + len(end)
    text = (text[:a].rstrip("\n") + "\n" + text[b:].lstrip("\n")).rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")
    return True


def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)


def fmt_s(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"
