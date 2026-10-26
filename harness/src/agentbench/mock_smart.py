"""Programmable mock responder that imitates the five frameworks' documented quirks.

Used by the tests and by `agentbench mock-server --behavior <name>` for integration smoke
runs. It is not a model: it answers the harness's known prompts from the fixtures (the
M7 reference calls, the G1 call, the G2 code read from the manifest), emits JSON-ish
filler for M3, honours max_tokens exactly, and simulates a message-granular prefix cache
so cached tokens and TTFT behave plausibly (cold > warm).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, AsyncIterator

LITHOS_FIELDS = frozenset({"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "seed",
                           "stop", "stream", "stream_options", "n", "tools", "tool_choice", "parallel_tool_calls",
                           "user", "metadata", "response_format", "reasoning_effort"})
VALID_EFFORTS = (None, "xhigh", "medium", "low")


@dataclass(frozen=True)
class Behavior:
    name: str = "generic"
    reasoning_field: str = "reasoning_content"  # reasoning_content | reasoning | inline
    images: bool = True
    tool_images: bool = True  # stock oMLX 0.7.1rc1 drops image_url parts of role=tool messages (OpenAI API)
    tool_result_images: bool = True  # False drops image blocks inside Anthropic tool_result blocks (test knob)
    split_usage: bool = False  # oMLX /v1/messages: with cache_control, input_tokens 0 and the prompt in the cache fields
    image_reject_status: int = 400
    image_reject_message: str = "image input is not supported"
    anthropic: bool = True
    strict_fields: frozenset | None = None
    effort_mode: str = "verbatim"  # verbatim (template error on bad values) | alias (mapped) | ignore | forbid
    can_think: bool = True
    default_thinking: bool = True  # Qwen3.8 template default when no thinking field is sent
    anthropic_thinking_allowed: tuple[str, ...] | None = None
    busy_429: bool = False
    tool_call_style: str = "incremental"  # incremental | single
    cached_report: str = "usage"  # usage | timings (usage + timings) | timings_only | log | extra | none
    template_error_status: int = 400
    track_model: bool = False  # mlx-lm: a model other than default_model reloads the model
    log_path: str | None = None
    reject_tools: bool = False  # mlc: `tools`, tool messages and assistant tool_calls are rejected
    structured: bool = False  # mlc: response_format {"type": "json_object", "schema": "<json string>"}
    prefix_cache: bool = True  # mlc runs with --prefix-cache-mode disable
    prefill_tps: float = 400_000.0
    token_delay: float = 0.0005


BEHAVIORS: dict[str, Behavior] = {
    "generic": Behavior(),
    "llamacpp": Behavior(name="llamacpp", cached_report="timings"),
    "omlx": Behavior(name="omlx", effort_mode="alias", tool_call_style="single", tool_images=False, split_usage=True),
    "vllm": Behavior(name="vllm", reasoning_field="reasoning"),
    "lithos": Behavior(name="lithos", images=False, strict_fields=LITHOS_FIELDS, effort_mode="forbid", can_think=False,
                       default_thinking=False, anthropic_thinking_allowed=("disabled",), busy_429=True,
                       cached_report="log"),
    "mlxlm": Behavior(name="mlxlm", reasoning_field="reasoning", images=False, image_reject_status=404,
                      image_reject_message="Only 'text' content type is supported.", anthropic=False,
                      effort_mode="ignore", tool_call_style="single", template_error_status=404, track_model=True),
    "mlc": Behavior(name="mlc", images=False, anthropic=False, effort_mode="ignore", can_think=False,
                    default_thinking=False, cached_report="extra", reject_tools=True, structured=True,
                    prefix_cache=False),
}

def _words(text: str) -> list[str]:
    ws = text.split(" ")
    return [w if i == 0 else " " + w for i, w in enumerate(ws)]


TEXT = _words("The screen shows the expected state of the window, so the next step can continue as planned. "
              "Nothing unexpected is visible and the previous action took effect.")
THINK = "Let me check the request carefully and work out the answer before replying.".split(" ")
JSON_UNIT = ('{"step": 1, "component": "scheduler", "action": "inspect", "target_host": "node-7", "priority": 2, '
             '"rationale": "Queue depth grew after the last deploy, so the scheduler state should be checked first."}, ')


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    out = []
    for p in content or []:
        if isinstance(p, dict) and p.get("type") == "text":
            out.append(p.get("text", ""))
        elif isinstance(p, dict) and p.get("type") == "tool_result":
            out.append(_text_of(p.get("content")))
    return "".join(out)


def _n_images(content: Any) -> int:
    """Image parts of a message content, including images nested in Anthropic tool_result blocks."""
    if not isinstance(content, list):
        return 0
    n = 0
    for p in content:
        if isinstance(p, dict) and p.get("type") in ("image_url", "image"):
            n += 1
        elif isinstance(p, dict) and p.get("type") == "tool_result":
            n += _n_images(p.get("content"))
    return n


def _drop_tool_images(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(m, content=[p for p in m["content"] if not (isinstance(p, dict) and p.get("type") == "image_url")])
            if m.get("role") == "tool" and isinstance(m.get("content"), list) else m for m in msgs]


def _drop_tool_result_images(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def strip(p: Any) -> Any:
        if isinstance(p, dict) and p.get("type") == "tool_result" and isinstance(p.get("content"), list):
            return dict(p, content=[q for q in p["content"] if not (isinstance(q, dict) and q.get("type") == "image")])
        return p
    return [dict(m, content=[strip(p) for p in m["content"]]) if isinstance(m.get("content"), list) else m for m in msgs]


def _has_cache_control(body: dict[str, Any]) -> bool:
    """A system, tool or message content block carries cache_control (oMLX `request_has_cache_control`)."""
    blocks = (body.get("system") if isinstance(body.get("system"), list) else []) + list(body.get("tools") or [])
    blocks += [p for m in body.get("messages") or [] if isinstance(m.get("content"), list) for p in m["content"]]
    return any(isinstance(p, dict) and p.get("cache_control") for p in blocks)


def _turn_images(msgs: list[dict[str, Any]]) -> int:
    """Images after the last assistant message: what the model is looking at now."""
    last_asst = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
    return sum(_n_images(m.get("content")) for m in msgs[last_asst + 1:])


def _user_text(msgs: list[dict[str, Any]]) -> str:
    return " ".join(_text_of(m.get("content")) for m in msgs if m.get("role") == "user")


def _strip_images(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("type") == "image_url":
            return {"type": "image_url", "sha": hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]}
        if obj.get("type") == "image":
            return {"type": "image", "sha": hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]}
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    return obj


def _sse(obj: dict[str, Any], event: str | None = None) -> str:
    return (f"event: {event}\n" if event else "") + "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@dataclass
class Plan:
    thinking: bool
    text: list[str] = field(default_factory=list)  # token strings
    calls: list[dict[str, Any]] = field(default_factory=list)


class SmartResponder:
    def __init__(self, behavior: Behavior | str = "generic", answers: dict[str, list[dict]] | None = None,
                 code: str | None = None, **overrides: Any) -> None:
        b = BEHAVIORS[behavior] if isinstance(behavior, str) else behavior
        self.b = replace(b, **overrides) if overrides else b
        self.answers = answers or {}
        self.code = code
        self.cache: list[list[tuple[str, int]]] = []
        self.in_flight = 0
        self.stats: Counter = Counter()
        self.requests: list[dict[str, Any]] = []

    @classmethod
    def from_fixtures(cls, root: Path, behavior: Behavior | str = "generic", **overrides: Any) -> "SmartResponder":
        root = Path(root)
        answers: dict[str, list[dict]] = {}
        m7 = json.loads((root / "m7_suite.json").read_text(encoding="utf-8"))
        for c in m7["cases"]:
            last = [m for m in c["messages"] if m["role"] == "user"][-1]
            answers[_text_of(last["content"])] = c["reference"]
        gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
        answers[gate["g1"]["prompt"]] = [{"name": "fs_read_file", "arguments": {"path": gate["g1"]["expect"]["path"]}}]
        code = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["g2_code"]
        return cls(behavior, answers, code, **overrides)

    # --------------------------------------------------------------- routing
    async def __call__(self, method: str, path: str, headers: dict[str, str], body: dict[str, Any]):
        if method == "GET":
            if path.startswith("/health"):
                return "json", 200, {"status": "ok"}
            if path.startswith("/v1/models"):
                return "json", 200, {"object": "list", "data": [{"id": "mock-model", "max_model_len": 131072}]}
            if path.startswith("/props"):
                return "json", 200, {"default_generation_settings": {"n_ctx": 131072}}
            return "json", 404, {"error": "Not Found"}
        self.stats["requests"] += 1
        self.requests.append({"path": path, "keys": sorted(body), "model": body.get("model", "<absent>"),
                              "stream_options": body.get("stream_options"), "seed": body.get("seed"),
                              "chat_template_kwargs": body.get("chat_template_kwargs"),
                              "reasoning_effort": body.get("reasoning_effort"), "thinking": body.get("thinking"),
                              "max_tokens": body.get("max_tokens"), "has_tools": "tools" in body,
                              "response_format": body.get("response_format"),
                              "roles": [m.get("role") for m in body.get("messages", [])],
                              "system": next((m.get("content") for m in body.get("messages", [])
                                              if m.get("role") == "system" and isinstance(m.get("content"), str)), None),
                              "n_images": sum(_n_images(m.get("content")) for m in body.get("messages", []))})
        if self.b.busy_429 and self.in_flight > 0:
            self.stats["429"] += 1
            return "json", 429, {"error": {"type": "rate_limit_error", "code": "model_busy",
                                           "message": "The model is busy; retry after the current request finishes"}}
        if path.startswith("/v1/chat/completions"):
            resp = self._openai(body)
        elif path.startswith("/v1/messages"):
            if not self.b.anthropic:
                return "json", 404, {"error": "Not Found"}
            resp = self._anthropic(body)
        else:
            return "json", 404, {"error": "Not Found"}
        if resp[0] == "sse":
            return "sse", await self._primed(resp[1])
        return resp

    # --------------------------------------------------------------- shared
    def _prompt_cache(self, segments: list[Any]) -> tuple[int, int]:
        segs = []
        for s in segments:
            stripped = json.dumps(_strip_images(s), ensure_ascii=False, sort_keys=True)
            n = len(stripped) // 4 + 1282 * stripped.count('"sha"')
            segs.append((hashlib.sha256(stripped.encode()).hexdigest(), n))
        total = sum(n for _, n in segs)
        best = 0
        for prev in self.cache:
            acc = 0
            for (h1, n1), (h2, _) in zip(segs, prev):
                if h1 != h2:
                    break
                acc += n1
            best = max(best, acc)
        self.cache = (self.cache + [segs])[-64:]
        return total, min(best, max(total - 1, 0))

    def _answer(self, last_user: Any, tools: bool, thinking: bool, images: int | None = None,
                question: str | None = None) -> Plan:
        """`images`/`question` (default: from `last_user`) let a screenshot that arrives in a tool
        result answer a question asked earlier in the conversation (gate G2b)."""
        text = _text_of(last_user)
        plan = Plan(thinking=thinking)
        n_img = _n_images(last_user) if images is None else images
        if self.code and n_img and "verification code" in (text if question is None else question).lower():
            plan.text = [self.code]
        elif tools and text in self.answers and self.answers[text]:
            plan.calls = self.answers[text]
        elif text in self.answers and not self.answers[text]:
            plan.text = _words("A 404 status means the server could not find the requested resource.")
        elif "JSON array" in text:
            plan.text = [JSON_UNIT[i:i + 4] for i in range(0, len(JSON_UNIT), 4)] * 400
        elif "17 multiplied by 23" in text:
            plan.text = ["391"]
        elif "single word: ready" in text:
            plan.text = ["ready"]
        else:
            plan.text = TEXT
        return plan

    def _log(self, cached: int, steps: int) -> None:
        if self.b.cached_report == "log" and self.b.log_path:
            with open(self.b.log_path, "a", encoding="utf-8") as f:
                f.write(f"INFO monolith.serve Generation: {{'steps': {steps}, 'cached_prompt_tokens': {cached}, "
                        f"'prefill_gpu_ms': 1.0}}\n")

    # --------------------------------------------------------------- OpenAI
    def _openai(self, body: dict[str, Any]):
        b = self.b
        if b.strict_fields is not None:
            extra = sorted(set(body) - b.strict_fields)
            if extra:
                return "json", 400, {"error": {"message": f"Extra inputs are not permitted: {extra}"}}
            if "model" not in body:
                return "json", 400, {"error": {"message": "model: Field required"}}
            so = body.get("stream_options")
            if so and set(so) - {"include_usage"}:
                return "json", 400, {"error": {"message": "Unsupported stream_options"}}
        so = body.get("stream_options")
        if so is not None and "include_usage" not in so:
            return "json", 500, {"error": "KeyError: 'include_usage'"}
        structured_reply = False
        if b.reject_tools:
            if "tools" in body:
                return "json", 400, {"error": "tools are not supported by this server"}
            if any(m.get("role") == "tool" or m.get("tool_calls") for m in body.get("messages") or []):
                return "json", 400, {"error": "tool messages and tool_calls are not supported"}
        if b.structured:
            rf = body.get("response_format")
            if rf is not None:
                if rf.get("type") not in ("text", "json_object"):
                    return "json", 422, {"detail": "response_format.type: Input should be 'text' or 'json_object'"}
                if "schema" in rf:
                    if not isinstance(rf["schema"], str):
                        return "json", 422, {"detail": "response_format.schema: Input should be a valid string"}
                    json.loads(rf["schema"])
            system = next((m.get("content") for m in body.get("messages") or [] if m.get("role") == "system"), "") or ""
            structured_reply = (rf or {}).get("type") == "json_object" or "Reply with exactly one JSON object" in str(system)
        if b.track_model and body.get("model", "default_model") != "default_model":
            self.stats["model_reloads"] += 1
        if "seed" in body:
            self.stats["seed_requests"] += 1
        msgs = body.get("messages") or []
        if not b.images and any(_n_images(m.get("content")) for m in msgs):
            return "json", b.image_reject_status, {"error": b.image_reject_message}
        if not b.tool_images:
            msgs = _drop_tool_images(msgs)
        kwargs = body.get("chat_template_kwargs") or {}
        effort = body.get("reasoning_effort")
        enable = kwargs.get("enable_thinking")
        if b.effort_mode == "forbid":
            if effort not in (None, "none"):
                return "json", 400, {"error": {"message": "Use reasoning_effort=none; this serving path disables thinking"}}
            thinking = False
        else:
            if enable is None:
                enable = (effort != "none") if (b.effort_mode == "verbatim" and effort) else b.default_thinking
            thinking = bool(enable) and b.can_think
            if thinking:
                t_eff = kwargs.get("reasoning_effort", effort if b.effort_mode == "verbatim" else None)
                if t_eff not in VALID_EFFORTS:
                    return "json", b.template_error_status, {"error": f"Unexpected reasoning effort {t_eff}."}
        last_user = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), "")
        plan = self._answer(last_user, bool(body.get("tools")) or structured_reply, thinking,
                            images=_turn_images(msgs), question=_user_text(msgs))
        if structured_reply:  # what a grammar-constrained server returns: one JSON step object as content
            if plan.calls:
                steps = [{"tool": c["name"], "arguments": c["arguments"]} for c in plan.calls]
                obj: dict[str, Any] = steps[0] if len(steps) == 1 else {"tool_calls": steps}
            else:
                obj = {"final": "".join(plan.text)}
            js = json.dumps(obj, ensure_ascii=False)
            plan = Plan(thinking=False, text=[js[i:i + 6] for i in range(0, len(js), 6)])
        total, cached = self._prompt_cache([body.get("tools")] + msgs)
        if not b.prefix_cache:
            cached = 0
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 256
        include_usage = bool((so or {}).get("include_usage"))
        return "sse", self._openai_stream(plan, total, cached, max_tokens, include_usage)

    @staticmethod
    async def _primed(gen: AsyncIterator[str]) -> AsyncIterator[str]:
        """Run the stream up to its first event now, so that `in_flight` is incremented
        atomically with accepting the request (lithos-style 429 checks are race-free)."""
        first = await gen.__anext__()

        async def chain() -> AsyncIterator[str]:
            try:
                yield first
                async for x in gen:
                    yield x
            finally:
                await gen.aclose()

        return chain()

    async def _openai_stream(self, plan: Plan, total: int, cached: int, max_tokens: int,
                             include_usage: bool) -> AsyncIterator[str]:
        b = self.b
        self.in_flight += 1
        try:
            def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
                return _sse({"object": "chat.completion.chunk", "model": "mock",
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

            yield chunk({"role": "assistant", "content": ""})
            yield ": keep-alive\n\n"
            await asyncio.sleep((total - cached) / b.prefill_tps)
            n = 0
            if plan.thinking:
                for i, w in enumerate(THINK[: max(1, min(len(THINK), max_tokens // 2))]):
                    if n >= max_tokens:
                        break
                    piece = w if i == 0 else " " + w
                    if b.reasoning_field == "inline":
                        yield chunk({"content": ("<think>\n" if i == 0 else "") + piece})
                    else:
                        yield chunk({b.reasoning_field: piece})
                    n += 1
                    await asyncio.sleep(b.token_delay)
                if b.reasoning_field == "inline" and n < max_tokens:
                    yield chunk({"content": "\n</think>\n\n"})
            finish = "stop"
            if plan.calls and n < max_tokens:
                for idx, c in enumerate(plan.calls):
                    args = json.dumps(c["arguments"], ensure_ascii=False)
                    cid = f"call_mock_{idx}"
                    if b.tool_call_style == "single":
                        await asyncio.sleep(b.token_delay * 20)
                        yield chunk({"tool_calls": [{"index": idx, "id": cid, "type": "function",
                                                     "function": {"name": c["name"], "arguments": args}}]})
                        n += 20
                    else:
                        yield chunk({"tool_calls": [{"index": idx, "id": cid, "type": "function",
                                                     "function": {"name": c["name"], "arguments": ""}}]})
                        for i in range(0, len(args), 8):
                            await asyncio.sleep(b.token_delay)
                            yield chunk({"tool_calls": [{"index": idx, "function": {"arguments": args[i:i + 8]}}]})
                            n += 1
                finish = "tool_calls"
            else:
                sent = 0
                for tok in plan.text:
                    if n >= max_tokens:
                        break
                    yield chunk({"content": tok})
                    n += 1
                    sent += 1
                    await asyncio.sleep(b.token_delay)
                finish = "length" if sent < len(plan.text) else "stop"
            fin: dict[str, Any] = {"object": "chat.completion.chunk", "model": "mock",
                                   "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            if b.cached_report in ("timings", "timings_only"):
                fin["timings"] = {"cache_n": cached, "prompt_n": total - cached, "predicted_n": n}
            yield _sse(fin)
            if include_usage:
                usage: dict[str, Any] = {"prompt_tokens": total, "completion_tokens": n, "total_tokens": total + n}
                if b.cached_report in ("usage", "timings"):
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}
                if b.cached_report == "extra":  # MLC --enable-debug: no prefix cache, prefill == prompt
                    usage["extra"] = {"prompt_tokens": total, "prefill_tokens": total, "ttft_s": 0.0}
                yield _sse({"object": "chat.completion.chunk", "model": "mock", "choices": [], "usage": usage})
            self._log(cached, n)
            yield "data: [DONE]\n\n"
        finally:
            self.in_flight -= 1

    # ------------------------------------------------------------ Anthropic
    def _anthropic(self, body: dict[str, Any]):
        b = self.b
        th = (body.get("thinking") or {}).get("type")
        if b.anthropic_thinking_allowed is not None and th is not None and th not in b.anthropic_thinking_allowed:
            return "json", 400, {"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "Extended thinking is unavailable"}}
        if "max_tokens" not in body:
            return "json", 500, {"type": "error", "error": {"type": "api_error", "message": "KeyError: 'max_tokens'"}}
        msgs = body.get("messages") or []
        if any(_n_images(m.get("content")) for m in msgs) and not b.images:
            return "json", 400, {"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "Unsupported Messages content block: image"}}
        if not b.tool_result_images:
            msgs = _drop_tool_result_images(msgs)
        thinking = (th == "enabled") or (th in (None, "adaptive") and b.default_thinking and b.can_think)
        last_user = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), "")
        plan = self._answer(last_user, bool(body.get("tools")), thinking, question=_user_text(msgs))
        total, cached = self._prompt_cache([body.get("tools"), body.get("system")] + msgs)
        return "sse", self._anthropic_stream(plan, total, cached, body["max_tokens"], _has_cache_control(body))

    async def _anthropic_stream(self, plan: Plan, total: int, cached: int, max_tokens: int,
                                cache_control: bool = False) -> AsyncIterator[str]:
        b = self.b
        self.in_flight += 1
        try:
            # Anthropic usage is disjoint: input_tokens + cache_creation + cache_read = prompt
            final: dict[str, Any] = {}
            if b.split_usage:  # oMLX: the real numbers come in message_delta; the cache fields only with cache_control
                usage: dict[str, Any] = {"input_tokens": 0 if cache_control else total, "output_tokens": 0}
                final = ({"input_tokens": 0, "cache_creation_input_tokens": total - cached,
                          "cache_read_input_tokens": cached} if cache_control else {"input_tokens": total})
            else:
                usage = {"input_tokens": total, "output_tokens": 1}
                if b.cached_report in ("usage", "timings"):  # as llama.cpp: input_tokens leaves out the cache hits
                    usage.update(input_tokens=total - cached, cache_read_input_tokens=cached)
            yield _sse({"type": "message_start", "message": {"id": "msg_mock", "role": "assistant", "content": [],
                                                              "usage": usage}}, "message_start")
            yield _sse({"type": "ping"}, "ping")
            await asyncio.sleep((total - cached) / b.prefill_tps)
            idx, n = 0, 0
            if plan.thinking:
                yield _sse({"type": "content_block_start", "index": idx, "content_block": {"type": "thinking", "thinking": ""}},
                           "content_block_start")
                for w in THINK:
                    yield _sse({"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": w + " "}},
                               "content_block_delta")
                    n += 1
                    await asyncio.sleep(b.token_delay)
                yield _sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
                idx += 1
            stop = "end_turn"
            if plan.calls:
                for k, c in enumerate(plan.calls):
                    yield _sse({"type": "content_block_start", "index": idx, "content_block": {
                        "type": "tool_use", "id": f"toolu_mock_{k}", "name": c["name"], "input": {}}}, "content_block_start")
                    args = json.dumps(c["arguments"], ensure_ascii=False)
                    for i in range(0, len(args), 8):
                        yield _sse({"type": "content_block_delta", "index": idx,
                                    "delta": {"type": "input_json_delta", "partial_json": args[i:i + 8]}}, "content_block_delta")
                        n += 1
                        await asyncio.sleep(b.token_delay)
                    yield _sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
                    idx += 1
                stop = "tool_use"
            else:
                yield _sse({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}},
                           "content_block_start")
                sent = 0
                for tok in plan.text:
                    if n >= max_tokens:
                        break
                    yield _sse({"type": "content_block_delta", "index": idx,
                                "delta": {"type": "text_delta", "text": tok}}, "content_block_delta")
                    n += 1
                    sent += 1
                    await asyncio.sleep(b.token_delay)
                if sent < len(plan.text):
                    stop = "max_tokens"
                yield _sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
            yield _sse({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": n, **final}},
                       "message_delta")
            self._log(cached, n)
            yield _sse({"type": "message_stop"}, "message_stop")
        finally:
            self.in_flight -= 1
