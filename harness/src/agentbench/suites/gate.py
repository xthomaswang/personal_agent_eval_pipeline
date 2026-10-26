"""Gate G1-G5 plus G2b (METHODOLOGY section 1), with Hermes-style and Claude Code-style requests.

Hermes-style (OpenAI API): every request carries top-level `reasoning_effort` ("none"
when thinking is off) and `chat_template_kwargs.enable_thinking`. Claude Code-style
(`/v1/messages`): `thinking: {type: "adaptive"}` or no thinking field, never "disabled";
system prompt as text blocks with `cache_control`.
Images are always probed, even with --text-only: a rejection is the gate result.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

import jsonschema

from ..m7check import _norm
from ..memory import MemorySampler
from ..provenance import harness_git
from .common import Outcome, RunConfig, Target, Writer, fmt_s, upsert_section
from ..fixtures.store import new_digits

REQ = {"G1": "OpenAI chat completions + SSE + Qwen3.8 tool calls",
       "G2": "Image input (base64 PNG image_url), code read back",
       "G2b": "Image inside a tool result (role=tool image_url, Hermes path), code read back",
       "G3": "Anthropic /v1/messages: text, tool_use, image, image in tool_result",
       "G4": "Thinking control (enable_thinking / reasoning_effort)",
       "G5": "Context >= 64K: a 64,000-token prompt completes"}

HERMES_OFF = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
HERMES_ON = {"reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": True}}
MATH = "What is 17 multiplied by 23? Reply with the number only."
SHOT_ASK = ("Take a screenshot with the computer_screenshot tool. The screen shows a dialog with a verification "
            "code. Reply with the verification code exactly as it is shown, and nothing else.")
SHOT_OK = "Screenshot captured (1280x1024)."
SHOT_ID = "call_g2b_screenshot"
DROP_TOKENS = 600  # an image is ~1.3K prompt tokens; a smaller gap is noise
ANTHROPIC_INPUT = (("input_tokens", "input_tokens"), ("cache_creation_input_tokens", "cache_creation"),
                   ("cache_read_input_tokens", "cache_read"))  # disjoint usage fields, summed to the prompt size


def _alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _style(rec) -> str:
    if rec.tool_call_mode == "structured_output":
        n = sum(1 for c in rec.chunks if c.kind == "content")
        return f"JSON content in {n} chunks; first chunk at {fmt_s(rec.ttft, 2)} s, e2e {fmt_s(rec.e2e, 2)} s"
    n = sum(1 for c in rec.chunks if c.kind in ("tool_call", "input_json"))
    if not rec.tool_calls:
        return "no tool call"
    first = next((c.t for c in rec.chunks if c.kind in ("tool_call", "input_json")), None)
    style = "one delta" if n <= len(rec.tool_calls) else f"incremental ({n} deltas)"
    return f"{style}; first call chunk at {fmt_s(first, 2)} s, e2e {fmt_s(rec.e2e, 2)} s"


def _reasoning(o: Outcome) -> str:
    r = o.rec
    if not r.reasoning:
        return "no reasoning"
    where = "+".join(r.reasoning_fields) or "?"
    return f"reasoning {len(r.reasoning)} chars in {where}"


def _http(o: Outcome) -> str:
    if o.skipped:
        return "not sent"
    if o.rec.error and o.rec.error.get("type") == "http":
        return f"HTTP {o.rec.http_status}"
    if o.rec.error:
        return o.rec.error.get("type", "error")
    return f"HTTP {o.rec.http_status}"


def _anthropic_prompt(rec) -> tuple[int | None, str]:
    """Prompt size of a /v1/messages request and how it was reported. Anthropic usage splits the
    prompt into three disjoint fields, so the prompt is their sum (a missing field counts as 0):
    with `cache_control` oMLX reports input_tokens 0 and the rest as cache_creation/cache_read,
    and llama.cpp and vLLM leave prefix-cache hits out of input_tokens. `usage_raw` is already
    the final usage (message_delta fields over message_start). None when no field is reported."""
    u = rec.usage_raw or {}
    if not any(isinstance(u.get(k), (int, float)) for k, _ in ANTHROPIC_INPUT):
        return None, "input_tokens=None"
    n = {k: int(u[k]) if isinstance(u.get(k), (int, float)) else 0 for k, _ in ANTHROPIC_INPUT}
    total = sum(n.values())
    split = [f"+{label} {n[k]}" for k, label in ANTHROPIC_INPUT[1:] if n[k]]
    return total, f"input_tokens={n['input_tokens']}" + ("".join(split) + f" (sum {total})" if split else "")


class Gate:
    def __init__(self, cfg: RunConfig, target: Target, writer: Writer) -> None:
        self.cfg, self.t, self.w = cfg, target, writer
        self.fx = target.fx
        self.items: dict[str, dict[str, Any]] = {}
        self.probes: list[dict[str, Any]] = []

    def item(self, gid: str, result: str, detail: str, sub: list[dict[str, Any]] | None = None, **extra: Any) -> None:
        self.items[gid] = {"id": gid, "requirement": REQ[gid], "result": result, "detail": detail, "sub": sub or [], **extra}
        self.w.write({"type": "gate_item", **self.items[gid]})

    def record(self, pid: str, shape: str, o: Outcome, outcome: str, ok: bool | None) -> None:
        notes = "; ".join(o.shaping + ([o.skipped] if o.skipped else []))
        self.probes.append({"probe": pid, "shape": shape, "http": _http(o), "outcome": outcome, "ok": ok, "notes": notes})
        self.w.write(o.line("gate", self.t.count_tokens if o.rec.ok else None, probe=pid, shape=shape,
                            judgement={"ok": ok, "outcome": outcome}))

    def err(self, o: Outcome) -> str:
        return o.skipped or (o.metrics()["error"] or "no output")

    # -------------------------------------------------------------------- G1
    async def g1(self) -> None:
        fx = self.fx
        msgs = [fx.agent_system(), {"role": "user", "content": fx.gate["g1"]["prompt"]}]
        exp = fx.gate["g1"]["expect"]

        def judge(calls, content) -> tuple[bool, str]:
            if not calls:
                return False, f"no tool call (content: {content[:80]!r})"
            c = calls[0]
            args = c.get("arguments_parsed")
            if c.get("name") != exp["name"]:
                return False, f"called {c.get('name')!r}"
            if not isinstance(args, dict):
                return False, f"arguments not JSON-parseable: {c.get('arguments')!r}"
            if _norm(args.get("path"), "path") != exp["path"]:
                return False, f"path {args.get('path')!r}"
            return True, f"{c['name']} {c['arguments']}"

        structured = self.t.profile.tool_call_mode == "structured_output"
        schema = fx.schemas()[exp["name"]]

        def judge_structured(o: Outcome) -> tuple[bool, str]:
            if not o.rec.ok:
                return False, self.err(o)
            if o.rec.structured_error:
                return False, f"structured output did not parse: {o.rec.structured_error} (raw {str(o.rec.structured_raw)[:80]!r})"
            ok, what = judge(o.rec.tool_calls, o.rec.content)
            if ok:
                errs = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(
                    o.rec.tool_calls[0]["arguments_parsed"])]
                if errs:
                    return False, f"arguments fail the tool schema: {errs[0]}"
            return ok, what

        o = await self.t.chat(msgs, tools=fx.tools(), max_tokens=256, raw_fields=HERMES_OFF)
        if structured:
            ok, what = judge_structured(o)
        else:
            ok, what = judge(o.rec.tool_calls, o.rec.content) if o.rec.ok else (False, self.err(o))
        leak = o.rec.markup_leak()
        if leak:
            ok, what = False, what + f"; markup leaked into content: {leak}"
        shape = "Hermes (reasoning_effort=none, enable_thinking=false)"
        if structured:
            shape = f"structured output (response_format={self.t.profile.structured_format}, tools inlined)"
        self.record("G1.hermes", shape, o, what, ok)
        sub = [{"probe": "G1.hermes", "ok": ok, "outcome": what}]
        if structured:
            if not ok and o.rec.http_status in (400, 404, 422):
                p = await self.t.chat(msgs, tools=fx.tools(), max_tokens=256, raw_fields=HERMES_OFF,
                                      structured_format="none")
                pok, pwhat = judge_structured(p)
                self.record("G1.no-grammar", "structured output without response_format (instruction only)", p, pwhat, pok)
                sub.append({"probe": "G1.no-grammar", "ok": pok, "outcome": pwhat})
            detail = what + (f"; {_style(o.rec)}; finish={o.rec.finish_reason}" if ok else "")
            self.item("G1", "PASS via structured output (non-standard)" if ok else "FAIL", detail, sub,
                      tool_call_mode="structured_output")
            return
        if not ok and o.rec.http_status in (400, 404, 422):
            p = await self.t.chat(msgs, tools=fx.tools(), max_tokens=2048, raw_fields={})
            pok, pwhat = judge(p.rec.tool_calls, p.rec.content) if p.rec.ok else (False, self.err(p))
            self.record("G1.plain", "plain (no thinking fields)", p, pwhat, pok)
            sub.append({"probe": "G1.plain", "ok": pok, "outcome": pwhat})
        stripped = [n for n in o.shaping if "stripped" in n]
        if stripped:
            sub.append({"probe": "G1.hermes-unmodified", "ok": None,
                        "outcome": "N/A: the unmodified Hermes shape is not sent (" + "; ".join(stripped) + ")"})
        detail = what + (f"; {_style(o.rec)}; finish={o.rec.finish_reason}" if ok else "")
        if stripped:
            detail += " [profile stripped: " + "; ".join(stripped) + "]"
        self.item("G1", "PASS" if ok else "FAIL", detail, sub)

    # -------------------------------------------------------------------- G2
    async def g2(self) -> None:
        fx = self.fx
        code = fx.manifest["g2_code"]
        msgs = fx.resolve([fx.screenshot_message(fx.gate["g2"]["image"], fx.gate["g2"]["prompt"])], text_only=False)
        o = await self.t.chat(msgs, max_tokens=64, raw_fields=HERMES_OFF, count_local=True)
        if not o.rec.ok:
            ok, what = False, "rejected: " + self.err(o)
        else:
            ans = o.rec.content.strip()
            ok = _alnum(code) in _alnum(ans)
            pt = o.rec.prompt_tokens
            what = (f"read {code} correctly" if ok else f"image accepted but code not read (answer {ans[:40]!r}, expected {code})")
            if pt is not None:
                what += f"; prompt_tokens={pt}" + (" (image probably dropped)" if pt < 600 else "")
        self.record("G2.image", "Hermes, image_url data URI + question", o, what, ok)
        rejected = bool(o.rec.error and o.rec.error.get("type") == "http" and (o.rec.http_status or 0) >= 400)
        self.item("G2", "PASS" if ok else "FAIL", what, rejected=rejected)

    # ------------------------------------------------------------------- G2b
    def shot_tools(self, api: str = "openai") -> list[dict[str, Any]]:
        key = (lambda t: t["name"]) if api == "anthropic" else (lambda t: t["function"]["name"])
        return [t for t in self.fx.tools(api=api) if key(t) == "computer_screenshot"]

    def g2b_messages(self, in_tool: bool) -> list[dict[str, Any]]:
        """Screenshot returned by a tool: in the role=tool message itself (what Hermes sends once
        `supports_vision` is set), or, as a control, in a following user message."""
        img = self.fx.gate["g2"]["image"]
        call = {"id": SHOT_ID, "type": "function", "function": {"name": "computer_screenshot", "arguments": "{}"}}
        # a fresh nonce first, so neither request reuses the other's prefix and prompt_tokens compare cleanly
        msgs: list[dict[str, Any]] = [{"role": "user", "content": f"[run {new_digits()}] {SHOT_ASK}"},
                                      {"role": "assistant", "content": "I'll take a screenshot.", "tool_calls": [call]}]
        if in_tool:
            msgs.append({"role": "tool", "tool_call_id": SHOT_ID,
                         "content": [{"type": "text", "text": SHOT_OK}, {"type": "image_ref", "image": img}]})
        else:
            msgs += [{"role": "tool", "tool_call_id": SHOT_ID, "content": SHOT_OK + " It is attached below."},
                     self.fx.screenshot_message(img, "Here is the screenshot.")]
        return self.fx.resolve(msgs, text_only=False)

    def read_code(self, o: Outcome) -> tuple[bool, str]:
        code = self.fx.manifest["g2_code"]
        if not o.rec.ok:
            return False, "rejected: " + self.err(o)
        ans = o.rec.content.strip()
        if _alnum(code) in _alnum(ans):
            return True, f"read {code}"
        calls = [c.get("name") for c in o.rec.tool_calls]
        return False, (f"code not read (answer {ans[:40]!r}" + (f", tool calls {calls}" if calls else "") + f", expected {code})")

    @staticmethod
    def drop_note(pt: int | None, pt_ctrl: int | None) -> tuple[bool, str]:
        if pt is None or pt_ctrl is None:
            return False, ""
        gap = pt_ctrl - pt
        return gap > DROP_TOKENS, (f"; image dropped: {gap} fewer prompt tokens than the control" if gap > DROP_TOKENS else "")

    async def g2b(self) -> None:
        tools = self.shot_tools()
        o = await self.t.chat(self.g2b_messages(True), tools=tools, max_tokens=64, raw_fields=HERMES_OFF, count_local=True)
        c = await self.t.chat(self.g2b_messages(False), tools=tools, max_tokens=64, raw_fields=HERMES_OFF, count_local=True)
        ok, what = self.read_code(o)
        cok, cwhat = self.read_code(c)
        dropped, note = self.drop_note(o.rec.prompt_tokens, c.rec.prompt_tokens)
        what += f"; prompt_tokens={o.rec.prompt_tokens}" + note
        cwhat += f"; prompt_tokens={c.rec.prompt_tokens}"
        self.record("G2b.tool-image", "Hermes, screenshot as image_url inside the role=tool message", o, what, ok)
        self.record("G2b.user-control", "Hermes, same screenshot in a following user message (control)", c, cwhat, cok)
        sub = [{"probe": "G2b.tool-image", "ok": ok, "outcome": what},
               {"probe": "G2b.user-control", "ok": cok, "outcome": cwhat}]
        detail = f"tool-message image: {what}; control (user message): {cwhat}"
        self.item("G2b", "PASS" if ok else "FAIL", detail, sub, dropped=dropped)

    # -------------------------------------------------------------------- G3
    def cc_body(self, messages: list[dict[str, Any]], system: str, tools: list | None = None,
                thinking: str | None = None, max_tokens: int = 4096) -> dict[str, Any]:
        body: dict[str, Any] = {"max_tokens": max_tokens, "stream": True,
                                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                "messages": messages}
        if tools:
            body["tools"] = tools
        if thinking:
            body["thinking"] = {"type": thinking}
        return body

    async def g3(self) -> None:
        fx = self.fx
        code = fx.manifest["g2_code"]
        sub = []
        text_ok = False
        for th in ("adaptive", None):
            pid = f"G3.text-{th or 'nothinking'}"
            shape = f"Claude Code, thinking={th or 'absent'}"
            blocked = self.t.profile.anthropic_thinking_blocked(th)
            if blocked:
                self.probes.append({"probe": pid, "shape": shape, "http": "not sent", "outcome": "N/A: " + blocked,
                                    "ok": None, "notes": ""})
                sub.append({"probe": pid, "ok": None, "outcome": "N/A: " + blocked})
                continue
            o = await self.t.anthropic(self.cc_body([{"role": "user", "content": [
                {"type": "text", "text": "Reply with the single word: ready"}]}], "You are a helpful assistant.", thinking=th))
            truncated = (o.rec.ok and not o.rec.content.strip() and bool(o.rec.reasoning)
                         and o.rec.finish_reason in ("max_tokens", "length"))
            ok = o.rec.ok and (bool(o.rec.content.strip()) or truncated)
            if truncated:
                what = f"streaming ok, answer truncated by default thinking ({_reasoning(o)}, finish={o.rec.finish_reason})"
            elif ok:
                what = f"text ok ({o.rec.content.strip()[:24]!r}); {_reasoning(o)}"
            else:
                what = self.err(o) if not o.rec.ok else f"no text ({_reasoning(o)}, finish={o.rec.finish_reason})"
            self.record(pid, shape, o, what, ok)
            sub.append({"probe": pid, "ok": ok, "outcome": what})
            text_ok = text_ok or ok
        # tool_use
        o = await self.t.anthropic(self.cc_body([{"role": "user", "content": fx.gate["g1"]["prompt"]}], fx.system_prompt,
                                                tools=fx.tools(api="anthropic")))
        calls = o.rec.tool_calls
        tool_ok = bool(o.rec.ok and calls and calls[0]["name"] == "fs_read_file"
                       and isinstance(calls[0].get("arguments_parsed"), dict)
                       and _norm(calls[0]["arguments_parsed"].get("path"), "path") == "docs/setup.md")
        what = (f"tool_use {calls[0]['name']} {calls[0]['arguments_parsed']}; {_style(o.rec)}" if tool_ok
                else (self.err(o) if not o.rec.ok else f"no valid tool_use (calls={[c.get('name') for c in calls]})"))
        self.record("G3.tool_use", "Claude Code, tools, thinking absent", o, what, tool_ok)
        sub.append({"probe": "G3.tool_use", "ok": tool_ok, "outcome": what})
        # image block
        o = await self.t.anthropic(self.cc_body([{"role": "user", "content": fx.resolve_content(
            [{"type": "image_ref", "image": fx.gate["g2"]["image"]}, {"type": "text", "text": fx.gate["g2"]["prompt"]}],
            text_only=False, api="anthropic")}], "You are a helpful assistant."))
        img_ok = bool(o.rec.ok and _alnum(code) in _alnum(o.rec.content))
        what = (f"read {code}" if img_ok else (self.err(o) if not o.rec.ok else f"code not read ({o.rec.content[:40]!r})"))
        self.record("G3.image", "Claude Code, image block (base64)", o, what, img_ok)
        sub.append({"probe": "G3.image", "ok": img_ok, "outcome": what})
        # screenshot returned by a tool: image block inside tool_result (Claude Code MCP), and a control
        img_block = fx.resolve_content([{"type": "image_ref", "image": fx.gate["g2"]["image"]}], text_only=False,
                                       api="anthropic")
        def ask() -> dict[str, Any]:  # fresh nonce per request (see g2b_messages)
            return {"role": "user", "content": [{"type": "text", "text": f"[run {new_digits()}] {SHOT_ASK}"}]}
        use = {"role": "assistant", "content": [{"type": "text", "text": "I'll take a screenshot."},
                                                {"type": "tool_use", "id": "toolu_g3_screenshot",
                                                 "name": "computer_screenshot", "input": {}}]}
        in_result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_g3_screenshot",
                                                  "content": img_block + [{"type": "text", "text": SHOT_OK}]}]}
        control = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_g3_screenshot",
                                                "content": SHOT_OK + " It is attached below."}]
                   + img_block + [{"type": "text", "text": "Here is the screenshot."}]}
        tools = self.shot_tools("anthropic")
        o = await self.t.anthropic(self.cc_body([ask(), use, in_result], "You are a helpful assistant.", tools=tools))
        c = await self.t.anthropic(self.cc_body([ask(), use, control], "You are a helpful assistant.", tools=tools))
        tr_ok, what = self.read_code(o)
        cok, cwhat = self.read_code(c)
        # compare whole prompts: input_tokens alone misses the cache fields (see _anthropic_prompt)
        (pt, pt_text), (pt_ctrl, ctrl_text) = _anthropic_prompt(o.rec), _anthropic_prompt(c.rec)
        dropped, note = self.drop_note(pt, pt_ctrl)
        what += f"; {pt_text}" + note
        cwhat += f"; {ctrl_text}"
        self.record("G3.tool_result-image", "Claude Code, screenshot as image block inside tool_result", o, what, tr_ok)
        self.record("G3.tool_result-control", "Claude Code, same image after the tool_result (control)", c, cwhat, cok)
        sub.append({"probe": "G3.tool_result-image", "ok": tr_ok, "outcome": what})
        sub.append({"probe": "G3.tool_result-control", "ok": cok, "outcome": cwhat})
        parts = [("text", text_ok), ("tool_use", tool_ok), ("image", img_ok), ("tool_result image", tr_ok)]
        res = "PASS" if all(v for _, v in parts) else "FAIL"
        detail = ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in parts)
        first_fail = next((x["outcome"] for x in sub if x["ok"] is False), None)
        if first_fail:
            detail += f" (first failure: {first_fail[:160]})"
        thinking_notes = [s["outcome"] for s in sub if s["probe"].startswith("G3.text") and s["ok"]]
        if thinking_notes:
            detail += "; " + " / ".join(thinking_notes)
        self.item("G3", res, detail, sub, dropped=dropped)

    # -------------------------------------------------------------------- G4
    async def g4(self) -> None:
        msgs = [{"role": "user", "content": MATH}]
        probes = [("G4.hermes-off", HERMES_OFF, False), ("G4.hermes-on", HERMES_ON, True),
                  ("G4.kwargs-off", {"chat_template_kwargs": {"enable_thinking": False}}, False),
                  ("G4.kwargs-on", {"chat_template_kwargs": {"enable_thinking": True}}, True),
                  ("G4.effort-none-only", {"reasoning_effort": "none"}, None),
                  ("G4.effort-high", {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": True}}, None),
                  ("G4.effort-minimal", {"reasoning_effort": "minimal", "chat_template_kwargs": {"enable_thinking": True}}, None),
                  ("G4.default", {}, None)]
        res: dict[str, tuple[bool | None, str]] = {}
        sub = []
        for pid, fields, want in probes:
            needs_on = want is True or "high" in str(fields) or "minimal" in str(fields)
            blocked = self.t.profile.thinking_blocked(True) if needs_on else None
            shape = ", ".join(f"{k}={v}" for k, v in fields.items()) or "no thinking fields"
            if blocked:
                self.probes.append({"probe": pid, "shape": shape, "http": "not sent", "outcome": "N/A: " + blocked,
                                    "ok": None, "notes": ""})
                res[pid] = (None, "N/A: " + blocked)
                sub.append({"probe": pid, "ok": None, "outcome": "N/A: " + blocked})
                continue
            o = await self.t.chat(msgs, max_tokens=512 if want is not False else 64, raw_fields=fields, count_local=False)
            has = bool(o.rec.reasoning)
            if not o.rec.ok:
                ok, what = (False if want is not None else None), "error: " + self.err(o)
            elif want is None:
                ok, what = None, ("accepted; " + _reasoning(o))
            else:
                ok = has == want
                what = _reasoning(o) + ("" if ok else (" (expected reasoning)" if want else " (expected none)"))
            stripped = [n for n in o.shaping if "stripped" in n]
            if stripped:
                what += " [shaped: " + "; ".join(stripped) + "]"
            self.record(pid, shape, o, what, ok)
            res[pid] = (ok, what)
            sub.append({"probe": pid, "ok": ok, "outcome": what})
        off_ok, on_ok = res["G4.hermes-off"][0], res["G4.hermes-on"][0]
        kw_ok = res["G4.kwargs-off"][0] and res["G4.kwargs-on"][0]
        if off_ok and on_ok:
            result = "PASS"
            detail = f"Hermes off: {res['G4.hermes-off'][1]}; Hermes on: {res['G4.hermes-on'][1]}"
        elif on_ok is None:
            result = "FAIL"
            detail = res["G4.hermes-on"][1] + f"; off: {res['G4.hermes-off'][1]}"
        elif kw_ok:
            result = "PARTIAL"
            detail = (f"Hermes shape does not work (off: {res['G4.hermes-off'][1]}; on: {res['G4.hermes-on'][1]}), "
                      "but enable_thinking alone does")
        else:
            result = "FAIL"
            detail = f"Hermes off: {res['G4.hermes-off'][1]}; on: {res['G4.hermes-on'][1]}; enable_thinking alone does not work"
        eff_none = res.get("G4.effort-none-only")
        if eff_none and eff_none[0] is None and "no reasoning" not in eff_none[1] and not eff_none[1].startswith("N/A"):
            detail += "; note: reasoning_effort=none alone does not disable thinking (top-level effort ignored)"
        info = [f"{p}: {res[p][1]}" for p in ("G4.effort-none-only", "G4.effort-high", "G4.effort-minimal", "G4.default")
                if p in res]
        self.item("G4", result, detail + (" | " + " | ".join(info) if info else ""), sub)

    # -------------------------------------------------------------------- G5
    async def g5(self) -> None:
        info: dict[str, Any] = {}
        if self.t.profile.probe_models:
            st, js = await self.t.get_json("/v1/models")
            info["v1_models"] = {"status": st, "context_fields": _ctx_fields(js)}
        if self.t.profile.name == "llamacpp":
            st, js = await self.t.get_json("/props")
            info["props"] = {"status": st, "context_fields": _ctx_fields(js)}
        self.w.write({"type": "server_info", **info})
        size = 2000 if self.cfg.quick else 64000
        msgs = self.fx.prompt_messages(f"cold-{size}", nonce=new_digits())
        peak = None
        if self.cfg.server_pid:
            async with MemorySampler(self.cfg.server_pid, sink=lambda s: self.w.write(s), interval=1.0) as ms:
                o = await self.t.chat(msgs, max_tokens=8, raw_fields=HERMES_OFF)
            peak = ms.peak_tree_bytes()
        else:
            o = await self.t.chat(msgs, max_tokens=8, raw_fields=HERMES_OFF)
        ok = o.rec.ok
        m = o.metrics()
        ctx = self.cfg.context_config or next((v for d in info.values() for v in d["context_fields"].values()), None)
        what = (f"{size}-token prompt completed: prompt_tokens={m['prompt_tokens']}, TTFT {fmt_s(m['ttft_s'], 1)} s"
                if ok else f"{size}-token prompt failed: {self.err(o)}")
        what += f"; configured context: {ctx if ctx else 'unknown'}"
        if peak:
            what += f"; peak server footprint {peak / 2**30:.1f} GiB"
        self.record(f"G5.cold-{size}", "Hermes off, cold prompt with nonce", o, what, ok)
        result = ("INFO" if self.cfg.quick else ("PASS" if ok else "FAIL"))
        if self.cfg.quick:
            what = "quick mode (2K prompt, not a G5 result): " + what
        self.item("G5", result, what)

    # ------------------------------------------------------------------ report
    def table(self) -> str:
        rows = ["| ID | Requirement | Result | Detail |", "| --- | --- | --- | --- |"]
        for gid in ("G1", "G2", "G2b", "G3", "G4", "G5"):
            it = self.items.get(gid)
            if it:
                rows.append(f"| {gid} | {it['requirement']} | **{it['result']}** | {_md(it['detail'])} |")
        return "\n".join(rows)

    def probe_table(self) -> str:
        rows = ["| Probe | Request shape | HTTP | Outcome |", "| --- | --- | --- | --- |"]
        for p in self.probes:
            out = p["outcome"] + (f" ({p['notes']})" if p["notes"] and p["notes"] not in p["outcome"] else "")
            rows.append(f"| {p['probe']} | {_md(p['shape'])} | {p['http']} | {_md(out)} |")
        return "\n".join(rows)

    def fragment(self) -> str:
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        hg = harness_git()
        commit = (hg.get("commit") or "unknown")[:10] + (" (dirty)" if hg.get("dirty") else "")
        mode = " (quick smoke run)" if self.cfg.quick else ""
        tcm = ("; tool calls via **structured output** (`response_format` "
               f"`{self.cfg.profile.structured_format}`, {self.cfg.profile.structured_union}; non-standard)"
               if self.cfg.profile.tool_call_mode == "structured_output" else "")
        return (f"## Gate G1-G5 + G2b, tag `{self.cfg.tag}`{mode}\n\n"
                f"Run {now}; profile `{self.cfg.profile.name}`; framework version `{self.cfg.framework_version or 'n/a'}`; "
                f"harness `{commit}`{tcm}. Raw: `raw/{self.w.path.name}`.\n\n"
                f"{self.table()}\n\n### Probes\n\n{self.probe_table()}\n")


def _md(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _ctx_fields(js: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}.{k}" if path else k
                if k in ("max_model_len", "context_length", "n_ctx", "max_context_length", "max_context_window",
                         "context_window", "n_ctx_train", "max_position_embeddings") and isinstance(v, (int, float)):
                    out[p] = v
                else:
                    walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o[:8]):
                walk(v, f"{path}[{i}]")

    walk(js, "")
    return out


async def run(cfg: RunConfig, target: Target, writer: Writer, opts: Any = None) -> dict[str, Any]:
    g = Gate(cfg, target, writer)
    st, js = await target.get_json(cfg.profile.health_path)
    writer.write({"type": "health", "status": st, "body": js if not isinstance(js, str) else js[:300]})
    if st is None:
        raise RuntimeError(f"server at {cfg.base_url} is not reachable: {js}")
    for step in (g.g1, g.g2, g.g2b, g.g3, g.g4, g.g5):
        await step()
    frag = g.fragment()
    md = writer.path.with_suffix(".md")
    md.write_text(frag, encoding="utf-8")
    cap = cfg.fw_dir / ("raw/_unverified/capabilities.md" if cfg.unverified else "capabilities.md")
    upsert_section(cap, f"gate {cfg.tag}", frag, f"{cfg.framework or cfg.profile.framework}: capabilities")
    summary = {"type": "summary", "suite": "gate", "items": g.items, "probes": g.probes, "fragment": md.name}
    writer.write(summary)
    print(g.table())
    return summary
