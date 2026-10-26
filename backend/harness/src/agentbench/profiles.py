"""Per-framework request shaping (`--profile llamacpp|omlx|vllm|lithos|mlxlm`).

Shaping is the last pass over a request body, after `--extra-body` has been merged, and
returns notes that say what was changed and why. Rules come from <framework>/env.md and
resume.md ("Harness requirements found during framework research"):

- mlxlm: omit `model` (any other value reloads the model and drops the prompt cache);
  never send `seed` (it forces the unbatched path without segment checkpoints);
  reasoning arrives in `reasoning`.
- lithos: strict request schema: strip `chat_template_kwargs`, any `reasoning_effort`
  other than "none", and Anthropic `thinking` unless it is "disabled". Thinking cannot
  be enabled. Concurrent requests get HTTP 429. Cached tokens are only in the server
  log (`'cached_prompt_tokens': N`).
- vllm: reasoning arrives in `reasoning`.
- llamacpp: cached tokens come from `prompt_tokens_details` or `timings.cache_n`.
- mlc: no Qwen tool-call parser, so `tools` are never sent. Tool schemas are inlined into the
  system message, tool history is flattened, and the reply is constrained with
  `response_format` (xgrammar) and parsed back into tool calls (`structured.py`). Thinking
  is fixed by the served model config, so it is not requested per request; images are
  unsupported on the nightly path (metric suites default to --text-only).
- all: reasoning is accepted in both `reasoning` and `reasoning_content`, and every
  streaming request carries exactly `stream_options: {"include_usage": true}`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Profile:
    name: str
    framework: str  # default <framework> output directory
    port: int | None = None
    model_policy: str = "send"  # send | omit | default_model
    drop_seed: bool = False
    strip_chat_template_kwargs: bool = False
    allowed_efforts: tuple[str, ...] | None = None  # None: any value is passed through
    anthropic_thinking_allowed: tuple[str, ...] | None = None  # None: any type is passed through
    can_think: bool = True
    concurrency: str = "batch"  # batch | reject-429
    reasoning_field: str = "reasoning_content"  # informational: the parser accepts both fields
    cached_source: str = "usage"  # usage | usage-or-timings | server-log
    log_cached_regex: str | None = None
    health_path: str = "/health"
    probe_models: bool = True  # GET /v1/models during the gate (mlx-lm scans the whole HF cache)
    tool_call_mode: str = "tool_calls"  # tool_calls (native) | structured_output
    structured_format: str = "mlc"  # mlc ({"type": "json_object", "schema": str}) | openai (json_schema) | none
    structured_union: str = "anyOf"  # anyOf | oneOf (MLC's pinned xgrammar ignores oneOf)
    force_text_only: str | None = None  # reason, when metric suites must not send images
    string_content: bool = False  # strict schema: send an all-text content list as one string
    thinking_reason: str | None = None

    # ---------------------------------------------------------------- OpenAI
    def shape_openai(self, body: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        b = copy.deepcopy(body)
        notes: list[str] = []
        if b.get("stream"):
            if b.get("stream_options") != {"include_usage": True}:
                notes.append("stream_options set to exactly {include_usage: true}")
            b["stream_options"] = {"include_usage": True}
        self._model(b, notes)
        if self.string_content:
            self._flatten_text_content(b, notes)
        if self.drop_seed and "seed" in b:
            b.pop("seed")
            notes.append(f"{self.name}: dropped seed (forces the unbatched path without prompt-cache checkpoints)")
        if self.strip_chat_template_kwargs and "chat_template_kwargs" in b:
            b.pop("chat_template_kwargs")
            notes.append(f"{self.name}: stripped chat_template_kwargs (strict schema rejects it with HTTP 400)")
        eff = b.get("reasoning_effort")
        if self.allowed_efforts is not None and eff is not None and eff not in self.allowed_efforts:
            b.pop("reasoning_effort")
            notes.append(f"{self.name}: stripped reasoning_effort={eff!r} (only {list(self.allowed_efforts)} accepted)")
        if self.tool_call_mode == "structured_output":
            from .structured import to_structured_request
            b, more = to_structured_request(b, self.structured_format, self.structured_union)
            notes += more
        return b, notes

    # ------------------------------------------------------------- Anthropic
    def shape_anthropic(self, body: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        b = copy.deepcopy(body)
        notes: list[str] = []
        self._model(b, notes)
        th = b.get("thinking")
        if self.anthropic_thinking_allowed is not None and isinstance(th, dict) \
                and th.get("type") not in self.anthropic_thinking_allowed:
            b.pop("thinking")
            notes.append(f"{self.name}: stripped thinking={th.get('type')!r} (only {list(self.anthropic_thinking_allowed)} accepted)")
        return b, notes

    def _flatten_text_content(self, b: dict[str, Any], notes: list[str]) -> None:
        """Join content lists made only of text parts into one string, as the chat template would
        (Qwen's renders text parts back to back). Lists with an image part are left as they are, so
        the gate's image probes still show the server's own answer."""
        n = 0
        for m in b.get("messages") or []:
            c = m.get("content")
            if isinstance(c, list) and c and all(isinstance(x, dict) and x.get("type") == "text" for x in c):
                m["content"] = "".join(x.get("text", "") for x in c)
                n += 1
        if n:
            notes.append(f"{self.name}: sent {n} all-text content list(s) as strings (strict schema)")

    def _model(self, b: dict[str, Any], notes: list[str]) -> None:
        if self.model_policy == "omit" and "model" in b:
            b.pop("model")
            notes.append(f"{self.name}: omitted model (any other value reloads the model and drops the prompt cache)")
        elif self.model_policy == "default_model":
            b["model"] = "default_model"
        elif self.model_policy == "send" and not b.get("model"):
            b.pop("model", None)

    # ------------------------------------------------------------- semantics
    def thinking_blocked(self, thinking: bool | None, effort: str | None = None) -> str | None:
        """Reason why a request with this thinking setting cannot be expressed, or None."""
        if thinking and not self.can_think:
            return self.thinking_reason or (
                f"{self.name}: thinking cannot be enabled (chat_template_kwargs and reasoning_effort other than "
                "'none' are stripped; the server hard-wires thinking off)")
        return None

    def anthropic_thinking_blocked(self, thinking_type: str | None) -> str | None:
        if thinking_type and self.anthropic_thinking_allowed is not None and thinking_type not in self.anthropic_thinking_allowed:
            return f"{self.name}: Anthropic thinking={thinking_type!r} would be stripped (server accepts only absent or 'disabled')"
        return None


PROFILES: dict[str, Profile] = {
    "generic": Profile("generic", framework="generic"),
    "llamacpp": Profile("llamacpp", framework="llama.cpp", port=8702, cached_source="usage-or-timings"),
    "omlx": Profile("omlx", framework="omlx", port=8703),
    "lithos": Profile("lithos", framework="lithos-metal", port=8704, strip_chat_template_kwargs=True,
                      allowed_efforts=("none",), anthropic_thinking_allowed=("disabled",), can_think=False,
                      concurrency="reject-429", cached_source="server-log",
                      log_cached_regex=r"'cached_prompt_tokens':\s*(\d+)", string_content=True,
                      force_text_only="profile lithos: lithos-metal accepts only string message content; "
                                      "images are rejected (G2)"),
    "vllm": Profile("vllm", framework="vllm-metal", port=8705, reasoning_field="reasoning"),
    "mlxlm": Profile("mlxlm", framework="mlx-lm", port=8701, model_policy="omit", drop_seed=True,
                     reasoning_field="reasoning", probe_models=False,
                     force_text_only="profile mlxlm: mlx_lm.server accepts only text content parts; "
                                     "images are rejected (G2)"),
    "mlc": Profile("mlc", framework="mlc-llm", port=8700, strip_chat_template_kwargs=True, allowed_efforts=(),
                   can_think=False, tool_call_mode="structured_output", cached_source="usage-extra",
                   force_text_only="profile mlc: images are unsupported on the MLC nightly path",
                   thinking_reason=("mlc: no per-request thinking control (fixed by the served model config, "
                                    "qwen3_5 vs qwen3_5_nothink); thinking fields are stripped")),
}


def get_profile(name: str | None) -> Profile:
    if not name:
        return PROFILES["generic"]
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; choose one of {', '.join(p for p in PROFILES if p != 'generic')}")
    return PROFILES[name]
