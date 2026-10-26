"""Rewrite request bodies so every harness runs at the same thinking level.

The rewrite targets oMLX 0.3.9 (omlx/server.py, omlx/api/anthropic_models.py):

* ``/v1/chat/completions`` reads thinking only from ``chat_template_kwargs``
  (``enable_thinking``, ``reasoning_effort``). The top-level
  ``reasoning_effort`` that Hermes always sends is not a request field and is
  ignored.
* ``/v1/messages`` also accepts ``chat_template_kwargs`` (a field of
  ``MessagesRequest``). The server merges it first, then maps ``thinking.type``
  onto ``enable_thinking`` (enabled/adaptive -> true, disabled -> false).
  ``reasoning_effort`` from ``chat_template_kwargs`` survives that step, so the
  effort level is controllable through this endpoint as well.
  ``output_config.effort`` is ignored by the server.

In every mode other than ``passthrough`` the proxy makes
``chat_template_kwargs`` the single source of truth and removes the other
thinking controls, so a newer server cannot pick up a conflicting value:

* OpenAI: top-level ``reasoning_effort``, ``reasoning`` and
  ``thinking_budget`` are removed.
* Anthropic: ``thinking`` is replaced (any ``budget_tokens`` or ``display`` is
  dropped) and ``output_config.effort`` is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Values accepted by --thinking effort:<level>. Whether a level is valid is up
# to the model's chat template: Qwen3.8's template accepts only xhigh, medium
# and low, and rejects anything else while thinking is on.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh")

MODES_HELP = "passthrough | off | on | effort:{" + ",".join(EFFORT_LEVELS) + "}"

# Top-level keys of an OpenAI chat-completions body that control thinking.
OPENAI_THINKING_KEYS = ("chat_template_kwargs", "reasoning_effort", "reasoning", "thinking_budget")


@dataclass(frozen=True)
class ThinkingMode:
    """One value of --thinking."""

    kind: str  # "passthrough" | "off" | "on" | "effort"
    effort: str | None = None

    @classmethod
    def parse(cls, text: str) -> ThinkingMode:
        value = text.strip().lower()
        if value in ("passthrough", "off", "on"):
            return cls(value)
        if value.startswith("effort:"):
            level = value.split(":", 1)[1]
            if level in EFFORT_LEVELS:
                return cls("effort", level)
        raise ValueError(f"invalid thinking mode {text!r}; expected {MODES_HELP}")

    @property
    def rewrites(self) -> bool:
        return self.kind != "passthrough"

    def __str__(self) -> str:
        return f"effort:{self.effort}" if self.kind == "effort" else self.kind


PASSTHROUGH = ThinkingMode("passthrough")


def thinking_view(protocol: str | None, body: dict[str, Any]) -> dict[str, Any]:
    """Return the thinking-related settings present in a request body.

    Keys that are absent from the body are absent from the view, so ``{}``
    means "the client left everything to the server default".
    """
    view: dict[str, Any] = {}
    if protocol == "openai":
        for key in OPENAI_THINKING_KEYS:
            if key in body:
                view[key] = body[key]
    elif protocol == "anthropic":
        for key in ("thinking", "chat_template_kwargs"):
            if key in body:
                view[key] = body[key]
        output_config = body.get("output_config")
        if isinstance(output_config, dict) and "effort" in output_config:
            view["output_config"] = {"effort": output_config["effort"]}
    return view


def _template_kwargs(mode: ThinkingMode, current: Any) -> dict[str, Any]:
    """Build the chat_template_kwargs for a mode, keeping unrelated keys."""
    kwargs = dict(current) if isinstance(current, dict) else {}
    if mode.kind == "off":
        kwargs["enable_thinking"] = False
        kwargs.pop("reasoning_effort", None)
    elif mode.kind == "on":
        # "on" means the template's default effort (xhigh for Qwen3.8).
        kwargs["enable_thinking"] = True
        kwargs.pop("reasoning_effort", None)
    else:
        kwargs["enable_thinking"] = True
        kwargs["reasoning_effort"] = mode.effort
    return kwargs


def rewrite_thinking(protocol: str | None, body: dict[str, Any], mode: ThinkingMode) -> list[str]:
    """Rewrite ``body`` in place to ``mode`` and return warnings.

    Only top-level keys are replaced; nested objects of the original body are
    never mutated, so a shallow copy of the original is enough to keep it.
    """
    warnings: list[str] = []
    if not mode.rewrites or protocol not in ("openai", "anthropic"):
        return warnings

    current = body.get("chat_template_kwargs")
    if current is not None and not isinstance(current, dict):
        warnings.append("chat_template_kwargs was not an object and was replaced")
    body["chat_template_kwargs"] = _template_kwargs(mode, current)

    if protocol == "openai":
        for key in ("reasoning_effort", "reasoning", "thinking_budget"):
            body.pop(key, None)
    else:
        # enabled and adaptive both map to enable_thinking=true in oMLX 0.3.9.
        body["thinking"] = {"type": "disabled"} if mode.kind == "off" else {"type": "enabled"}
        output_config = body.get("output_config")
        if isinstance(output_config, dict) and "effort" in output_config:
            rest = {k: v for k, v in output_config.items() if k != "effort"}
            if rest:
                body["output_config"] = rest
            else:
                del body["output_config"]
    return warnings


def startup_warnings(mode: ThinkingMode) -> list[str]:
    """Warnings worth printing once when the proxy starts."""
    if mode.kind == "effort" and mode.effort == "high":
        return [
            "effort:high is not accepted by the Qwen3.8 chat template (it allows only "
            "xhigh, medium and low while thinking is on); oMLX answers such requests with "
            "HTTP 400 'Chat template error'. Use effort:high only with a model whose "
            "template accepts it."
        ]
    return []
