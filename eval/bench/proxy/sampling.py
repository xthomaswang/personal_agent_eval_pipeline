"""Apply one sampling policy to every generation request.

Policies (``--sampling``):

* ``passthrough``: forward the client's sampling fields unchanged.
* ``strip``: remove the client's sampling fields, so the server defaults apply.
* ``inject:<preset>``: remove them and set a named preset instead.

What oMLX 0.7.1rc1 reads from a request (omlx/server.py, get_sampling_params
and its callers), ahead of the model's and the global settings unless
force_sampling is set: /v1/chat/completions uses temperature, top_p, top_k,
min_p, repetition_penalty, presence_penalty, frequency_penalty,
xtc_probability and xtc_threshold; /v1/messages uses temperature, top_p,
top_k and repetition_penalty. So every injected field takes effect. (oMLX
0.3.9 never read top_k or repetition_penalty from a request.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Request fields that control sampling. strip and inject remove all of them.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "typical_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "xtc_probability",
    "xtc_threshold",
    "seed",
)

PRESETS: dict[str, dict[str, Any]] = {
    # generation_config.json of Qwen/Qwen3.8-27B @ 1d4bf0f2 (the source of our oQ checkpoints), the same as
    # mlx-community/Qwen3.8-27B-8bit @ 815b83c0
    "qwen3.8-gen-config": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    # oMLX's global sampling defaults (~/.omlx/settings.json)
    "omlx-default": {"temperature": 1.0, "top_p": 0.95, "top_k": 0},
}

POLICIES_HELP = "passthrough | strip | inject:{" + ",".join(PRESETS) + "}"


@dataclass(frozen=True)
class SamplingPolicy:
    kind: str  # "passthrough" | "strip" | "inject"
    preset: str | None = None

    @classmethod
    def parse(cls, text: str) -> SamplingPolicy:
        value = text.strip().lower()
        if value in ("passthrough", "strip"):
            return cls(value)
        if value.startswith("inject:"):
            preset = value.split(":", 1)[1]
            if preset in PRESETS:
                return cls("inject", preset)
        raise ValueError(f"invalid sampling policy {text!r}; expected {POLICIES_HELP}")

    @property
    def values(self) -> dict[str, Any]:
        return dict(PRESETS[self.preset]) if self.kind == "inject" and self.preset else {}

    def __str__(self) -> str:
        return f"inject:{self.preset}" if self.kind == "inject" else self.kind


# The bench default, chosen by the user: the model's own generation config.
DEFAULT_SAMPLING = SamplingPolicy("inject", "qwen3.8-gen-config")
PASSTHROUGH_SAMPLING = SamplingPolicy("passthrough")


def sampling_view(body: dict[str, Any]) -> dict[str, Any]:
    """The sampling fields present in a request body."""
    return {key: body[key] for key in SAMPLING_KEYS if key in body}


def apply_sampling(body: dict[str, Any], policy: SamplingPolicy) -> None:
    """Rewrite the top-level sampling fields of ``body`` in place."""
    if policy.kind == "passthrough":
        return
    for key in SAMPLING_KEYS:
        body.pop(key, None)
    body.update(policy.values)
