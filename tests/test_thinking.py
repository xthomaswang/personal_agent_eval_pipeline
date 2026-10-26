"""Unit tests for the thinking rewrite (pure functions, no servers)."""

from __future__ import annotations

import copy

import pytest

from bench.proxy.thinking import ThinkingMode, rewrite_thinking, startup_warnings, thinking_view

# What Hermes 0.22.2 sends: chat_template_kwargs from extra_body plus a
# top-level reasoning_effort ("none" when thinking is off, "medium" by default).
HERMES_BODY = {
    "model": "Qwen3.8-27B-8bit",
    "messages": [{"role": "user", "content": "hi"}],
    "reasoning_effort": "none",
    "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
}

# What Claude Code 2.1.302 sends by default.
CLAUDE_CODE_BODY = {
    "model": "Qwen3.8-27B-8bit",
    "max_tokens": 32000,
    "messages": [{"role": "user", "content": "hi"}],
    "thinking": {"type": "adaptive", "display": "omitted"},
    "output_config": {"effort": "high"},
    "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
}


def rewritten(protocol: str, body: dict, mode: str) -> dict:
    out = dict(body)
    assert rewrite_thinking(protocol, out, ThinkingMode.parse(mode)) == []
    return out


@pytest.mark.parametrize(
    "text,expected",
    [
        ("passthrough", "passthrough"),
        ("off", "off"),
        ("on", "on"),
        ("effort:low", "effort:low"),
        ("effort:medium", "effort:medium"),
        ("effort:high", "effort:high"),
        ("Effort:XHIGH", "effort:xhigh"),
    ],
)
def test_parse_accepts_documented_modes(text: str, expected: str) -> None:
    assert str(ThinkingMode.parse(text)) == expected


@pytest.mark.parametrize("text", ["", "auto", "effort", "effort:max", "effort:none", "disabled"])
def test_parse_rejects_other_values(text: str) -> None:
    with pytest.raises(ValueError):
        ThinkingMode.parse(text)


def test_passthrough_changes_nothing() -> None:
    for protocol, body in (("openai", HERMES_BODY), ("anthropic", CLAUDE_CODE_BODY)):
        assert rewritten(protocol, body, "passthrough") == body


@pytest.mark.parametrize(
    "mode,template_kwargs",
    [
        ("off", {"enable_thinking": False, "preserve_thinking": True}),
        ("on", {"enable_thinking": True, "preserve_thinking": True}),
        ("effort:low", {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"}),
        ("effort:xhigh", {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}),
    ],
)
def test_openai_rewrite(mode: str, template_kwargs: dict) -> None:
    body = dict(HERMES_BODY, thinking_budget=512, reasoning={"effort": "low"})
    out = rewritten("openai", body, mode)
    # chat_template_kwargs is the only thinking control left.
    assert thinking_view("openai", out) == {"chat_template_kwargs": template_kwargs}
    assert out["messages"] == body["messages"]


def test_openai_on_drops_client_effort_so_the_template_default_applies() -> None:
    body = dict(HERMES_BODY, chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"})
    out = rewritten("openai", body, "on")
    assert out["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in out


def test_openai_rewrite_adds_template_kwargs_when_missing() -> None:
    # localmelo sends no thinking controls at all.
    body = {"model": "m", "messages": []}
    assert rewritten("openai", body, "effort:medium")["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("off", {"thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": False}}),
        ("on", {"thinking": {"type": "enabled"}, "chat_template_kwargs": {"enable_thinking": True}}),
        (
            "effort:medium",
            {"thinking": {"type": "enabled"}, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}},
        ),
    ],
)
def test_anthropic_rewrite(mode: str, expected: dict) -> None:
    out = rewritten("anthropic", CLAUDE_CODE_BODY, mode)
    assert thinking_view("anthropic", out) == expected
    assert "output_config" not in out  # it held only the ignored effort
    assert out["context_management"] == CLAUDE_CODE_BODY["context_management"]
    assert out["max_tokens"] == 32000


def test_anthropic_rewrite_without_thinking_field() -> None:
    # Claude Code with MAX_THINKING_TOKENS=0 omits the thinking field.
    body = {k: v for k, v in CLAUDE_CODE_BODY.items() if k != "thinking"}
    assert rewritten("anthropic", body, "off")["thinking"] == {"type": "disabled"}
    assert "thinking" not in rewritten("anthropic", body, "passthrough")


def test_anthropic_rewrite_drops_budget_and_keeps_other_output_config() -> None:
    body = dict(
        CLAUDE_CODE_BODY,
        thinking={"type": "enabled", "budget_tokens": 4096},
        output_config={"effort": "low", "format": {"type": "json_schema"}},
    )
    out = rewritten("anthropic", body, "effort:low")
    assert out["thinking"] == {"type": "enabled"}
    assert out["output_config"] == {"format": {"type": "json_schema"}}


def test_rewrite_never_mutates_nested_objects_of_the_original() -> None:
    for protocol, body in (("openai", HERMES_BODY), ("anthropic", CLAUDE_CODE_BODY)):
        original = copy.deepcopy(body)
        shallow = dict(body)
        rewrite_thinking(protocol, shallow, ThinkingMode.parse("effort:low"))
        assert body == original


def test_non_object_template_kwargs_are_replaced_with_a_warning() -> None:
    body = {"model": "m", "messages": [], "chat_template_kwargs": "enable_thinking=false"}
    warnings = rewrite_thinking("openai", body, ThinkingMode.parse("off"))
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert warnings and "replaced" in warnings[0]


def test_views_hold_only_thinking_settings() -> None:
    assert thinking_view("openai", HERMES_BODY) == {
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
        "reasoning_effort": "none",
    }
    assert thinking_view("anthropic", CLAUDE_CODE_BODY) == {
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "high"},
    }
    assert thinking_view("openai", {"model": "m"}) == {}


def test_effort_high_warns_at_startup() -> None:
    assert startup_warnings(ThinkingMode.parse("effort:high"))
    for mode in ("passthrough", "off", "on", "effort:low", "effort:medium", "effort:xhigh"):
        assert startup_warnings(ThinkingMode.parse(mode)) == []
