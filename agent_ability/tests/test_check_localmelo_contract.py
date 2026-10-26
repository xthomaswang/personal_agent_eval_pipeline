"""Criterion 7 of the localmelo contract checker: the reflection retry of localmelo PR #31, and its
trigger under the reflection gate's `flags` setting (localmelo's gate PR)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_localmelo_contract.py"
_spec = importlib.util.spec_from_file_location("check_localmelo_contract", _SCRIPT)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)

SYSTEM = {"role": "system", "content": f"{contract.REFLECTION_START} Reflect on what happened."}
REQUEST = "This was attempt 0. The step budget for this attempt was exhausted. Reflect and decide."
TURNS = [
    {"role": "user", "content": "Fix transpose.py."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "shell_exec", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "1 failed"},
]
OBJECT = '{"recommended_action": "continue"}'
TOOL_CALL_TEXT = '<tool_call>\n=python_exec({"code": "x"})'  # E4: a tool call written as text


def first(call: int, reply: str | None = "", turns: list[dict] = TURNS) -> dict:
    return {"call": call, "kind": "first", "body": {"messages": [SYSTEM, *turns, {"role": "user", "content": REQUEST}]}, "reply": reply}


def retry(call: int, text: str | None = None, system: dict = SYSTEM) -> dict:
    if text is None:
        text = f"{contract.TRANSCRIPT_INTRO}\n\nUser: Fix transpose.py.\n\n{REQUEST}\n\n{contract.RETRY_NOTE}"
    return {"call": call, "kind": "retry", "body": {"messages": [system, {"role": "user", "content": text}]}, "reply": OBJECT}


def planning(call: int) -> dict:
    return {"call": call, "kind": None, "body": None, "reply": None}


@pytest.mark.parametrize(
    ("reply", "holds"),
    [
        (OBJECT, True),
        (f"```json\n{OBJECT}\n```", True),
        (f"My reflection: {OBJECT} That is all.", True),
        ('{"a": "{"} {"b": 2}', True),
        ("", False),
        ("continue", False),
        ('"done"', False),
        (f"[{OBJECT}]", False),  # JSON as a whole, but not an object
        ("{not json}", False),
        # The search goes on from where the decoder gave up, the end here, so the inner object is not read
        ('{"a": {"b": 1}', False),
    ],
)
def test_holds_reflection_object(reply: str, holds: bool) -> None:
    """The cases give the same answer as localmelo's _holds_reflection_object at 791d348 (PR #31)."""
    assert contract.holds_reflection_object(reply) is holds


def test_reflection_kind() -> None:
    assert contract.reflection_kind(first(1)["body"]) == "first"
    assert contract.reflection_kind(retry(2)["body"]) == "retry"
    assert contract.reflection_kind({"messages": [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "hi"}]}) is None


def test_a_reply_without_an_object_is_asked_again() -> None:
    assert contract.check_retries([planning(0), first(1), retry(2), planning(3)], expected=True, killed=False) == []


def test_a_missing_retry_fails_when_localmelo_retries() -> None:
    (problem,) = contract.check_retries([planning(0), first(1)], expected=True, killed=False)
    assert "no retry follows" in problem
    (problem,) = contract.check_retries([first(1), planning(2)], expected=True, killed=False)
    assert "call 2 is not its retry" in problem


def test_a_missing_retry_passes_before_pr31_unknown_or_killed() -> None:
    assert contract.check_retries([first(1)], expected=False, killed=False) == []
    assert contract.check_retries([first(1)], expected=None, killed=False) == []
    assert contract.check_retries([first(1)], expected=True, killed=True) == []


def test_a_reply_with_an_object_is_not_asked_again() -> None:
    assert contract.check_retries([first(1, reply=OBJECT), planning(2)], expected=True, killed=False) == []
    (problem,) = contract.check_retries([first(1, reply=OBJECT), retry(2)], expected=True, killed=False)
    assert "holds a JSON object" in problem


def test_a_retry_is_not_asked_again() -> None:
    problems = contract.check_retries([first(1), retry(2), retry(3)], expected=True, killed=False)
    assert problems == ["7: the retry at call 3 does not follow a reflection"]


def test_a_retry_from_a_localmelo_without_it_fails() -> None:
    (problem,) = contract.check_retries([first(1), retry(2)], expected=False, killed=False)
    assert "without the retry" in problem


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (retry(2, system={"role": "system", "content": f"{contract.REFLECTION_START} Other."}), "changes the system message"),
        (retry(2, text=f"{REQUEST}\n\n{contract.RETRY_NOTE}"), "does not open with the turns"),
        (retry(2, text=f"{contract.TRANSCRIPT_INTRO}\n\nUser: x\n\nAnother request.\n\n{contract.RETRY_NOTE}"), "does not end with"),
    ],
)
def test_the_retry_shape(item: dict, message: str) -> None:
    (problem,) = contract.check_retries([first(1), item], expected=True, killed=False)
    assert message in problem


def test_a_retry_after_no_turns_is_the_request_and_the_note() -> None:
    seq = [first(1, turns=[]), retry(2, text=f"{REQUEST}\n\n{contract.RETRY_NOTE}")]
    assert contract.check_retries(seq, expected=True, killed=False) == []


@pytest.mark.parametrize(
    ("reply", "holds"),
    [
        (OBJECT, True),
        (f"```json\n{OBJECT}\n```", True),
        (f"My reflection: {OBJECT} That is all.", True),
        (f"{TOOL_CALL_TEXT}\n{OBJECT}", True),  # the search goes on after an object without an action
        ('{"code": "x"} {"recommended_action": "done"}', True),
        (TOOL_CALL_TEXT, False),  # its only object holds the tool's arguments
        ('{"recommended_action": "proceed"}', False),  # not one of REFLECTION_ACTIONS
        ('{"recommended_action": "Continue"}', False),
        ('{"n": ' + '9' * 5000 + '} {"recommended_action": "continue"}', False),  # the number ends the search, as in localmelo
        (f"[{OBJECT}]", False),  # JSON as a whole, but not an object
        ('{"summary": "x"}', False),
        ("", False),
    ],
)
def test_holds_reflection_under_flags(reply: str, holds: bool) -> None:
    assert contract.holds_reflection(reply) is holds


def test_reflection_object_is_the_first_with_a_valid_action() -> None:
    assert contract.reflection_object(f"{TOOL_CALL_TEXT}\n{OBJECT}") == {"recommended_action": "continue"}
    assert contract.holds_reflection_object(TOOL_CALL_TEXT)  # any object: not retried before the gate PR


def test_flags_asks_again_after_a_tool_call_written_as_text() -> None:
    seq = [first(1, reply=TOOL_CALL_TEXT), retry(2)]
    assert contract.check_retries(seq, expected=True, killed=False, gate="flags") == []
    for gate in ("utility", None):
        (problem,) = contract.check_retries(seq, expected=True, killed=False, gate=gate)
        assert "holds a JSON object" in problem


def test_flags_expects_the_retry_after_a_reply_without_a_reflection() -> None:
    seq = [first(1, reply=TOOL_CALL_TEXT), planning(2)]
    (problem,) = contract.check_retries(seq, expected=True, killed=False, gate="flags")
    assert "holds no reflection, and call 2 is not its retry" in problem
    assert contract.check_retries(seq, expected=True, killed=False, gate="utility") == []
    (problem,) = contract.check_retries([first(1, reply=OBJECT), retry(2)], expected=True, killed=False, gate="flags")
    assert "holds a reflection" in problem


def test_the_gate_setting_comes_from_the_line_the_flag_or_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    line = {"config": {"reflection_gate": "utility"}}
    assert contract.reflection_gate(line, {"harness_args": ["--reflection-gate", "flags"]}, None) == "utility"
    args = {"harness_args": ["--reflection-gate", "utility", "--output-format", "json"]}
    assert contract.reflection_gate({}, args, None) == "utility"  # a killed run: no JSON line
    assert contract.reflection_gate({}, {"harness_args": ["--reflection-gate=flags"]}, None) == "flags"
    localmelo = {"path": "/repo", "commit": "abc"}
    monkeypatch.setattr(contract, "_defines", lambda *a: True)
    assert contract.reflection_gate({}, {"harness_args": ["--output-format", "json"]}, localmelo) == "flags"
    monkeypatch.setattr(contract, "_defines", lambda *a: False)
    assert contract.reflection_gate({}, {}, localmelo) is None
    assert contract.reflection_gate({}, {}, None) is None
