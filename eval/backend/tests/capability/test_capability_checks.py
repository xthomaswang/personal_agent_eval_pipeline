"""calls.jsonl checks and the verdict mapping (DESIGN §6.4), on synthetic proxy records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modeleval.capability import checks

SERVED = "Nanbeige4.3-33B-oQ5"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 0}
OFF_OPENAI = {"chat_template_kwargs": {"enable_thinking": False}}
OFF_ANTHROPIC = {"thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": False}}


def call(**kw: Any) -> dict[str, Any]:
    base = {
        "call_index": 0, "protocol": "openai", "endpoint": "chat_completions", "role": "main", "model": SERVED,
        "status": 200, "error": None, "handled_locally": False, "budget_exhausted": False,
        "thinking_forwarded": OFF_OPENAI, "sampling_forwarded": dict(SAMPLING), "output_reasoning_chars": 0,
        "duration_s": 2.5,
    }  # fmt: skip
    return {**base, **kw}


def run(records: list[dict[str, Any]], thinking: str = "off") -> dict[str, Any]:
    return checks.check_calls(records, served_id=SERVED, sampling=SAMPLING, thinking=thinking)


def test_clean_calls() -> None:
    anthropic = call(protocol="anthropic", endpoint="messages", thinking_forwarded=OFF_ANTHROPIC)
    other = call(endpoint="other", protocol=None, model=None, thinking_forwarded=None, sampling_forwarded=None,
                 status=404, duration_s=0.01)  # fmt: skip
    result = run([call(), anthropic, other])
    assert result["generation_calls"] == 2 and result["model_s"] == 5.0
    assert not any(result[k] for k in ("model_mismatch", "sampling_mismatch", "thinking_mismatch", "http_5xx",
                                       "reasoning_leak", "upstream_error"))  # fmt: skip
    assert checks.invalid_reasons(result) == []


@pytest.mark.parametrize(
    ("record", "kind"),
    [
        (call(model="Qwen3.8-27B-oQ5-mtp"), "model_mismatch"),
        (call(model=None), "model_mismatch"),  # a successful call to an unknown model
        (call(sampling_forwarded={"temperature": 1.0, "top_p": 0.95, "top_k": 20}), "sampling_mismatch"),
        (call(sampling_forwarded=None), "sampling_mismatch"),
        (call(thinking_forwarded={"chat_template_kwargs": {"enable_thinking": True}}), "thinking_mismatch"),
        (call(thinking_forwarded={**OFF_OPENAI, "reasoning_effort": "none"}), "thinking_mismatch"),
        (call(thinking_forwarded={"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}),
         "thinking_mismatch"),
        (call(protocol="anthropic", endpoint="messages", thinking_forwarded={"thinking": {"type": "adaptive"},
              "chat_template_kwargs": {"enable_thinking": False}}), "thinking_mismatch"),
        (call(protocol="anthropic", endpoint="count_tokens", thinking_forwarded={**OFF_ANTHROPIC,
              "output_config": {"effort": "high"}}), "thinking_mismatch"),
        (call(status=500), "http_5xx"),
        (call(status=502, error={"source": "proxy", "type": "upstream_unreachable"}), "upstream_error"),
        (call(error={"source": "upstream", "type": "incomplete_stream"}), "upstream_error"),
        (call(output_reasoning_chars=120), "reasoning_leak"),
    ],
)  # fmt: skip
def test_each_finding(record: dict[str, Any], kind: str) -> None:
    result = run([record])
    assert result[kind] == 1, result
    assert any(kind in example for example in result["examples"])


def test_budget_refusals_and_local_answers_are_not_checked() -> None:
    refused = call(status=400, budget_exhausted=True, handled_locally=True, sampling_forwarded=None,
                   error={"source": "proxy", "type": "bench_budget_exhausted"})  # fmt: skip
    hello = call(endpoint="hello", protocol=None, handled_locally=True, model=None)
    result = run([refused, hello])
    assert result["budget_refusals"] == 1 and result["sampling_mismatch"] == 0 and result["model_mismatch"] == 0


def test_thinking_on_expects_enabled_and_allows_reasoning() -> None:
    on = call(thinking_forwarded={"chat_template_kwargs": {"enable_thinking": True}}, output_reasoning_chars=500)
    assert run([on], thinking="on")["thinking_mismatch"] == 0
    assert run([on], thinking="on")["reasoning_leak"] == 0
    anthropic_on = call(protocol="anthropic", endpoint="messages",
                        thinking_forwarded={"thinking": {"type": "enabled"}, "chat_template_kwargs": {"enable_thinking": True}})
    assert run([anthropic_on], thinking="on")["thinking_mismatch"] == 0  # fmt: skip


def test_load_calls_tolerates_a_truncated_last_line(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps(call()) + "\n" + '{"call_index": 1, "mod')
    records, bad = checks.load_calls(path)
    assert len(records) == 1 and bad == 1
    assert checks.load_calls(tmp_path / "absent.jsonl") == ([], 0)


def smoke_record(harness: str = "hermes", verdict: str = "pass", passed: bool = True, **kw: Any) -> dict[str, Any]:
    base = {"harness": harness, "verdict": verdict, "check": {"passed": passed}, "n_llm": 5, "wall_s": 100.0,
            "timed_out": verdict == "timeout", "harness_report": {}}  # fmt: skip
    return {**base, **kw}


CLEAN = run([call()])


def test_verdicts() -> None:
    out = checks.outcome(smoke_record(), CLEAN, max_steps=30)
    assert (out["verdict"], out["valid"], out["solved"], out["solved_within_1200"]) == ("pass", True, True, True)
    out = checks.outcome(smoke_record(verdict="fail", passed=False), CLEAN, max_steps=30)
    assert (out["verdict"], out["solved"]) == ("fail", False)


def test_a_wall_timeout_whose_check_passes_is_solved_but_not_within_1200() -> None:
    out = checks.outcome(smoke_record(verdict="timeout", passed=True, wall_s=3600.0), CLEAN, max_steps=30)
    assert (out["verdict"], out["valid"], out["solved"], out["solved_within_1200"]) == ("timeout", True, True, False)
    out = checks.outcome(smoke_record(verdict="timeout", passed=False, wall_s=3600.0), CLEAN, max_steps=30)
    assert (out["verdict"], out["solved"]) == ("timeout", False)


def test_localmelos_own_timeout_maps_to_the_timeout_verdict() -> None:
    report = {"run_record": {"status": "timeout", "steps": 4}}
    out = checks.outcome(smoke_record("localmelo", "fail", False, harness_report=report), CLEAN, max_steps=30)
    assert (out["verdict"], out["localmelo_status"], out["solved"]) == ("timeout", "timeout", False)
    out = checks.outcome(smoke_record("localmelo", "pass", True, harness_report=report), CLEAN, max_steps=30)
    assert (out["verdict"], out["solved"]) == ("timeout", True)
    # Only localmelo's record means localmelo's own stop
    out = checks.outcome(smoke_record("hermes", "fail", False, harness_report=report), CLEAN, max_steps=30)
    assert out["verdict"] == "fail"


def test_invalid_and_server_lost_are_not_scored() -> None:
    mismatch = run([call(model="Qwen3.8-27B-oQ5-mtp", sampling_forwarded={})])
    out = checks.outcome(smoke_record(), mismatch, max_steps=30)
    assert (out["verdict"], out["valid"], out["solved"], out["invalid_reasons"]) == ("invalid", False, False, ["model", "sampling"])
    out = checks.outcome(smoke_record(), CLEAN, max_steps=30, server_lost=True)
    assert (out["verdict"], out["valid"], out["solved"]) == ("server_lost", False, False)


def test_a_harness_error_before_any_model_call_is_the_pipelines() -> None:
    out = checks.outcome(smoke_record(verdict="harness_error", passed=False, n_llm=0, error="RuntimeError: proxy"),
                         run([]), max_steps=30)  # fmt: skip
    assert (out["verdict"], out["valid"], out["invalid_reasons"]) == ("invalid", False, ["no_model_call"])
    hermes_probe_only = run([call(endpoint="other", protocol=None, model=None, status=404)])
    out = checks.outcome(smoke_record(verdict="harness_error", passed=False, n_llm=0), hermes_probe_only, max_steps=30)
    assert out["verdict"] == "invalid"
    out = checks.outcome(smoke_record(verdict="harness_error", passed=False, n_llm=0), run([call(status=400)]), max_steps=30)
    assert (out["verdict"], out["valid"]) == ("harness_error", True)  # the model was called: its failure


def test_model_faults_are_flagged_not_invalid() -> None:
    faults = run([call(status=500), call(output_reasoning_chars=10), call(error={"source": "upstream", "type": "x"})])
    out = checks.outcome(smoke_record(verdict="harness_error", passed=False), faults, max_steps=30, template_errors=2)
    assert out["valid"] and out["verdict"] == "harness_error"
    assert out["anomalies"] == ["http_5xx", "reasoning_leak", "upstream_error", "template_error", "harness_error"]


def test_over_budget_is_never_solved() -> None:
    out = checks.outcome(smoke_record(n_llm=31), CLEAN, max_steps=30)
    assert out["valid"] and not out["solved"]


def budget_refusal() -> dict[str, Any]:
    """What the proxy records for a main call beyond --max-steps: HTTP 400 bench_budget_exhausted, never sent upstream."""
    return call(status=400, budget_exhausted=True, handled_locally=True, sampling_forwarded=None,
                error={"source": "proxy", "type": "bench_budget_exhausted"})  # fmt: skip


def test_budget_exhausted_is_true_for_a_localmelo_provider_error_and_localmelo_keeps_its_word() -> None:
    """localmelo calls any HTTP error `provider_error`, a budget refusal included. The row says plainly that the
    step budget was used up, and localmelo_status stays localmelo's own status."""
    report = {"run_record": {"status": "provider_error", "steps": 29}}
    record = smoke_record("localmelo", "fail", False, n_llm=30, harness_report=report)
    refused = run([call(), budget_refusal()])
    assert refused["budget_refusals"] == 1
    out = checks.outcome(record, refused, max_steps=30)
    assert out["budget_exhausted"] is True
    assert (out["verdict"], out["localmelo_status"], out["valid"], out["solved"]) == ("fail", "provider_error", True, False)
    assert out["anomalies"] == [] and out["invalid_reasons"] == []  # a refusal is neither a fault nor a mismatch
    # The same record when localmelo ends its run for another reason, and no call was refused
    other = {"run_record": {"status": "completed", "steps": 12}}
    out = checks.outcome(smoke_record("localmelo", "fail", False, harness_report=other), CLEAN, max_steps=30)
    assert (out["budget_exhausted"], out["localmelo_status"]) == (False, "completed")


@pytest.mark.parametrize("harness", ["claude_code", "hermes", "localmelo"])
def test_budget_exhausted_is_the_same_for_every_harness_and_changes_no_other_field(harness: str) -> None:
    record = smoke_record(harness, "fail", False, n_llm=30)
    with_refusals = checks.outcome(record, run([call(), budget_refusal(), budget_refusal()]), max_steps=30)
    without = checks.outcome(record, CLEAN, max_steps=30)
    assert with_refusals["budget_exhausted"] is True and without["budget_exhausted"] is False
    assert {**with_refusals, "budget_exhausted": False} == without
    if harness != "localmelo":
        assert with_refusals["localmelo_status"] is None  # localmelo's own status exists for localmelo only


def test_budget_exhausted_is_not_set_by_other_failed_calls() -> None:
    bad_request = call(status=400, error={"source": "proxy", "type": "invalid_request"})
    out = checks.outcome(smoke_record(), run([call(), bad_request]), max_steps=30)
    assert out["budget_exhausted"] is False


def test_a_budget_refusal_does_not_change_the_verdict_or_the_score() -> None:
    """A run can use its 30 steps, be refused the 31st call and still leave a workspace that passes."""
    out = checks.outcome(smoke_record(n_llm=30), run([call(), budget_refusal()]), max_steps=30)
    assert (out["verdict"], out["valid"], out["solved"], out["budget_exhausted"]) == ("pass", True, True, True)


def test_native_turns() -> None:
    assert checks.native_turns("claude_code", {"num_turns": 7}) == 7
    assert checks.native_turns("hermes", {"api_call_count": 4}) == 4
    assert checks.native_turns("localmelo", {"steps": None}) is None


def test_endpoint_probes_are_not_anomalies():
    """Hermes probes /api/v1/models, /api/tags, /v1/props, /props and /version at every start; oMLX
    answers them 404 (seen on every Hermes run of 2026-10-26). They say nothing about the model."""
    probes = [call(method="GET", path=p, endpoint="other", role=None, status=404,
                   error={"source": "upstream", "status": 404, "message": "Not Found"})
              for p in ("/api/v1/models", "/api/tags", "/v1/props", "/props", "/version")]
    out = run(probes + [call()])
    assert out["upstream_error"] == 0 and out["http_5xx"] == 0 and out["generation_calls"] == 1
