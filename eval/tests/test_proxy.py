"""End-to-end tests: client -> proxy -> fake oMLX, all over real sockets on port 0."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from bench.proxy import ThinkingMode
from bench.proxy.thinking import thinking_view
from tests.support import ERROR_RESPONSES, MODEL, FakeUpstream, ProxyHandle, free_port

WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
}
READ_TOOL = {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}
BASH_TOOL = {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}
SECRET = "the-secret-prompt-4711"


def chat_body(**extra: object) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is the weather in Paris?"},
        ],
        "tools": [WEATHER_TOOL],
    }
    body.update(extra)
    return body


def messages_body(**extra: object) -> dict:
    body = {
        "model": MODEL,
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "Read /tmp/a"}],
        "tools": [READ_TOOL, BASH_TOOL],
    }
    body.update(extra)
    return body


def post(proxy: ProxyHandle, path: str, body: dict | bytes, **kwargs: object) -> httpx.Response:
    headers = {"content-type": "application/json", **kwargs.pop("headers", {})}  # type: ignore[dict-item]
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.post(proxy.url + path, content=content, headers=headers, timeout=10, **kwargs)  # type: ignore[arg-type]


def stream(proxy: ProxyHandle, path: str, body: dict, headers: dict | None = None) -> httpx.Response:
    with httpx.stream("POST", proxy.url + path, json=body, headers=headers or {}, timeout=10) as response:
        response.read()
    return response


# -- OpenAI ------------------------------------------------------------------


def test_openai_non_streaming_call_is_recorded(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    raw = json.dumps(chat_body()).encode()
    response = post(proxy, "/v1/chat/completions", raw)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert response.headers["x-bench-call-index"] == "0"
    # passthrough and no stream: the upstream gets the client's exact bytes
    assert upstream.requests[-1]["body"] == raw

    rec = proxy.record()
    assert rec["schema_version"] == 2
    assert (rec["run_id"], rec["run_id_source"], rec["call_index"], rec["harness"]) == ("test-run", "cli", 0, "pytest")
    assert (rec["protocol"], rec["endpoint"], rec["method"], rec["path"]) == ("openai", "chat_completions", "POST", "/v1/chat/completions")
    assert rec["model"] == MODEL and rec["stream"] is False
    assert rec["request_bytes"] == rec["forwarded_bytes"] == len(raw)
    assert (rec["messages"], rec["images"], rec["tools"]) == (2, 0, 1)
    assert rec["status"] == 200 and rec["error"] is None
    assert rec["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "cached_tokens": 3}
    assert rec["tool_calls"] == 1 and rec["tool_names"] == ["get_weather"]
    assert rec["stop_reason"] == "tool_calls"
    assert (rec["output_text_chars"], rec["output_reasoning_chars"]) == (len("Hello world"), len("Thinking about it."))
    assert rec["response_bytes"] == len(response.content)
    assert rec["ts_start"] <= rec["ts_first_byte"] <= rec["ts_end"]
    assert rec["mono_start"] <= rec["mono_first_byte"] <= rec["mono_end"]
    assert rec["duration_s"] >= rec["ttfb_s"] >= 0
    assert rec["ttft_s"] is None  # token timing needs a stream
    assert rec["bodies"] is None


def test_openai_stream_records_usage_tools_and_timing(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    response = stream(proxy, "/v1/chat/completions", chat_body(stream=True))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # the client gets the upstream's bytes unchanged, usage chunk included
    assert response.content == bytes(upstream.sent)
    assert b'"choices": [], "usage"' in response.content

    sent = upstream.requests[-1]["json"]
    assert sent["stream_options"] == {"include_usage": True}

    rec = proxy.record()
    assert rec["stream"] is True and rec["usage_injected"] is True
    assert rec["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "cached_tokens": 3}
    assert rec["tool_calls"] == 2 and rec["tool_names"] == ["get_weather", "get_time"]
    assert rec["stop_reason"] == "tool_calls"
    assert rec["sse_keepalives"] == 1 and rec["stream_complete"] is True
    assert rec["error"] is None and rec["client_disconnected"] is False
    assert rec["mono_first_byte"] <= rec["mono_first_output"] <= rec["mono_first_content"] <= rec["mono_end"]
    assert rec["ts_first_byte"] <= rec["ts_first_output"] <= rec["ts_first_content"] <= rec["ts_end"]
    assert rec["ttft_s"] is not None and rec["ttft_s"] >= rec["ttfb_s"]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_stream_chunks_are_relayed_without_buffering(
    upstream: FakeUpstream, proxy: ProxyHandle, protocol: str
) -> None:
    """The fake upstream sends part i, then waits until the client has part i.

    A buffering proxy would hold part 0 back, the upstream would wait in vain
    and record a gate timeout.
    """
    path, body = (
        ("/v1/chat/completions", chat_body(stream=True)) if protocol == "openai" else ("/v1/messages", messages_body(stream=True))
    )
    received = bytearray()
    with httpx.Client(timeout=10) as client:
        with client.stream("POST", proxy.url + path, json=body, headers={"x-fake-scenario": "gated"}) as response:
            assert response.status_code == 200
            chunks = response.iter_raw()
            for i in range(3):
                while f"part-{i}".encode() not in received:
                    received += next(chunks)
                if i < 2:
                    upstream.gates[i].set()  # let the upstream send the next part
            for chunk in chunks:
                received += chunk
    assert upstream.gate_timeouts == 0
    assert bytes(received) == bytes(upstream.sent)
    rec = proxy.record()
    assert rec["stream_complete"] is True and rec["error"] is None
    assert rec["output_text_chars"] == len("part-0part-1part-2")
    assert rec["mono_first_byte"] <= rec["mono_first_output"] <= rec["mono_end"]


def test_openai_sdk_handles_the_injected_usage_chunk(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=f"{proxy.url}/v1", api_key="unused", max_retries=0, timeout=10)
    try:
        chunks = list(
            client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}], stream=True)
        )
    finally:
        client.close()
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None and chunks[-1].usage.prompt_tokens == 11
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello world"
    # The sync SDK closes the connection right after [DONE] without reading to
    # the end, so the record can land just after the client is done. Such an
    # early close of a complete stream is not an error.
    rec = proxy.wait_for_record(lambda r: True)
    assert rec["usage_injected"] is True and rec["stream_complete"] is True
    assert rec["error"] is None and rec["usage"]["prompt_tokens"] == 11


def test_usage_injection_can_be_disabled(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(inject_stream_usage=False)
    raw = json.dumps(chat_body(stream=True)).encode()
    with httpx.stream("POST", proxy.url + "/v1/chat/completions", content=raw, timeout=10) as response:
        response.read()
    assert upstream.requests[-1]["body"] == raw
    rec = proxy.record()
    assert rec["usage_injected"] is False and rec["usage"] is None
    assert rec["tool_names"] == ["get_weather", "get_time"] and rec["stream_complete"] is True


def test_usage_requested_by_the_client_is_not_injected(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    # Hermes 0.22.2 asks for usage itself on every stream.
    stream(proxy, "/v1/chat/completions", chat_body(stream=True, stream_options={"include_usage": True}))
    rec = proxy.record()
    assert rec["usage_injected"] is False and rec["usage"]["prompt_tokens"] == 11


# -- Anthropic ---------------------------------------------------------------


def test_anthropic_non_streaming_call_is_recorded(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    headers = {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "claude-code-20250219, context-management-2025-06-27",
        "x-api-key": "sk-test-key",
    }
    response = post(proxy, "/v1/messages?beta=true", messages_body(), headers=headers)
    assert response.status_code == 200
    assert response.json()["content"][2]["name"] == "Read"

    sent = upstream.requests[-1]
    assert (sent["path"], sent["query"]) == ("/v1/messages", "beta=true")
    assert sent["headers"]["anthropic-version"] == "2023-06-01"

    rec = proxy.record()
    assert (rec["protocol"], rec["endpoint"], rec["query"]) == ("anthropic", "messages", "beta=true")
    assert (rec["messages"], rec["tools"], rec["max_tokens"]) == (1, 2, 32000)
    assert rec["anthropic_beta"] == ["claude-code-20250219", "context-management-2025-06-27"]
    assert rec["usage"] == {"prompt_tokens": 12, "completion_tokens": 9, "cached_tokens": 4}
    assert rec["tool_names"] == ["Read"] and rec["stop_reason"] == "tool_use"
    assert (rec["output_text_chars"], rec["output_reasoning_chars"]) == (len("Reading."), len("Hmm."))
    assert "sk-test-key" not in proxy.calls_path.read_text()


def test_anthropic_stream_records_usage_and_tool_use_names(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    response = stream(proxy, "/v1/messages", messages_body(stream=True))
    assert response.status_code == 200
    assert response.content == bytes(upstream.sent)
    rec = proxy.record()
    assert rec["stream"] is True and rec["usage_injected"] is False
    # final counts from message_delta replace message_start's estimate
    assert rec["usage"] == {"prompt_tokens": 12, "completion_tokens": 9, "cached_tokens": 4}
    assert rec["usage_raw"]["cache_creation_input_tokens"] == 8
    assert rec["tool_calls"] == 2 and rec["tool_names"] == ["Read", "Bash"]
    assert rec["stop_reason"] == "tool_use"
    assert rec["sse_keepalives"] == 1 and rec["stream_complete"] is True
    assert rec["mono_first_output"] <= rec["mono_first_content"]


def test_count_tokens_is_relayed(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(thinking=ThinkingMode.parse("off"))
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "adaptive"}}
    response = post(proxy, "/v1/messages/count_tokens", body)
    assert response.json() == {"input_tokens": 42}
    assert upstream.requests[-1]["json"]["thinking"] == {"type": "disabled"}
    rec = proxy.record()
    assert rec["endpoint"] == "count_tokens" and rec["counted_input_tokens"] == 42
    assert rec["usage"] is None and rec["thinking_rewritten"] is True


# -- request summary ---------------------------------------------------------


def test_images_are_counted_in_both_formats(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    png = {"url": "data:image/png;base64,iVBORw0KGgo="}
    openai = chat_body(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "compare"}, {"type": "image_url", "image_url": png}, {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image_url", "image_url": png}]},
        ]
    )
    source = {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}
    anthropic = messages_body(
        messages=[
            {"role": "user", "content": [{"type": "image", "source": source}, {"type": "text", "text": "what is this"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/tmp/shot.png"}}]},
            {
                "role": "user",
                "content": [
                    # Claude Code returns screenshots inside tool results
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "image", "source": source}, {"type": "text", "text": "shot"}]},
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
                ],
            },
        ]
    )
    post(proxy, "/v1/chat/completions", openai)
    post(proxy, "/v1/messages", anthropic)
    first, second = proxy.records()
    assert (first["messages"], first["images"]) == (3, 3)
    assert (second["messages"], second["images"]) == (3, 2)


# -- thinking ------------------------------------------------------------------

# Hermes 0.22.2 with thinking off in its config.
HERMES_THINKING = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("passthrough", HERMES_THINKING),
        ("off", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("on", {"chat_template_kwargs": {"enable_thinking": True}}),
        ("effort:low", {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}),
        ("effort:medium", {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}}),
        ("effort:xhigh", {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}),
    ],
)
def test_openai_thinking_rewrite(
    upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle], mode: str, expected: dict
) -> None:
    proxy = make_proxy(thinking=ThinkingMode.parse(mode))
    body = chat_body(**HERMES_THINKING)
    post(proxy, "/v1/chat/completions", body)
    sent = upstream.requests[-1]["json"]
    assert thinking_view("openai", sent) == expected
    assert sent["messages"] == body["messages"] and sent["tools"] == body["tools"]
    rec = proxy.record()
    assert rec["thinking_mode"] == mode
    assert rec["thinking_original"] == HERMES_THINKING
    assert rec["thinking_forwarded"] == expected
    assert rec["thinking_rewritten"] is (mode != "passthrough")
    if mode != "passthrough":
        assert rec["forwarded_bytes"] == len(upstream.requests[-1]["body"]) != rec["request_bytes"]


# Claude Code 2.1.302 defaults.
CLAUDE_CODE_THINKING = {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}}


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("passthrough", CLAUDE_CODE_THINKING),
        ("off", {"thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": False}}),
        ("on", {"thinking": {"type": "enabled"}, "chat_template_kwargs": {"enable_thinking": True}}),
        (
            "effort:low",
            {"thinking": {"type": "enabled"}, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}},
        ),
    ],
)
def test_anthropic_thinking_rewrite(
    upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle], mode: str, expected: dict
) -> None:
    proxy = make_proxy(thinking=ThinkingMode.parse(mode))
    context = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
    post(proxy, "/v1/messages", messages_body(context_management=context, **CLAUDE_CODE_THINKING))
    sent = upstream.requests[-1]["json"]
    assert thinking_view("anthropic", sent) == expected
    assert sent["context_management"] == context and sent["max_tokens"] == 32000
    rec = proxy.record()
    assert rec["thinking_original"] == CLAUDE_CODE_THINKING
    assert rec["thinking_forwarded"] == expected
    assert rec["thinking_rewritten"] is (mode != "passthrough")


def test_thinking_rewrite_applies_when_the_client_sends_no_controls(
    upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]
) -> None:
    # localmelo direct mode and Claude Code with MAX_THINKING_TOKENS=0
    proxy = make_proxy(thinking=ThinkingMode.parse("off"))
    post(proxy, "/v1/chat/completions", chat_body())
    post(proxy, "/v1/messages", messages_body())
    openai_sent, anthropic_sent = (r["json"] for r in upstream.requests[-2:])
    assert openai_sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert anthropic_sent["thinking"] == {"type": "disabled"}
    assert [r["thinking_original"] for r in proxy.records()] == [{}, {}]


def test_non_json_body_is_relayed_unchanged(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(thinking=ThinkingMode.parse("off"))
    response = post(proxy, "/v1/chat/completions", b"not json")
    assert response.status_code == 422
    assert upstream.requests[-1]["body"] == b"not json"
    rec = proxy.record()
    assert rec["warnings"] == ["request body is not JSON; forwarded unchanged"]
    assert rec["error"]["type"] == "validation_error"


# -- identity --------------------------------------------------------------------


def test_run_id_header_overrides_the_cli_value(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    post(proxy, "/v1/chat/completions", chat_body(), headers={"X-Bench-Run-Id": "run-from-header"})
    post(proxy, "/v1/chat/completions", chat_body())
    first, second = proxy.records()
    assert (first["run_id"], first["run_id_source"], first["call_index"]) == ("run-from-header", "header", 0)
    assert (second["run_id"], second["run_id_source"], second["call_index"]) == ("test-run", "cli", 1)
    assert "x-bench-run-id" not in upstream.requests[0]["headers"]


# -- errors ----------------------------------------------------------------------


@pytest.mark.parametrize("stream_flag", [False, True])
@pytest.mark.parametrize("scenario", sorted(ERROR_RESPONSES))
def test_upstream_errors_are_passed_through(
    upstream: FakeUpstream, proxy: ProxyHandle, scenario: str, stream_flag: bool
) -> None:
    status, payload = ERROR_RESPONSES[scenario]
    response = post(proxy, "/v1/chat/completions", chat_body(stream=stream_flag), headers={"x-fake-scenario": scenario})
    assert response.status_code == status
    assert response.json() == payload
    rec = proxy.record()
    assert rec["status"] == status
    assert rec["error"]["source"] == "upstream" and rec["error"]["status"] == status
    if scenario == "error-422":
        assert rec["error"]["details"] == [{"type": "missing", "loc": ["body", "max_tokens"], "msg": "Field required"}]
        assert "TOP-SECRET-PROMPT" not in proxy.calls_path.read_text()
    else:
        assert rec["error"]["message"] == payload["detail"]


def test_anthropic_error_is_passed_through(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    response = post(proxy, "/v1/messages", messages_body(stream=True), headers={"x-fake-scenario": "error-400"})
    assert response.status_code == 400
    assert proxy.record()["error"]["message"].startswith("Chat template error")


def test_unreachable_upstream_gives_502(make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(upstream=f"http://127.0.0.1:{free_port()}")
    response = post(proxy, "/v1/chat/completions", chat_body())
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unreachable"
    response = post(proxy, "/v1/messages", messages_body())
    assert response.status_code == 502
    assert response.json()["type"] == "error"  # Anthropic error shape
    first, second = proxy.records()
    assert first["error"]["source"] == "proxy" and first["error"]["type"] == "upstream_unreachable"
    assert first["status"] == 502 and second["status"] == 502


def test_upstream_failure_mid_stream_reaches_the_client(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    with pytest.raises((httpx.RemoteProtocolError, httpx.ReadError)):
        stream(proxy, "/v1/chat/completions", chat_body(stream=True), headers={"x-fake-scenario": "abort"})
    rec = proxy.record()
    assert rec["status"] == 200
    assert rec["error"]["source"] == "upstream"
    assert rec["stream_complete"] is False


def test_client_disconnect_closes_the_upstream_stream(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    received = b""
    with httpx.Client(timeout=10) as client:
        with client.stream(
            "POST", proxy.url + "/v1/chat/completions", json=chat_body(stream=True), headers={"x-fake-scenario": "gated"}
        ) as response:
            chunks = response.iter_raw()
            while b"part-0" not in received:
                received += next(chunks)
        # leaving the block closes the connection in the middle of the stream
    # The upstream generator is cancelled well before its 5 s gate timeout,
    # so the proxy closed its upstream connection right away.
    assert upstream.stream_cancelled.wait(timeout=3)
    assert upstream.gate_timeouts == 0
    rec = proxy.wait_for_record(lambda r: r["client_disconnected"])
    assert rec["error"] == {"source": "client", "type": "client_disconnected"}
    assert rec["stream_complete"] is False


# -- other endpoints -----------------------------------------------------------


def test_api_hello_is_answered_locally(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    assert httpx.head(proxy.url + "/api/hello", timeout=10).status_code == 200
    assert httpx.get(proxy.url + "/api/hello", timeout=10).status_code == 200
    assert upstream.requests == []
    rec = proxy.records()[0]
    assert (rec["endpoint"], rec["method"], rec["status"], rec["handled_locally"]) == ("hello", "HEAD", 200, True)


def test_other_paths_are_passed_through(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    response = httpx.get(proxy.url + "/v1/models?limit=5", timeout=10)
    assert response.status_code == 200 and response.json()["data"][0]["id"] == MODEL
    assert (upstream.requests[-1]["path"], upstream.requests[-1]["query"]) == ("/v1/models", "limit=5")
    assert httpx.get(proxy.url + "/nowhere", timeout=10).status_code == 404
    models, missing = proxy.records()
    assert (models["protocol"], models["endpoint"], models["status"], models["error"]) == (None, "other", 200, None)
    assert models["usage"] is None and models["tool_calls"] is None
    assert missing["status"] == 404 and missing["error"]["message"] == "Not Found"


# -- body logging ----------------------------------------------------------------


def test_bodies_are_not_logged_by_default(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    body = chat_body()
    body["messages"][1]["content"] = SECRET
    post(proxy, "/v1/chat/completions", body)
    stream(proxy, "/v1/chat/completions", dict(body, stream=True))
    assert not (proxy.run_dir / "bodies").exists()
    text = proxy.calls_path.read_text()
    assert SECRET not in text and "Hello world" not in text and "Paris" not in text
    assert all(r["bodies"] is None for r in proxy.records())


def test_log_bodies_stores_exact_bytes(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(log_bodies=True, thinking=ThinkingMode.parse("off"))
    body = chat_body()
    body["messages"][1]["content"] = SECRET
    raw = json.dumps(body).encode()
    response = post(proxy, "/v1/chat/completions", raw, headers={"authorization": "Bearer sk-secret-token"})
    streamed = stream(proxy, "/v1/chat/completions", dict(body, stream=True))

    first, second = proxy.records()
    run_dir = proxy.run_dir
    assert (run_dir / first["bodies"]["request"]).read_bytes() == raw
    forwarded = json.loads((run_dir / first["bodies"]["forwarded"]).read_bytes())
    assert forwarded["chat_template_kwargs"] == {"enable_thinking": False}
    assert (run_dir / first["bodies"]["forwarded"]).read_bytes() == upstream.requests[0]["body"]
    assert (run_dir / first["bodies"]["response"]).read_bytes() == response.content
    headers_text = (run_dir / first["bodies"]["headers"]).read_text()
    assert ["authorization", "<redacted>"] in json.loads(headers_text)["request"]
    assert "sk-secret-token" not in headers_text

    assert second["bodies"]["response"].endswith(".sse")
    assert (run_dir / second["bodies"]["response"]).read_bytes() == streamed.content
    # content goes to bodies/, never into calls.jsonl
    assert SECRET not in proxy.calls_path.read_text()


def test_upstream_token_replaces_client_authorization(
    upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]
) -> None:
    proxy = make_proxy(upstream_token="tok-upstream-123", log_bodies=True)
    post(proxy, "/v1/chat/completions", chat_body(), headers={"authorization": "Bearer placeholder"})
    post(proxy, "/v1/chat/completions", chat_body())
    assert [r["headers"]["authorization"] for r in upstream.requests] == ["Bearer tok-upstream-123"] * 2
    recorded = proxy.calls_path.read_text()
    recorded += "".join(p.read_text(errors="replace") for p in (proxy.run_dir / "bodies").iterdir())
    assert "tok-upstream-123" not in recorded
    assert "tok-upstream-123" not in repr(proxy.config)


def test_without_upstream_token_the_client_authorization_is_relayed(
    upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]
) -> None:
    proxy = make_proxy()
    post(proxy, "/v1/chat/completions", chat_body(), headers={"authorization": "Bearer client-key"})
    assert upstream.requests[0]["headers"]["authorization"] == "Bearer client-key"


def test_upstream_token_cmd() -> None:
    from bench.proxy.config import upstream_token

    assert upstream_token("printf tok-123") == "tok-123"
    for command in ("false", "printf ''", "printf 'two words'", "/nonexistent/rec auth token"):
        with pytest.raises(ValueError, match="token command") as failure:
            upstream_token(command)
        assert "two words" not in str(failure.value).replace(command, "")  # errors never echo the output


def test_one_reasoning_field(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    turns = [
        {"role": "assistant", "content": "", "reasoning_content": "look it up", "reasoning": "look it up", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "assistant", "content": "Sunny.", "reasoning_content": "", "reasoning": "only here"},
        {"role": "user", "content": "And tomorrow?"},
        {"role": "assistant", "content": "Rain.", "reasoning": "alone"},
    ]
    body = chat_body()
    body["messages"] += turns

    plain = make_proxy()
    post(plain, "/v1/chat/completions", body)
    assert upstream.requests[-1]["json"]["messages"] == body["messages"]  # off by default

    proxy = make_proxy(one_reasoning_field=True)
    post(proxy, "/v1/chat/completions", body)
    sent = [m for m in upstream.requests[-1]["json"]["messages"] if m["role"] == "assistant"]
    assert [(m.get("reasoning_content"), m.get("reasoning")) for m in sent] == [
        ("look it up", None), (None, "only here"), (None, "alone")
    ]
    assert sent[0]["tool_calls"] == [call]
    assert proxy.record()["warnings"] == ["one reasoning field: dropped a duplicate from 2 assistant message(s)"]
