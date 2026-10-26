"""Unit tests for SSE parsing, response observation and request summaries."""

from __future__ import annotations

import json

import pytest

from bench.proxy.observe import ResponseObserver, normalize_usage, summarize_error
from bench.proxy.request_info import classify, summarize_request, tool_names_sha256, tools_raw_sha256
from bench.proxy.sse import SSEParser
from tests.support import ANTHROPIC_MESSAGE, OPENAI_COMPLETION, anthropic_stream, openai_stream


def parse_all(data: bytes, chunk_size: int) -> list:
    parser = SSEParser()
    events = []
    for i in range(0, len(data), chunk_size):
        events += parser.feed(data[i : i + chunk_size])
    return events + parser.close()


def test_sse_parser_is_independent_of_chunking() -> None:
    # Multi-byte characters ("é" is 2 bytes, "€" 3) get split by small chunks.
    data = b"".join(anthropic_stream()) + 'data: {"t": "café €"}\n\n'.encode()
    whole = parse_all(data, len(data))
    assert len(whole) == len(anthropic_stream()) + 1
    for size in (1, 2, 3, 7, 64):
        assert parse_all(data, size) == whole
    assert json.loads(whole[-1].data) == {"t": "café €"}


def test_sse_parser_handles_crlf_comments_and_multiline_data() -> None:
    data = b": keep-alive\r\n\r\nevent: x\r\ndata: a\r\ndata: b\r\n\r\ndata: tail"
    events = parse_all(data, 5)
    assert events[0].comment
    assert (events[1].event, events[1].data) == ("x", "a\nb")
    assert events[2].data == "tail"  # flushed by close()


def observe(protocol: str, endpoint: str, data: bytes, content_type: str, chunk_size: int = 0) -> ResponseObserver:
    observer = ResponseObserver(protocol, endpoint, 200, content_type)
    step = chunk_size or len(data) or 1
    for i in range(0, len(data), step):
        observer.feed(data[i : i + step], wall=100.0 + i, mono=float(i))
    observer.finish()
    return observer


@pytest.mark.parametrize("chunk_size", [0, 1, 13])
def test_openai_stream_observation(chunk_size: int) -> None:
    observer = observe("openai", "chat_completions", b"".join(openai_stream(True)), "text/event-stream", chunk_size)
    r = observer.results()
    assert r["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "cached_tokens": 3}
    assert r["tool_names"] == ["get_weather", "get_time"] and r["tool_calls"] == 2
    assert r["stop_reason"] == "tool_calls"
    assert r["sse_keepalives"] == 1 and r["stream_complete"] is True
    assert (r["output_text_chars"], r["output_reasoning_chars"]) == (11, 18)
    # Thinking comes first, so the first output precedes the first content
    # (they share a timestamp when everything arrives in one chunk).
    assert observer.first_output is not None and observer.first_content is not None
    if chunk_size:
        assert observer.first_output[1] < observer.first_content[1]
    else:
        assert observer.first_output == observer.first_content


@pytest.mark.parametrize("chunk_size", [0, 1, 29])
def test_anthropic_stream_observation(chunk_size: int) -> None:
    observer = observe("anthropic", "messages", b"".join(anthropic_stream()), "text/event-stream", chunk_size)
    r = observer.results()
    # message_delta (input 0 + creation 8 + read 4) replaces message_start's estimate of 10
    assert r["usage"] == {"prompt_tokens": 12, "completion_tokens": 9, "cached_tokens": 4}
    assert r["usage_raw"]["input_tokens"] == 0
    assert r["tool_names"] == ["Read", "Bash"] and r["tool_calls"] == 2
    assert r["stop_reason"] == "tool_use"
    assert r["sse_keepalives"] == 1 and r["stream_complete"] is True
    assert (r["output_text_chars"], r["output_reasoning_chars"]) == (len("Reading."), len("Hmm."))


def test_non_streaming_bodies_with_keepalive_spaces() -> None:
    data = b"   " + json.dumps(OPENAI_COMPLETION).encode()
    r = observe("openai", "chat_completions", data, "application/json", 2).results()
    assert r["tool_names"] == ["get_weather"] and r["usage"]["prompt_tokens"] == 11
    assert r["sse_events"] is None and r["stream_complete"] is None

    r = observe("anthropic", "messages", b" " + json.dumps(ANTHROPIC_MESSAGE).encode(), "application/json").results()
    assert r["tool_names"] == ["Read"] and r["stop_reason"] == "tool_use"
    assert r["usage"] == {"prompt_tokens": 12, "completion_tokens": 9, "cached_tokens": 4}


def test_stream_without_end_marker_is_not_complete() -> None:
    observer = observe("openai", "chat_completions", b"".join(openai_stream(False)[:-1]), "text/event-stream")
    assert observer.results()["stream_complete"] is False


def test_usage_normalization() -> None:
    assert normalize_usage("anthropic", {"input_tokens": 30, "output_tokens": 2}) == {
        "prompt_tokens": 30,
        "completion_tokens": 2,
        "cached_tokens": None,
    }
    assert normalize_usage("openai", {"prompt_tokens": 5, "completion_tokens": 1}) == {
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "cached_tokens": None,
    }
    assert normalize_usage("openai", None) is None


def test_error_summary_drops_validation_input() -> None:
    body = {"detail": [{"type": "missing", "loc": ["body", "x"], "msg": "Field required", "input": {"secret": "s3cr3t"}}]}
    summary = summarize_error(body, 422)
    assert summary == {
        "source": "upstream",
        "status": 422,
        "type": "validation_error",
        "details": [{"type": "missing", "loc": ["body", "x"], "msg": "Field required"}],
    }
    anthropic = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    assert summarize_error(anthropic, 529) == {"source": "upstream", "status": 529, "type": "overloaded_error", "message": "busy"}
    assert len(summarize_error({"detail": "x" * 1000}, 400)["message"]) < 400


def test_classify() -> None:
    assert classify("/v1/chat/completions") == ("openai", "chat_completions")
    assert classify("/v1/messages") == ("anthropic", "messages")
    assert classify("/v1/messages/count_tokens") == ("anthropic", "count_tokens")
    assert classify("/api/hello") == (None, "hello")
    assert classify("/v1/models") == (None, "other")


def test_request_summary_counts_without_content() -> None:
    body = {
        "model": "m",
        "stream": True,
        "max_tokens": 32000,
        "tools": [{"name": "Read"}, {"name": "Bash"}],
        "messages": [
            {"role": "user", "content": [{"type": "image", "source": {}}, {"type": "text", "text": "x"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": [{"type": "image", "source": {}}]}]},
            {"role": "user", "content": [{"type": "document", "source": {}}]},
        ],
    }
    summary = summarize_request("anthropic", "messages", body)
    assert summary == {
        "model": "m",
        "stream": True,
        "role": "main",
        "messages": 3,
        "tools": 2,
        "request_tool_names": ["Bash", "Read"],
        "request_tools_sha256": tool_names_sha256(["Read", "Bash"]),
        "request_tools_raw_sha256": tools_raw_sha256([{"name": "Read"}, {"name": "Bash"}]),
        "max_tokens": 32000,
        "images": 2,
        # the sources are empty, so the images cannot be identified
        "image_sha256": [None, None],
        "image_wh": [None, None],
    }
