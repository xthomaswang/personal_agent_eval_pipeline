import pytest

from agentbench.client import ServerConfig, StreamClient, run_concurrent, split_inline_think
from agentbench.metrics import decode_metrics
from agentbench.mock_server import MockServer
from agentbench.sse import SSEParser


@pytest.fixture(scope="module")
def mock():
    s = MockServer().start()
    yield s
    s.stop()


def client(mock, **kw):
    return StreamClient(ServerConfig(base_url=mock.url, model="mock", **kw))


def body(c):
    return c.openai_body([{"role": "user", "content": "hi"}], max_tokens=16)


async def test_role_keepalive_progress_ignored(mock):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = "text"
        r = await c.chat(body(c))
    assert r.ok and r.content == "Hello world!" and r.finish_reason == "stop"
    assert r.ttft is not None and r.ttft >= 0.15  # first content arrives after ~0.18 s
    assert r.n_ignored["keepalive"] == 1 and r.n_ignored["progress"] >= 2 and r.n_ignored["role_only"] == 1
    assert r.cached_tokens == 1024 and r.prompt_tokens == 1200 and r.completion_tokens == 5
    m = decode_metrics(r)
    assert m["tokens_source"] == "usage" and m["decode_tps"] > 0 and m["n_token_chunks"] == 3


async def test_tool_calls_split_across_deltas(mock):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = "tools"
        r = await c.chat(body(c))
    assert [t["arguments_parsed"]["path"] for t in r.tool_calls] == ["~/project/README.md", "~/project/setup.cfg"]
    assert all(t["name"] == "fs_read_file" for t in r.tool_calls) and r.finish_reason == "tool_calls"
    assert r.ttft is not None and not r.markup_leak()


async def test_reasoning_and_inline_think(mock):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = "reasoning"
        r = await c.chat(body(c))
        assert r.reasoning == "Thinking." and r.content == "Answer" and r.chunks[0].kind == "reasoning"
        c.http.headers["x-mock-scenario"] = "inline_think"
        r = await c.chat(body(c))
        assert r.reasoning_inline and r.reasoning == "plan" and r.content == "Done"
    assert split_inline_think("abc\n</think>\n\nxyz") == ("xyz", "abc", True)


async def test_markup_leak_flagged(mock):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = "leak"
        r = await c.chat(body(c))
    assert "<tool_call>" in r.markup_leak()


@pytest.mark.parametrize("code", [400, 500])
async def test_http_errors(mock, code):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = f"http_{code}"
        r = await c.chat(body(c))
    assert r.error["type"] == "http" and r.error["status"] == code and r.ttft is None


async def test_read_timeout_and_total_timeout(mock):
    async with client(mock, read_timeout_s=0.5) as c:
        c.http.headers["x-mock-scenario"] = "stall"
        r = await c.chat(body(c))
    assert r.error["type"] == "timeout" and r.content == "a"
    async with client(mock, total_timeout_s=0.5) as c:
        c.http.headers["x-mock-scenario"] = "slow_headers"
        r = await c.chat(body(c))
    assert r.error == {"type": "timeout", "phase": "total", "limit_s": 0.5}


async def test_stream_error_event(mock):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = "error_event"
        r = await c.chat(body(c))
    assert r.error["type"] == "stream_error"


async def test_anthropic_stream(mock):
    async with client(mock) as c:
        r = await c.messages({"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
    assert r.ok and r.reasoning == "hmm" and r.content == "Reading it." and r.finish_reason == "tool_use"
    assert r.tool_calls[0]["name"] == "fs_read_file" and r.tool_calls[0]["arguments_parsed"] == {"path": "~/a.txt"}
    assert r.ttft >= 0.15 and r.cached_tokens == 800 and r.completion_tokens == 17 and r.n_ignored["keepalive"] == 1


async def test_anthropic_error_event(mock):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = "error_event"
        r = await c.messages({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
    assert r.error["type"] == "stream_error"


async def test_concurrency_runner_starts_together(mock):
    async with client(mock) as c:
        c.http.headers["x-mock-scenario"] = "text"
        recs, spread = await run_concurrent([lambda: c.chat(body(c)) for _ in range(3)])
    assert all(r.ok for r in recs) and spread < 0.05
    sends = sorted(r.t_send_wall for r in recs)
    assert sends[-1] - sends[0] < 0.05
    assert max(r.e2e for r in recs) < 3 * min(r.e2e for r in recs)  # overlapped, not serialized


def test_sse_parser_rules():
    p = SSEParser()
    assert p.feed_line(": ping") is None and p.n_comments == 1
    assert p.feed_line("event: delta") is None and p.feed_line("data:{\"a\":1}") is None
    ev = p.feed_line("")
    assert ev.event == "delta" and ev.data == '{"a":1}'
    p.feed_line("data: x")
    p.feed_line("data: y")
    assert p.feed_line("\r").data == "x\ny"
