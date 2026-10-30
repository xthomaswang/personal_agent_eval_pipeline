"""The timing proxy in front of a fake model server: bytes pass through unchanged, and every request is timed."""
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit

import pytest
from memory.across import proxy
from memory.tests.across_fake import FakeModelServer

CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}


def call(base, path, body=None, method="POST"):
    """(status, headers, body) of one request to ``base`` + ``path``."""
    url = urlsplit(base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    try:
        conn.request(method, url.path + path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def logged(path, n=1):
    """The proxy's log once it holds ``n`` rows: a request is logged after its last byte has gone out, so a client that
    has read the whole reply can get here first."""
    deadline = time.time() + 5
    while time.time() < deadline:
        rows = proxy.read_log(path)
        if len(rows) >= n:
            return rows
        time.sleep(0.01)
    return proxy.read_log(path)


@pytest.fixture
def upstream():
    server = FakeModelServer(["chat-model", "embed-model"], first_delay=0.1, chunk_delay=0.1, embed_delay=0.2)
    yield server
    server.stop()


@pytest.fixture
def timing(upstream, tmp_path):
    p = proxy.TimingProxy(upstream.base_url, tmp_path / "proxy.jsonl")
    yield p
    p.stop()


def test_the_proxy_keeps_the_path_prefix_so_the_client_needs_no_change(timing, upstream):
    assert urlsplit(timing.url).path == urlsplit(upstream.base_url).path == "/v1"
    assert urlsplit(timing.url).port != urlsplit(upstream.base_url).port


def test_a_whole_reply_passes_through_and_is_timed(timing, upstream):
    direct = call(upstream.base_url, "/chat/completions", CHAT)
    status, headers, body = call(timing.url, "/chat/completions", CHAT)
    assert (status, body) == (direct[0], direct[2]) and headers["Content-Type"] == "application/json"
    (row,) = logged(timing.log_path)
    assert (row["method"], row["path"], row["model"], row["stream"], row["tools"], row["status"]) == ("POST", "/v1/chat/completions", "m", False, True, 200)
    assert row["bytes"] == len(body) and row["req_bytes"] == len(json.dumps(CHAT)) and row["error"] == ""
    assert row["t_first_byte"] - row["t_start"] >= 0.1  # the server sends nothing before it has the whole reply
    assert row["t_start"] <= row["t_headers"] <= row["t_first_byte"] <= row["t_end"]
    assert abs(row["t_start"] - time.time()) < 5  # seconds since the epoch, the clock of the gateway's events


def test_a_stream_is_passed_on_as_it_arrives_and_its_first_byte_is_logged(timing, upstream):
    direct = call(upstream.base_url, "/chat/completions", {**CHAT, "stream": True, "tools": None})[2]
    url = urlsplit(timing.url)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    t0 = time.perf_counter()
    conn.request("POST", url.path + "/chat/completions", json.dumps({**CHAT, "stream": True}), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    first = resp.read1(65536)
    got_first = time.perf_counter() - t0
    rest = resp.read()
    total = time.perf_counter() - t0
    conn.close()
    assert first + rest == direct and b"[DONE]" in rest
    assert got_first < total - 0.15  # the first event reached the client long before the last (no buffering)
    row = logged(timing.log_path)[-1]
    assert row["stream"] is True and row["tools"] is True and row["bytes"] == len(direct)
    assert 0.09 <= row["t_first_byte"] - row["t_start"] < 0.2 and row["t_end"] - row["t_first_byte"] >= 0.15


def test_embeddings_and_other_requests(timing):
    status, _, body = call(timing.url, "/embeddings", {"model": "embed-model", "input": ["a", "b"]})
    assert status == 200 and json.loads(body)["data"]
    call(timing.url, "/models", method="GET")
    embeds, models = logged(timing.log_path, 2)
    assert (embeds["path"], embeds["model"], embeds["n_input"], embeds["stream"], embeds["tools"]) == ("/v1/embeddings", "embed-model", 2, None, None)
    assert (models["method"], models["path"], models["model"], models["req_bytes"]) == ("GET", "/v1/models", None, 0)
    assert embeds["t_end"] - embeds["t_start"] >= 0.2


def test_concurrent_requests_are_logged_apart_and_overlap(timing):
    threads = [threading.Thread(target=call, args=(timing.url, "/embeddings", {"model": "e", "input": "x"})) for _ in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()
    first, second = sorted(logged(timing.log_path, 2), key=lambda r: r["t_start"])
    assert second["t_start"] < first["t_end"]  # one was in flight when the other began


def test_an_upstream_that_is_down_is_a_502_and_is_logged(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    p = proxy.TimingProxy(f"http://127.0.0.1:{port}/v1", tmp_path / "log.jsonl")
    try:
        status, _, body = call(p.url, "/chat/completions", CHAT)
    finally:
        p.stop()
    assert status == 502 and b"did not answer" in body
    (row,) = logged(tmp_path / "log.jsonl")
    assert row["status"] == 502 and row["t_first_byte"] is None and "Error" in row["error"] and row["t_end"] >= row["t_start"]


def test_only_plain_http_upstreams_are_forwarded_to(tmp_path):
    for bad in ("https://127.0.0.1:1/v1", "127.0.0.1:1"):
        with pytest.raises(ValueError):
            proxy.TimingProxy(bad, tmp_path / "x.jsonl")


def test_describe_reads_what_the_log_needs():
    assert proxy.describe("/v1/chat/completions", b'{"model": "m", "stream": true}') == {"model": "m", "stream": True, "tools": False, "n_input": None}
    assert proxy.describe("/v1/embeddings", b'{"model": "e", "input": "x"}')["n_input"] == 1
    for junk in (b"", b"not json", b"[1]", b"{" + b" " * 9_000_000 + b"}"):
        assert proxy.describe("/v1/chat/completions", junk) == {"model": None, "stream": None, "tools": None, "n_input": None}


def test_read_log_skips_lines_it_cannot_read(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"path": "/a"}\nnot json\n[1]\n{"path": "/b"}\n')
    assert [r["path"] for r in proxy.read_log(path)] == ["/a", "/b"] and proxy.read_log(tmp_path / "none.jsonl") == []
