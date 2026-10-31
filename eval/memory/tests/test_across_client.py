"""The across-chats gateway client against the fake gateway."""
import io

import pytest
from memory.across import client
from memory.tests.across_fake import TOKEN, FakeGateway


@pytest.fixture
def gateway(tmp_path):
    fake = FakeGateway(tmp_path, ["memory.retriever=bm25"])
    yield fake
    fake.stop()


def gw(fake, token=TOKEN):
    return client.Gateway(fake.base_url, token, timeout=5.0)


def test_only_loopback_http_urls_are_accepted():
    for bad in ("https://127.0.0.1:1", "http://example.com:1", "http://127.0.0.1:1/api", "ftp://localhost:1"):
        with pytest.raises(ValueError):
            client.Gateway(bad, "t")
    assert client.Gateway("http://localhost:8401", "t").port == 8401


def test_a_wrong_token_is_a_gateway_error(gateway):
    with pytest.raises(client.GatewayError) as err:
        gw(gateway, "wrong").health()
    assert err.value.status == 401
    assert gw(gateway).health()["configured"] is True


def test_run_turn_creates_a_chat_and_a_second_turn_continues_it(gateway):
    first = client.run_turn(gw(gateway), "My cat is called Biscuit.", "k-1", workspace=str(gateway.home))
    assert (first.status, first.turn_no, first.answer, first.http_status) == ("completed", 1, "Noted.", 201)
    assert [e["type"] for e in first.events] == ["turn.started", "turn.memory", "run.started", "assistant.message", "turn.finished"]
    assert all(e["t_ms"] >= 0 for e in first.events) and first.ttfe_ms == first.events[0]["t_ms"]
    assert first.memory()["remembered"][0]["text"] == "My cat is called Biscuit."
    second = client.run_turn(gw(gateway), "What is my cat called?", "k-2", session_id=first.session_id)
    assert (second.session_id, second.turn_no, second.http_status) == (first.session_id, 2, 202)
    assert second.answer.startswith("From my notes") and [e["turn"] for e in second.events] == [2] * len(second.events)
    assert second.server_ms()["run_started"] > 0
    repeated = gw(gateway).request("POST", f"/api/sessions/{first.session_id}/turns", {"text": "x", "client_request_id": "bad key"})
    assert repeated[0] == 400


def test_memory_is_the_last_turn_memory_event():
    """Recall sends the first event and a note saved by a tool a second, fuller one; the timing stays on the first."""
    recall = {"turn": 1, "used": [], "remembered": [], "refused": ""}
    saved = {**recall, "remembered": [{"id": 1, "text": "You live in Lisbon.", "topic": "about"}]}
    result = client.TurnResult("s", "t", 1, "completed", events=[
        {"type": "turn.started", "ts": 10.0, "t_ms": 1.0}, {"type": "turn.memory", "ts": 10.02, "data": recall},
        {"type": "tool.started", "ts": 11.0}, {"type": "turn.memory", "ts": 11.5, "data": saved},
        {"type": "turn.finished", "ts": 12.0}])
    assert result.memory()["remembered"][0]["text"] == "You live in Lisbon."
    assert result.server_ms()["memory"] == 20.0
    assert client.TurnResult("s", "t", 1, "completed").memory() is None


def test_a_key_outside_the_gateways_grammar_is_refused_before_sending(gateway):
    with pytest.raises(ValueError):
        client.run_turn(gw(gateway), "hi", "bad key")
    assert gateway.requests == []


def test_a_turn_that_does_not_end_is_a_timeout(tmp_path):
    slow = FakeGateway(tmp_path, [], delay=0.4)
    try:
        for use_stream in (True, False):
            result = client.run_turn(gw(slow), "hello there", "k-slow", timeout_s=0.3, poll_s=0.02, use_stream=use_stream)
            assert result.status == "timeout" and "did not finish" in result.error and result.wall_ms >= 300
    finally:
        slow.stop()


def test_server_ms_is_none_without_event_clocks():
    result = client.TurnResult("s", "t", 1, "completed", events=[{"type": "turn.started"}, {"type": "turn.finished", "ts": 5.0}])
    assert result.server_ms() == {"memory": None, "run_started": None, "first_message": None, "finished": None}
    assert result.memory() is None and result.ttfe_ms is None


def test_frames_skip_noise():
    body = io.BytesIO(b"retry: 2000\n\nid: 1\ndata: {\"id\": 1}\n\n: keepalive\n\ndata: not json\n\ndata: [1]\n\ndata:{\"id\": 2}\n")
    assert [e["id"] for e in client._frames(body)] == [1, 2]
