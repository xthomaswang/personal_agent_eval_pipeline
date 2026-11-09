"""Synthetic-only TM60 tests; run directly with the pinned vendor Python -I -B."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import threading
import unittest
from collections import deque
from contextlib import ExitStack, asynccontextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

judge = None
_ALLOWED_PORTS = set()
_GUARDS = []
MODEL = "synthetic-tm60-model"
PAIR_MARKER = "PAIR_TEXT_SENTINEL"
REPLY_MARKER = "REPLY_PROSE_SENTINEL"


def setUpModule():
    """Isolate HOME before importing the runner and forbid external effects."""
    global judge, _sandbox
    stack = ExitStack()
    unittest.addModuleCleanup(stack.close)
    _sandbox = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="tm60-unit-")))
    home = _sandbox / "home"
    home.mkdir()
    stack.enter_context(mock.patch.dict(os.environ, {
        "HOME": str(home), "XDG_CONFIG_HOME": str(home / "config"),
        "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1",
    }, clear=True))
    _GUARDS.append(stack.enter_context(mock.patch.object(
        subprocess, "Popen", side_effect=AssertionError("subprocesses are forbidden"))))

    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def checked(original):
        def guarded(sock, address):
            if (not isinstance(address, tuple) or address[0] != "127.0.0.1"
                    or address[1] not in _ALLOWED_PORTS):
                raise AssertionError("only the synthetic HTTP server may be contacted")
            return original(sock, address)
        return guarded

    stack.enter_context(mock.patch.object(socket.socket, "connect", checked(connect)))
    stack.enter_context(mock.patch.object(socket.socket, "connect_ex", checked(connect_ex)))
    path = Path(__file__).resolve().parents[1] / "pairs" / "tm60_judge.py"
    spec = importlib.util.spec_from_file_location("_tm60_judge_under_test", path)
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    for target in (
        "localmelo.support.config.load", "localmelo.support.hosted.api_key",
        "localmelo.support.keychain.read", "localmelo.support.melo_serve.snapshot",
        "localmelo.support.melo_serve.ensure_running",
    ):
        _GUARDS.append(stack.enter_context(mock.patch(
            target, side_effect=AssertionError("real configuration/services are forbidden"))))
    for name in ("main", "provenance"):
        _GUARDS.append(stack.enter_context(mock.patch.object(
            judge, name, side_effect=AssertionError("entrypoint/provenance is forbidden"))))


def pair(label="other", index=0, *, old_kind="profile", new_kind="profile"):
    return {
        "label": label,
        "old": {"kind": old_kind, "text": f"{PAIR_MARKER} {index}: the toy rover uses slot 7."},
        "new": {"kind": new_kind, "text": f"{PAIR_MARKER} {index}: the toy rover now uses slot 9."},
    }


def config_for(pairs, base_url="http://127.0.0.1:0/v1"):
    return {
        "protocol": "synthetic-tm60-test",
        "set_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
        "vendor_sha": "synthetic-vendor", "model": MODEL, "base_url": base_url,
        "old_date": "2026-10-01", "new_date": "2026-11-01",
        "prompt_sha256": hashlib.sha256(judge.hippo_judge.SYSTEM_PROMPT.encode()).hexdigest(),
        "judge_version": judge.hippo_judge.JUDGE_VERSION,
        "sw_vers": "synthetic OS build", "xcodebuild_version": "synthetic Xcode build",
        "python": "synthetic-python",
    }


def reply(verdict="other", item=None, *, content=None, status=200, tokens=12):
    content = json.dumps({"verdict": verdict, "item": item}) if content is None else content
    body = {"choices": [{"message": {
        "role": "assistant", "content": content, "reasoning_content": REPLY_MARKER,
    }, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": tokens}}
    if status != 200:
        body = {"error": {"message": content}}
    return status, json.dumps(body).encode()


class FakeChatServer:
    """A real loopback HTTP server with responses and requests kept only in RAM."""

    def __init__(self, replies):
        self.replies = deque(replies)
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                owner.requests.append((self.command, self.path, dict(self.headers), json.loads(body)))
                status, payload = owner.replies.popleft() if owner.replies else (500, b'{}')
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                owner.requests.append((self.command, self.path, dict(self.headers), None))
                self.send_error(405)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        _ALLOWED_PORTS.add(self.port)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        _ALLOWED_PORTS.discard(self.port)


@asynccontextmanager
async def chat(replies):
    with FakeChatServer(replies) as server:
        provider = judge.make_provider(server.base_url, MODEL)
        try:
            # Config normalizes the oMLX alias to the vllm backend key.
            quiet = judge.hippo_encode._quiet(provider, "vllm")
            yield server, judge.send_from_provider(quiet), provider
        finally:
            await provider.close()


def metadata(expected="other", raw="other", *, after=None, supported=None,
             chinese=False, cross_kind=False, failure=None, seconds=1.0,
             http_requests=1, switching_retries=0, switching_responses=0):
    return {
        "pair_id": "p00", "expected": expected,
        "raw_verdict": None if raw is None else {"verdict": raw, "item": None if raw == "other" else 1},
        "verdict_after_support": after if after is not None else raw or "other",
        "supported": supported, "failure_type": failure, "seconds": seconds,
        "http_requests": http_requests, "switching_retries": switching_retries,
        "switching_responses": switching_responses,
        "chinese": chinese, "cross_kind": cross_kind,
    }


class FixtureMixin:
    def setUp(self):
        super().setUp()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(dir=_sandbox)))
        self.run_dir = self.root / "run"

    def tearDown(self):
        for guard in _GUARDS:
            guard.assert_not_called()
        super().tearDown()

    def rows(self):
        path = self.run_dir / "rows.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def assert_private(self, texts, captured=""):
        persisted = "\n".join(p.read_text() for p in self.run_dir.rglob("*") if p.is_file())
        for text in [PAIR_MARKER, REPLY_MARKER, *texts]:
            self.assertNotIn(text, persisted + captured)
            self.assertNotIn(json.dumps(text, ensure_ascii=True)[1:-1], persisted + captured)
        self.assertFalse(list(self.run_dir.rglob("*.tmp")))


class LoadPairsTests(FixtureMixin, unittest.TestCase):
    def load_synthetic(self, value):
        payload = json.dumps(value, ensure_ascii=False).encode()
        path = self.root / "synthetic.json"
        path.write_bytes(payload)
        return judge.load_pairs(path, hashlib.sha256(payload).hexdigest())

    def test_hash_mismatch_precedes_json_parsing(self):
        path = self.root / "invalid.json"
        path.write_bytes(b"not even JSON")
        with mock.patch.object(judge.json, "loads") as parse:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                judge.load_pairs(path, "0" * 64)
            parse.assert_not_called()

    def test_synthetic_sha_accepts_all_labels_and_kinds_without_frozen_counts(self):
        values = [pair("same"), pair("changes", 1, new_kind="episodic"),
                  pair("other", 2, old_kind="semantic")]
        values[1]["new"]["text"] = "合成玩具探测器停在九号槽。"
        self.assertEqual(self.load_synthetic(values), values)

    def test_schema_validation_uses_only_synthetic_inputs(self):
        invalid = [None, {}, [], [None], [dict(pair(), label="unknown")],
                   [dict(pair(), extra=True)], [dict(pair(), old={"kind": "profile"})],
                   [dict(pair(), new={"kind": "unknown", "text": "invented"})],
                   [dict(pair(), new={"kind": "profile", "text": 3})],
                   [dict(pair(), old={"kind": "profile", "text": "invented", "extra": True})]]
        for index, value in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.load_synthetic(value)


class HTTPJudgeTests(FixtureMixin, unittest.IsolatedAsyncioTestCase):
    async def test_valid_verdicts_use_exact_library_request_and_support_only_changes(self):
        pairs = [pair("same"), pair("changes", 1), pair("other", 2)]
        async with chat([reply("same", 1), reply("changes", 1), reply()]) as (server, send, provider):
            with mock.patch.object(judge.hippo_judge, "supported", wraps=judge.hippo_judge.supported) as support:
                for index, value in enumerate(pairs):
                    row = await judge.evaluate_pair(value, f"p{index:02d}", send)
                    self.assertEqual(row["raw_verdict"]["verdict"], value["label"])
                    self.assertEqual(row["verdict_after_support"], value["label"])
                    self.assertIsNone(row["failure_type"])
                    self.assertEqual((row["http_requests"], row["switching_retries"], row["switching_responses"]), (1, 0, 0))
                    self.assertIs(row["supported"], True if index == 1 else None)
                    self.assertTrue(math.isfinite(row["seconds"]) and row["seconds"] >= 0)
                support.assert_called_once_with(pairs[1]["new"]["text"], pairs[1]["old"]["text"],
                                                [pairs[1]["new"]["text"]])
            self.assertEqual(provider.extra_body_by_kind, {})
            self.assertEqual(len(server.requests), len(pairs))
            for value, (method, path, headers, body) in zip(pairs, server.requests):
                self.assertEqual((method, path), ("POST", "/v1/chat/completions"))
                headers = {key.lower(): val for key, val in headers.items()}
                self.assertEqual(headers["x-localmelo-call-kind"], "retrieve")
                self.assertNotIn("authorization", headers)
                self.assertEqual(body["model"], MODEL)
                self.assertEqual(body["temperature"], 0)
                self.assertEqual(body["max_tokens"], judge.hippo_encode.MAX_OUTPUT_TOKENS)
                self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], False)
                self.assertNotIn("tools", body)
                self.assertNotIn("stream", body)
                self.assertEqual(len(body["messages"]), 2)
                self.assertEqual(body["messages"][0], {
                    "role": "system", "content": judge.hippo_judge.SYSTEM_PROMPT})
                self.assertEqual(body["messages"][1]["role"], "user")
                self.assertEqual(json.loads(body["messages"][1]["content"]), {
                    "old": [{"item": 1, "date": "2026-10-01 00:00", "text": value["old"]["text"]}],
                    "new": {"date": "2026-11-01 00:00", "text": value["new"]["text"]},
                    "typed": [value["new"]["text"]],
                })

    async def test_unsupported_changes_keep_raw_but_fall_back_to_other(self):
        value = pair("changes")
        async with chat([reply("changes", 1)]) as (server, send, _):
            with mock.patch.object(judge.hippo_judge, "supported", return_value=False) as support:
                row = await judge.evaluate_pair(value, "p00", send)
            support.assert_called_once_with(value["new"]["text"], value["old"]["text"], [value["new"]["text"]])
            self.assertEqual(row["raw_verdict"], {"verdict": "changes", "item": 1})
            self.assertEqual(row["verdict_after_support"], "other")
            self.assertIs(row["supported"], False)
            self.assertIsNone(row["failure_type"])
            self.assertEqual(len(server.requests), 1)

    async def test_unreadable_out_of_range_and_http_failures_are_other_without_retry(self):
        cases = [
            ("prose", reply(content=REPLY_MARKER), "unreadable_reply"),
            ("fenced", reply(content='```json\n{"verdict":"same","item":1}\n```'), "unreadable_reply"),
            ("range-high", reply("same", 2), "unreadable_reply"),
            ("range-low", reply("changes", 0), "unreadable_reply"),
            ("boolean-item", reply("same", True), "unreadable_reply"),
            ("string-item", reply("changes", "1"), "unreadable_reply"),
            ("null-item", reply("same", None), "unreadable_reply"),
            ("unknown-verdict", reply("unrecognized", 1), "unreadable_reply"),
            ("extra-key", reply(content='{"verdict":"same","item":1,"prose":"REPLY_PROSE_SENTINEL"}'), "unreadable_reply"),
            ("json-list", reply(content="[]"), "unreadable_reply"),
            ("output-limit", reply(content='{"verdict":', tokens=judge.hippo_encode.MAX_OUTPUT_TOKENS), "output_limit"),
            ("bad-http-json", (200, REPLY_MARKER.encode()), "ProviderError"),
            ("http503", reply(status=503, content=f"{REPLY_MARKER} {PAIR_MARKER}"), "ProviderError"),
        ]
        out, err = io.StringIO(), io.StringIO()
        async with chat([response for _, response, _ in cases]) as (server, send, _):
            with redirect_stdout(out), redirect_stderr(err), mock.patch.object(
                    judge.hippo_judge, "supported") as support:
                for index, (name, _, failure) in enumerate(cases):
                    with self.subTest(case=name):
                        row = await judge.evaluate_pair(pair(), f"p{index:02d}", send)
                        self.assertIsNone(row["raw_verdict"])
                        self.assertEqual(row["verdict_after_support"], "other")
                        self.assertEqual(row["failure_type"], failure)
                        self.assertEqual((row["http_requests"], row["switching_retries"], row["switching_responses"]), (1, 0, 0))
                        self.assertIsNone(row["supported"])
                        self.assertEqual(len(server.requests), index + 1)
                support.assert_not_called()
            self.assertFalse(server.replies)
        for marker in (PAIR_MARKER, REPLY_MARKER):
            self.assertNotIn(marker, out.getvalue() + err.getvalue())

    async def test_http503_switching_retries_are_counted_and_kept_as_shipped(self):
        switching = (503, json.dumps({"error": {
            "type": "switching", "retry_after": 0.001, "message": REPLY_MARKER,
        }}).encode())
        async with chat([switching, reply(), reply("same", 1)]) as (server, send, _):
            first = await judge.evaluate_pair(pair(), "p00", send)
            self.assertEqual(
                (len(server.requests), first["raw_verdict"], first["failure_type"], first["verdict_after_support"]),
                (2, {"verdict": "other", "item": None}, None, "other"),
            )
            self.assertEqual((first["http_requests"], first["switching_retries"], first["switching_responses"]), (2, 1, 1))
            second = await judge.evaluate_pair(pair("same", 1), "p01", send)
            self.assertEqual((second["http_requests"], second["switching_retries"], second["switching_responses"]), (1, 0, 0))
            self.assertEqual(len(server.requests), 3)
            self.assertEqual(second["raw_verdict"], {"verdict": "same", "item": 1})
            report = judge.render_report([first, second], 2, config_for([]))
            self.assertIn("HTTP requests: 3; switching retries: 1; switching responses: 1.", report)
            self.assertIn("Pairs with >1 HTTP request: p00.", report)
            self.assertIn("| p00 | other | other | other | - | 2 | 1 | 1 |", report)

    async def test_cjk_ranges_check_either_side_and_cross_kind(self):
        chars = [("㐀", True), ("䶿", True), ("一", True), ("鿿", True),
                 ("㏿", False), ("䷀", False), ("ꀀ", False),
                 ("\U00020000", False), ("ひ한", False)]
        async with chat([reply("other", 1) for _ in chars]) as (server, send, _):
            for index, (char, chinese) in enumerate(chars):
                value = pair(index=index, new_kind="episodic" if index % 2 else "profile")
                value["old" if index % 2 else "new"]["text"] += char
                with self.subTest(case=index):
                    row = await judge.evaluate_pair(value, f"p{index:02d}", send)
                    self.assertIs(row["chinese"], chinese)
                    self.assertIs(row["cross_kind"], bool(index % 2))
                    self.assertEqual(row["raw_verdict"], {"verdict": "other", "item": 1})
            self.assertEqual(len(server.requests), len(chars))

    async def test_nonlocal_or_nonhttp_provider_urls_rejected_before_construction(self):
        with mock.patch.object(judge, "providers_for") as construct:
            for url in ("https://127.0.0.1:1/v1", "http://example.invalid/v1", "ftp://127.0.0.1/v1"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    judge.make_provider(url, MODEL)
            construct.assert_not_called()


class RunPairsTests(FixtureMixin, unittest.IsolatedAsyncioTestCase):
    async def test_sequential_ids_order_resume_and_private_checkpoints(self):
        pairs = [pair(judge.LABELS[i % 3], i) for i in range(12)]
        pairs[2]["old"]["text"] += " 合成文本"
        responses = [reply() for _ in pairs]
        responses[0] = reply("same", 1)
        responses[1] = reply("changes", 1)
        responses[2] = reply(status=503, content=REPLY_MARKER + pairs[2]["old"]["text"])
        out, err = io.StringIO(), io.StringIO()
        active = 0

        async with chat(responses) as (server, send, _):
            async def sequential(messages):
                nonlocal active
                self.assertEqual(active, 0)
                active += 1
                try:
                    await asyncio.sleep(0)
                    return await send(messages)
                finally:
                    active -= 1

            config = config_for(pairs, server.base_url)
            with redirect_stdout(out), redirect_stderr(err):
                code = await judge.run_pairs(pairs, self.run_dir, sequential, config,
                                             stop_requested=lambda: len(server.requests) == 3)
                self.assertEqual(code, 75)
                prefix = self.rows()
                self.assertEqual([r["pair_id"] for r in prefix], ["p00", "p01", "p02"])
                self.assertEqual(prefix[2]["failure_type"], "ProviderError")
                self.assertEqual(prefix[2]["verdict_after_support"], "other")
                self.assertEqual(await judge.run_pairs(pairs, self.run_dir, sequential, config), 0)
                rows = self.rows()
                self.assertEqual(rows[:3], prefix)
                self.assertEqual([r["pair_id"] for r in rows], [f"p{i:02d}" for i in range(12)])
                self.assertEqual([r["expected"] for r in rows], [p["label"] for p in pairs])
                done_send = mock.AsyncMock(side_effect=AssertionError("completed rows must not be asked again"))
                self.assertEqual(await judge.run_pairs(pairs, self.run_dir, done_send, config), 0)
                done_send.assert_not_awaited()
            self.assertEqual(len(server.requests), len(pairs))
            self.assertEqual([json.loads(r[3]["messages"][1]["content"])["typed"][0]
                              for r in server.requests], [p["new"]["text"] for p in pairs])
        self.assertEqual({p.name for p in self.run_dir.iterdir()}, {"config.json", "rows.jsonl", "report.md"})
        report = (self.run_dir / "report.md").read_text()
        self.assertIn("Status: complete; 12/12", report)
        self.assertIn("raw: **FAIL**", report)
        self.assert_private([p[side]["text"] for p in pairs for side in ("old", "new")],
                            out.getvalue() + err.getvalue())

    async def test_resume_configuration_identity_mismatch_rejected_without_requests(self):
        pairs = [pair()]
        original = config_for(pairs)
        self.run_dir.mkdir()
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(original))
        send = mock.AsyncMock()
        for key in ("set_sha256", "vendor_sha", "model", "base_url", "prompt_sha256",
                    "judge_version", "old_date", "new_date"):
            with self.subTest(key=key):
                changed = dict(original, **{key: "changed-synthetic-identity"})
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    await judge.run_pairs(pairs, self.run_dir, send, changed)
        send.assert_not_awaited()
        self.assertEqual(json.loads(path.read_text()), original)
        self.assertFalse((self.run_dir / "rows.jsonl").exists())

    async def test_resume_requires_an_ordered_prefix_and_matching_expected_labels(self):
        pairs = [pair("same"), pair("changes", 1)]
        first = metadata("same", "same")
        second = dict(metadata("changes", "changes"), pair_id="p01")
        bad_rows = [[second], [first, first], [second, first],
                    [dict(first, expected="other")], [dict(first, pair_id="p02")],
                    [first, second, dict(first, pair_id="p02")]]
        self.run_dir.mkdir()
        config = config_for(pairs)
        (self.run_dir / "config.json").write_text(json.dumps(config))
        rows_path = self.run_dir / "rows.jsonl"
        send = mock.AsyncMock()
        for index, rows in enumerate(bad_rows):
            with self.subTest(case=index):
                payload = "".join(json.dumps(row) + "\n" for row in rows)
                rows_path.write_text(payload)
                with self.assertRaises(ValueError):
                    await judge.run_pairs(pairs, self.run_dir, send, config)
                self.assertEqual(rows_path.read_text(), payload)
        send.assert_not_awaited()

    async def test_nonempty_unconfigured_directory_is_not_overwritten(self):
        self.run_dir.mkdir()
        (self.run_dir / "unrelated.txt").write_text("synthetic unrelated file")
        send = mock.AsyncMock()
        with self.assertRaisesRegex(ValueError, "nonempty run directory"):
            await judge.run_pairs([pair()], self.run_dir, send, config_for([pair()]))
        send.assert_not_awaited()
        self.assertFalse((self.run_dir / "config.json").exists())

    async def test_stop_file_before_first_pair_then_during_pair_and_resume(self):
        pairs = [pair(index=i) for i in range(2)]
        self.run_dir.mkdir()
        stop = self.run_dir / "STOP"
        stop.touch()
        out = io.StringIO()
        async with chat([reply(), reply()]) as (server, send, _):
            config = config_for(pairs, server.base_url)
            with redirect_stdout(out):
                self.assertEqual(await judge.run_pairs(pairs, self.run_dir, send, config), 75)
                self.assertEqual(server.requests, [])
                self.assertEqual(self.rows(), [])
                stop.unlink()

                async def stopping(messages):
                    stop.touch()  # After the boundary check, before this request finishes.
                    return await send(messages)

                self.assertEqual(await judge.run_pairs(pairs, self.run_dir, stopping, config), 75)
                self.assertEqual(len(server.requests), 1)
                self.assertEqual([r["pair_id"] for r in self.rows()], ["p00"])
                self.assertIn("stop reason: STOP", (self.run_dir / "report.md").read_text())
                stop.unlink()
                self.assertEqual(await judge.run_pairs(pairs, self.run_dir, send, config), 0)
            self.assertEqual(len(server.requests), 2)
        self.assert_private([], out.getvalue())

    async def test_deadline_equality_and_expiry_during_pair_stop_at_next_boundary(self):
        pairs = [pair(index=i) for i in range(2)]
        now = [100.0]
        clock = mock.Mock(side_effect=lambda: now[0])
        async with chat([reply()]) as (server, send, _):
            config = config_for(pairs, server.base_url)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(await judge.run_pairs(pairs, self.run_dir, send, config,
                                                       deadline=100.0, clock=clock), 75)
                self.assertEqual(server.requests, [])
                now[0] = 99.0

                async def expiring(messages):
                    now[0] = 100.0
                    return await send(messages)

                self.assertEqual(await judge.run_pairs(pairs, self.run_dir, expiring, config,
                                                       deadline=100.0, clock=clock), 75)
            self.assertEqual(len(server.requests), 1)
            self.assertEqual(len(self.rows()), 1)
            self.assertEqual(clock.call_count, 3)
            self.assertIn("stop reason: deadline", (self.run_dir / "report.md").read_text())

    async def test_term_and_int_finish_current_pair_and_restore_handlers(self):
        pairs = [pair(index=i) for i in range(2)]
        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=sig):
                run_dir = self.root / f"signal-{sig}"
                previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
                async with chat([reply()]) as (server, send, _):
                    with judge.stop_signals() as requested:
                        self.assertFalse(requested())

                        async def interrupted(messages, sig=sig):
                            signal.raise_signal(sig)
                            return await send(messages)

                        with redirect_stdout(io.StringIO()):
                            code = await judge.run_pairs(pairs, run_dir, interrupted,
                                                         config_for(pairs, server.base_url),
                                                         stop_requested=requested)
                        self.assertTrue(requested())
                    self.assertEqual(code, 75)
                    self.assertEqual(len(server.requests), 1)
                    self.assertEqual(len((run_dir / "rows.jsonl").read_text().splitlines()), 1)
                    self.assertIn("stop reason: signal", (run_dir / "report.md").read_text())
                self.assertEqual({s: signal.getsignal(s) for s in previous}, previous)

    async def test_initial_signal_wins_over_stop_and_deadline_without_request(self):
        self.run_dir.mkdir()
        (self.run_dir / "STOP").touch()
        send = mock.AsyncMock()
        with judge.stop_signals() as requested:
            signal.raise_signal(signal.SIGINT)
            with redirect_stdout(io.StringIO()):
                code = await judge.run_pairs([pair()], self.run_dir, send, config_for([pair()]),
                                             deadline=1.0, clock=lambda: 1.0, stop_requested=requested)
        self.assertEqual(code, 75)
        send.assert_not_awaited()
        self.assertIn("stop reason: signal", (self.run_dir / "report.md").read_text())

    async def test_stop_during_final_pair_does_not_mark_completed_run_partial(self):
        value = pair()
        async with chat([reply()]) as (server, send, _):
            with judge.stop_signals() as requested:
                async def finish(messages):
                    (self.run_dir / "STOP").touch()
                    signal.raise_signal(signal.SIGTERM)
                    return await send(messages)

                with redirect_stdout(io.StringIO()):
                    code = await judge.run_pairs([value], self.run_dir, finish,
                                                 config_for([value], server.base_url),
                                                 stop_requested=requested)
            self.assertEqual(code, 0)
            self.assertEqual(len(server.requests), 1)
            self.assertIn("Status: complete; 1/1", (self.run_dir / "report.md").read_text())


class GateAndReportTests(unittest.TestCase):
    def threshold_rows(self):
        return ([metadata("same", "same") for _ in range(10)]
                + [metadata("same", "changes") for _ in range(2)]
                + [metadata("same", "other") for _ in range(3)]
                + [metadata("changes", "changes") for _ in range(14)]
                + [metadata("changes", "same")]
                + [metadata("changes", "other") for _ in range(5)]
                + [metadata("other", "other") for _ in range(25)])

    def test_all_four_gate_thresholds_are_inclusive_and_one_step_outside_fails(self):
        rows = self.threshold_rows()
        expected = {"false_same": (1, 1), "false_changes": (2, 2),
                    "same_recall": (10, 10), "changes_recall": (14, 14)}
        for phase in ("raw", "after support"):
            with self.subTest(phase=phase):
                values = judge.gates(rows, phase)
                self.assertEqual(values, {key: {"count": count, "limit": limit, "pass": True}
                                          for key, (count, limit) in expected.items()})
        for gate, index, predicted in (("false_same", 35, "same"), ("false_changes", 35, "changes"),
                                        ("same_recall", 0, "other"), ("changes_recall", 15, "other")):
            with self.subTest(gate=gate):
                changed = copy.deepcopy(rows)
                changed[index]["raw_verdict"] = {"verdict": predicted, "item": 1}
                changed[index]["verdict_after_support"] = predicted
                for phase in ("raw", "after support"):
                    self.assertFalse(judge.gates(changed, phase)[gate]["pass"])

    def test_support_and_failures_affect_the_correct_gate_phase(self):
        rows = self.threshold_rows()
        rows[15]["verdict_after_support"] = "other"
        rows[15]["supported"] = False
        rows[-1] = metadata("other", None, failure="unreadable_reply")
        self.assertTrue(judge.gates(rows, "raw")["changes_recall"]["pass"])
        self.assertEqual(judge.gates(rows, "after support")["changes_recall"]["count"], 13)
        self.assertFalse(judge.gates(rows, "after support")["changes_recall"]["pass"])
        for phase in ("raw", "after support"):
            self.assertEqual(judge.gates(rows, phase)["false_same"]["count"], 1)
            self.assertEqual(judge.gates(rows, phase)["false_changes"]["count"], 2)

    def test_report_confusion_subsets_failure_counts_timing_and_partial_status(self):
        rows = [metadata("same", "same", chinese=True, seconds=1),
                metadata("changes", "changes", after="other", supported=False, cross_kind=True, seconds=2),
                metadata("other", None, chinese=True, cross_kind=True, failure="unreadable_reply", seconds=3),
                metadata("other", "same", seconds=4)]
        for index, row in enumerate(rows):
            row["pair_id"] = f"p{index:02d}"
        config = config_for([])
        report = judge.render_report(rows, 4, config)
        sections = {name: report.split(f"## {name}: ", 1)[1].split("\n## ", 1)[0]
                    for name in ("All", "Chinese", "Cross-kind")}
        self.assertTrue(sections["All"].startswith("4 pairs"))
        self.assertTrue(sections["Chinese"].startswith("2 pairs"))
        self.assertTrue(sections["Cross-kind"].startswith("2 pairs"))
        raw_all, after_all = sections["All"].split("### after support")
        self.assertIn("| same | 1 | 0 | 0 |", raw_all)
        self.assertIn("| changes | 0 | 1 | 0 |", raw_all)
        self.assertIn("| changes | 0 | 0 | 1 |", after_all)
        self.assertIn("| other | 1 | 0 | 1 |", raw_all)
        self.assertIn("| other | 1 | 0 | 1 |", after_all)
        self.assertIn("| other | 0 | 0 | 1 |", sections["Chinese"])
        self.assertIn("| same | 0 | 0 | 0 |", sections["Cross-kind"])
        self.assertIn('Failure types/counts: `{"unreadable_reply": 1}`', report)
        self.assertIn("Unsupported raw changes: 1.", report)
        self.assertIn("n=4, total=10.000000, mean=2.500000, median=2.500000, max=4.000000", report)
        self.assertIn("raw: **FAIL**", report)
        self.assertIn("after support: **FAIL**", report)
        partial = judge.render_report(rows, 60, config, "STOP")
        self.assertIn("Status: partial; 4/60", partial)
        self.assertIn("stop reason: STOP", partial)
        self.assertIn("raw: **PENDING**", partial)
        self.assertIn("after support: **PENDING**", partial)
        empty = judge.render_report([], 60, config)
        self.assertIn("Request seconds: n=0; mean, median and maximum not measured.", empty)
        complete = judge.render_report(self.threshold_rows(), 60, config)
        self.assertIn("raw: **PASS**", complete)
        self.assertIn("after support: **PASS**", complete)
        for gate, denominator in (("false_same", 45), ("false_changes", 40),
                                  ("same_recall", 15), ("changes_recall", 20)):
            self.assertRegex(complete, rf"\| raw \| {gate} \| \d+/{denominator} \|")

    def test_tm60b_reads_recall_after_support_only_and_reports_an_overall_verdict(self):
        config = dict(config_for([]), gate_profile="tm60b")
        rows = self.threshold_rows()
        # Raw recall fails (one same pair raw-other) while after-support recall holds: TM60b still passes.
        rows[0]["raw_verdict"] = {"verdict": "other", "item": None}
        report = judge.render_report(rows, 60, config)
        self.assertTrue(report.startswith("# TM60b pair check: held-out (gates)\n"))
        self.assertIn("raw: **PASS**. Precision passes; recall (fails) is not gated in this phase.", report)
        self.assertIn("after support: **PASS**", report)
        self.assertIn("Overall: **PASS**", report)
        tm60 = judge.render_report(rows, 60, config_for([]))
        self.assertIn("raw: **FAIL**", tm60)
        self.assertNotIn("Overall:", tm60)
        low_recall = copy.deepcopy(self.threshold_rows())
        low_recall[15]["verdict_after_support"] = "other"
        low_recall[15]["supported"] = False
        self.assertIn("Overall: **FAIL (recall only: safe, too little gain)**",
                      judge.render_report(low_recall, 60, config))
        raw_precision = copy.deepcopy(self.threshold_rows())
        raw_precision[35]["raw_verdict"] = {"verdict": "changes", "item": 1}
        report = judge.render_report(raw_precision, 60, config)
        self.assertIn("raw: **FAIL**", report)
        self.assertIn("Overall: **FAIL (precision)**", report)
        self.assertIn("Overall: **PENDING**", judge.render_report(rows[:4], 60, config, "STOP"))

    def test_no_gate_profile_writes_tables_without_gate_lines(self):
        config = dict(config_for([]), gate_profile="none")
        report = judge.render_report(self.threshold_rows(), 60, config)
        self.assertTrue(report.startswith("# Pair check: development (old set, no gate)\n"))
        self.assertIn("No gate: this set is development data", report)
        self.assertIn("## All: 60 pairs", report)
        self.assertIn("### after support", report)
        for absent in ("## Frozen gates", "**PASS**", "**FAIL**", "**PENDING**", "Overall:", "provisional"):
            self.assertNotIn(absent, report)


class ArgumentTests(unittest.TestCase):
    SHA = "a" * 64
    BUILD = "b" * 40

    def parse(self, *extra):
        return judge.parse_args(["--pairs", "set.json", "--run-dir", "run", *extra])

    def rejected(self, *extra):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse(*extra)

    def test_gated_profiles_pin_their_set_and_build(self):
        args = self.parse("--set-sha256", judge.SET_SHA256, "--gate", "tm60")
        self.assertEqual(args.vendor_sha, judge.VENDOR_SHA)
        args = self.parse("--set-sha256", judge.TM60B_SET_SHA256, "--gate", "tm60b")
        self.assertEqual(args.vendor_sha, judge.TM60B_VENDOR_SHA)
        self.rejected("--set-sha256", self.SHA, "--gate", "tm60b")
        self.rejected("--set-sha256", judge.SET_SHA256, "--gate", "tm60b")
        self.rejected("--set-sha256", judge.TM60B_SET_SHA256, "--gate", "tm60b", "--vendor-sha", self.BUILD)

    def test_no_gate_needs_an_explicit_build_and_a_well_formed_set_sha(self):
        args = self.parse("--set-sha256", self.SHA, "--gate", "none", "--vendor-sha", self.BUILD)
        self.assertEqual((args.gate, args.vendor_sha, args.set_sha256), ("none", self.BUILD, self.SHA))
        self.rejected("--set-sha256", self.SHA, "--gate", "none")
        self.rejected("--set-sha256", self.SHA, "--gate", "none", "--vendor-sha", "short")
        self.rejected("--set-sha256", "A" * 64, "--gate", "none", "--vendor-sha", self.BUILD)
        self.rejected("--set-sha256", self.SHA, "--vendor-sha", self.BUILD)
        self.rejected("--gate", "none", "--vendor-sha", self.BUILD)

    def test_frozen_label_counts_apply_only_when_requested(self):
        with tempfile.TemporaryDirectory(dir=_sandbox) as root:
            values = [pair("same"), pair("changes", 1), pair("other", 2)]
            payload = json.dumps(values).encode()
            path = Path(root) / "synthetic.json"
            path.write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            self.assertEqual(judge.load_pairs(path, sha), values)
            with self.assertRaisesRegex(ValueError, "frozen label counts"):
                judge.load_pairs(path, sha, judge.FROZEN_COUNTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
