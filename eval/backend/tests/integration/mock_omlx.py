"""A mock `omlx serve` for the integration tests: the real orchestrator launches it like oMLX.

It serves agentbench's MockServer with the speed tests' tokenizing responder (/v1/tokenize and
/v1/detokenize from a fake renderer whose token counts also feed the chat usage, so G0 is checked for
real) on 127.0.0.1:$OMLX_PORT, which the tests take from an ephemeral port. oMLX-shaped log lines go
to stdout, which the orchestrator's Server writes to server.log.

  MOCK_FIXTURES     the mini fixture tree (the same one speed.FIXTURES_DIR points at in the test)
  MOCK_SPEED_TESTS  tests/speed (speedkit and the tokenizing responder); this file runs as a copy
  MOCK_LOG_LINES    '\\n'-separated lines printed when the warm-up loads the model
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.environ["MOCK_SPEED_TESTS"])

from agentbench.mock_server import MockServer  # noqa: E402
from test_speed_e2e import make_responder  # noqa: E402


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')},000 - {msg}", flush=True)


class Responder:
    """The served model list and oMLX's non-streaming warm-up; everything else goes to the mock."""

    def __init__(self, inner, served: list[str]) -> None:
        self.inner, self.served, self.loaded = inner, served, False

    async def __call__(self, method: str, path: str, headers: dict, body: dict):
        if method == "GET" and path.startswith("/v1/models"):
            return "json", 200, {"object": "list",
                                 "data": [{"id": m, "object": "model", "max_model_len": 131072} for m in self.served]}
        if method == "POST" and path.startswith("/v1/chat/completions") and not body.get("stream"):
            if not self.loaded:   # oMLX loads the model on its first request
                self.loaded = True
                for line in filter(None, os.environ.get("MOCK_LOG_LINES", "").split("\n")):
                    log(f"omlx.mock - INFO - {line}")
                log(f"omlx.engine_pool - INFO - Loaded model: {body.get('model')} (actual: 1.00GB)")
            return "json", 200, {"id": "warm", "object": "chat.completion", "model": body.get("model"),
                                 "choices": [{"index": 0, "finish_reason": "length",
                                              "message": {"role": "assistant", "content": "Hello"}}],
                                 "usage": {"prompt_tokens": 12, "completion_tokens": body.get("max_tokens", 8)}}
        return await self.inner(method, path, headers, body)


def main() -> int:
    args = sys.argv[1:]
    assert args[:1] == ["serve"], args
    base = Path(args[args.index("--base-path") + 1])
    port = int(os.environ["OMLX_PORT"])
    served = list(json.loads((base / "model_settings.json").read_text())["models"])
    print("oMLX - mock\n└─ Version: 0.7.1rc1", flush=True)
    srv = MockServer(port=port, responder=Responder(make_responder(Path(os.environ["MOCK_FIXTURES"])), served))
    srv.start()
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_a: stop.set())
    print(f"Starting server at http://127.0.0.1:{port}", flush=True)
    while not stop.wait(0.2):
        pass
    print("Shutting down", flush=True)
    srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
