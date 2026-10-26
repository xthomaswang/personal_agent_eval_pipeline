"""Fake `omlx serve`: oMLX-shaped startup and load log lines, /v1/models and a chat endpoint.

Installed by conftest as an executable named `omlx` (shebang = the test interpreter). Behaviour:
  FAKE_OMLX_MODE   ok (default) | die (exit 3 before listening) | ignore-sigint (needs SIGKILL)
  FAKE_OMLX_LINES  extra log lines printed when the model loads (first chat request), '\\n'-separated
  FAKE_OMLX_CHILD  1: start a `sleep` child in the same process group (process-group kill test)
  FAKE_OMLX_STATE  a file: the server appends one line per request ("models" or "chat <model>")
Never binds anything but 127.0.0.1:$OMLX_PORT, which the tests take from an ephemeral port.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S") + ",000"
    print(f"{stamp} - {msg}", flush=True)


def main() -> int:
    args = sys.argv[1:]
    assert args[:1] == ["serve"], args
    base = Path(args[args.index("--base-path") + 1])
    port = int(os.environ["OMLX_PORT"])
    assert os.environ.get("HOME") == str(base / "home"), "HOME must be isolated to BASE/home"
    assert os.environ.get("OMLX_HF_CACHE_ENABLED") == "false"
    settings = json.loads((base / "settings.json").read_text())
    models = json.loads((base / "model_settings.json").read_text())["models"]
    mode = os.environ.get("FAKE_OMLX_MODE", "ok")
    state = os.environ.get("FAKE_OMLX_STATE")
    ssd = str(Path(settings["cache"]["ssd_cache_dir"]).resolve())

    print("\033[33moMLX - LLM inference, optimized for your Mac\033[0m")
    print("\033[33m└─ Version: 0.7.1rc1\033[0m", flush=True)
    if mode == "die":
        print("Configuration error: fake failure", flush=True)
        return 3
    print(f"Base path: {base}")
    print(f"Model directories: {os.environ['OMLX_MODEL_DIR']}")
    print(f"Memory guard: on (tier: {settings['memory']['memory_guard_tier']})")
    print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
    print(f"paged SSD cache: {ssd} (max: 128.0GB)")
    print(f"Hot cache: 20.0GB (in-memory)", flush=True)
    log(f"omlx.server - INFO - Default max tokens: {settings['sampling']['max_tokens']}")
    log("omlx.process_memory_enforcer - INFO - Process memory enforcer started (tier=balanced, ceiling=101.5GB, interval=1.0s)")
    if os.environ.get("FAKE_OMLX_CHILD") == "1":
        subprocess.Popen(["sleep", "600"])
    loaded: list[str] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _note(self, what: str) -> None:
            if state:
                with open(state, "a") as f:
                    f.write(what + "\n")

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._note("models")
                self._send(200, {"object": "list", "data": [{"id": k, "object": "model"} for k in models]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            mid = body.get("model")
            self._note(f"chat {mid}")
            if mid not in models:
                return self._send(404, {"error": f"model {mid} not found"})
            if mid not in loaded:
                loaded.append(mid)
                log(f"omlx.cache.paged_ssd_cache - INFO - PagedSSDCacheManager initialized: dir={ssd}, "
                    "max_size=128.00 GB, auto_size=False, existing_files=0, disk_free=400.00 GB")
                for line in filter(None, os.environ.get("FAKE_OMLX_LINES", "").split("\n")):
                    log(f"omlx.fake - INFO - {line}")
                log(f"omlx.engine_pool - INFO - Loaded model: {mid} (actual: 1.00GB, local estimate: 1.00GB)")
            self._send(200, {"id": "x", "object": "chat.completion", "model": mid,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there"},
                                          "finish_reason": "length"}],
                             "usage": {"prompt_tokens": 12, "completion_tokens": body.get("max_tokens", 8)}})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)

    def on_int(*_a) -> None:
        if mode == "ignore-sigint":
            print("ignoring SIGINT", flush=True)
            return
        print("Shutting down", flush=True)
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, on_int)
    signal.signal(signal.SIGTERM, on_int)
    print(f"Starting server at http://127.0.0.1:{port}", flush=True)
    srv.serve_forever(poll_interval=0.1)
    srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
