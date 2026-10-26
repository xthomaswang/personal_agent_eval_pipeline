from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from bench.proxy import ProxyConfig, create_app
from bench.proxy.sampling import PASSTHROUGH_SAMPLING
from tests.support import FakeUpstream, ProxyHandle, ServerThread


@pytest.fixture
def upstream() -> Iterator[FakeUpstream]:
    fake = FakeUpstream()
    server = ServerThread(fake.app).start()
    fake.url = server.url
    yield fake
    fake.release_gates()
    server.stop()


@pytest.fixture
def make_proxy(tmp_path: Path, upstream: FakeUpstream) -> Iterator[Callable[..., ProxyHandle]]:
    servers: list[ServerThread] = []

    def make(**overrides: Any) -> ProxyHandle:
        settings: dict[str, Any] = {
            "upstream": upstream.url,
            "run_dir": tmp_path / f"run-{len(servers)}",
            "run_id": "test-run",
            "harness": "pytest",
            "echo": False,
            # Most tests check that bodies pass through untouched; the bench
            # default (inject:qwen3.8-gen-config) has tests of its own.
            "sampling": PASSTHROUGH_SAMPLING,
        }
        settings.update(overrides)
        config = ProxyConfig(**settings)
        server = ServerThread(create_app(config)).start()
        servers.append(server)
        return ProxyHandle(url=server.url, config=config)

    yield make
    for server in servers:
        server.stop()


@pytest.fixture
def proxy(make_proxy: Callable[..., ProxyHandle]) -> ProxyHandle:
    return make_proxy()
