import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Fixtures built once per test session into a temporary directory."""
    try:
        from agentbench.tokenizer import snapshot_dir
        snapshot_dir()
    except FileNotFoundError:
        pytest.skip("local Qwen3.8 snapshot not present")
    from agentbench.fixtures.build import build

    root = tmp_path_factory.mktemp("fixtures")
    manifest = build(root, log=lambda s: None)
    return root, manifest


@pytest.fixture(scope="session")
def fx(built):
    from agentbench.fixtures.store import Fixtures

    return Fixtures(built[0])


@pytest.fixture(scope="session")
def qwen():
    from agentbench.tokenizer import default_prompt

    return default_prompt()


def start_mock(root: Path, behavior: str, **overrides):
    from agentbench.mock_server import MockServer
    from agentbench.mock_smart import SmartResponder

    resp = SmartResponder.from_fixtures(root, behavior, **overrides)
    return MockServer(port=0, responder=resp).start(), resp


def cli_args(url: str, root: Path, out: Path, profile: str, framework: str, *extra: str) -> list[str]:
    return ["--profile", profile, "--base-url", url, "--model", "mock", "--framework", framework, "--tag", "t",
            "--quick", "--out-root", str(out), "--fixtures", str(root), *extra]


MY_PID = str(os.getpid())
