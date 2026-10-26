"""Fixtures for tests/speed (helpers live in speedkit.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Writer headers run `git status` on this repo, which the main session commits to: never take the index lock.
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


@pytest.fixture(scope="session")
def fixtures_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from speedkit import mini_fixtures

    return mini_fixtures(tmp_path_factory.mktemp("fixtures"))
