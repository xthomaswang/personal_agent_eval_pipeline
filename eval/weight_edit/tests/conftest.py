"""Fixtures: a tiny random Qwen3 model (real tokenizer) in a temp dir, so tests never touch the real model or state."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# The tokenizer files come from the real model folder named by WSP_ORIGINAL; without it the tests that need it are skipped.
_REAL = os.environ.get("WSP_ORIGINAL")
REAL_MODEL = Path(_REAL).expanduser() if _REAL else None
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> Path:
    if REAL_MODEL is None or not (REAL_MODEL / "tokenizer.json").exists():
        pytest.skip("WSP_ORIGINAL is not set (needs the real model folder for the tokenizer files)")
    import torch
    from transformers import Qwen3Config, Qwen3Model

    d = tmp_path_factory.mktemp("tiny_original")
    cfg = Qwen3Config(
        vocab_size=151669,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen3Model(cfg).to(torch.bfloat16)
    model.save_pretrained(d, safe_serialization=True)
    for f in TOKENIZER_FILES:
        shutil.copy2(REAL_MODEL / f, d / f)
    (d / "1_Pooling").mkdir()
    (d / "1_Pooling" / "config.json").write_text("{}")
    return d


@pytest.fixture()
def env(tiny_model, tmp_path, monkeypatch):
    """Point every wsp location at temp dirs; the 'original' is the tiny model (copied so tests can check it stays intact)."""
    orig = tmp_path / "original"
    shutil.copytree(tiny_model, orig)
    monkeypatch.setenv("WSP_ORIGINAL", str(orig))
    monkeypatch.setenv("WSP_EDITS", str(tmp_path / "edits"))
    monkeypatch.setenv("WSP_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("WSP_RESULTS", str(tmp_path / "results"))
    monkeypatch.delenv("NIGHT_DEADLINE", raising=False)
    monkeypatch.delenv("WSP_ALLOW_MPS", raising=False)
    monkeypatch.delenv("WSP_GPU_OWNER", raising=False)
    monkeypatch.delenv("WSP_BUDGET_ACCOUNT", raising=False)
    monkeypatch.delenv("WSP_AGENT_MODE", raising=False)
    return tmp_path
