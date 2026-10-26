"""The per-run Claude Code config copies (DESIGN §6.1): byte copies with only named values replaced,
and the safety variables asserted on what the launcher's shell actually sees."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modeleval import paths
from modeleval.capability import epoch as epochs
from modeleval.capability import runner

LIVE_ENV = paths.MLSYS / "agent/configs/claude-code/claude-code.env"
LIVE_SETTINGS = paths.MLSYS / "agent/homes/claude-code/settings.json"
CLAUDE_BIN = paths.HOME / ".local/share/claude/versions/2.1.302"

pytestmark = pytest.mark.skipif(not (LIVE_ENV.exists() and LIVE_SETTINGS.exists()), reason="live Claude Code config missing")


def _patched(tmp_path: Path, text: str, served_id: str = "Nanbeige4.3-33B-oQ5") -> tuple[Path, str]:
    patched = runner.patch_claude_env(text, runner.claude_values(served_id, CLAUDE_BIN))
    path = tmp_path / "claude-code.env"
    path.write_text(patched)
    return path, patched


def test_only_the_named_lines_change(tmp_path: Path) -> None:
    original = LIVE_ENV.read_text()
    path, patched = _patched(tmp_path, original)
    changed = [(a, b) for a, b in zip(original.splitlines(), patched.splitlines(), strict=True) if a != b]
    assert {b.split("=")[0].removeprefix("export ") for _, b in changed} == set(runner.CLAUDE_MODEL_VARS)
    assert all('"Nanbeige4.3-33B-oQ5"' in b for _, b in changed)
    env = runner.shell_env(path)
    assert runner.claude_env_problems(original, patched, env, "Nanbeige4.3-33B-oQ5", CLAUDE_BIN) == []
    assert env["ANTHROPIC_API_KEY"] == "" and env["ANTHROPIC_AUTH_TOKEN"]
    assert env["__SHELL__CLAUDE_BIN"] == str(CLAUDE_BIN) and "CLAUDE_BIN" not in env  # never exported


def test_the_reference_model_gets_a_byte_identical_copy(tmp_path: Path) -> None:
    original = LIVE_ENV.read_text()
    _, patched = _patched(tmp_path, original, served_id="Qwen3.8-27B-oQ5-mtp")
    assert patched == original


def test_values_are_replaced_by_name_wherever_they_are(tmp_path: Path) -> None:
    text = 'export ANTHROPIC_MODEL=old   # comment\nCLAUDE_BIN=/x/y\n' + "".join(
        f'export {n}="old"\n' for n in runner.CLAUDE_MODEL_VARS[1:]
    ) + "export CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000\n"
    patched = runner.patch_claude_env(text, runner.claude_values("M-1", Path("/opt/claude/2.1.302")))
    assert patched.splitlines()[0] == "export ANTHROPIC_MODEL=M-1   # comment"
    assert patched.splitlines()[1] == "CLAUDE_BIN=/opt/claude/2.1.302"
    assert patched.splitlines()[-1] == "export CLAUDE_CODE_AUTO_COMPACT_WINDOW=131072"


def test_a_missing_variable_is_refused() -> None:
    text = LIVE_ENV.read_text().replace("export CLAUDE_CODE_SUBAGENT_MODEL=", "export SOMETHING_ELSE=")
    with pytest.raises(runner.Refused, match="CLAUDE_CODE_SUBAGENT_MODEL"):
        runner.patch_claude_env(text, runner.claude_values("M", CLAUDE_BIN))


@pytest.mark.parametrize(
    ("edit", "complaint"),
    [
        (lambda t: t.replace('export ANTHROPIC_API_KEY=""', "# key line removed"), "ANTHROPIC_API_KEY"),
        (lambda t: t + 'export ANTHROPIC_API_KEY="sk-real"\n', "ANTHROPIC_API_KEY"),
        (lambda t: t.replace("export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1", "export CLAUDE_CODE_DISABLE_CLAUDE_MDS=0"),
         "CLAUDE_CODE_DISABLE_CLAUDE_MDS"),
        (lambda t: t.replace("export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1", ""), "CLAUDE_CODE_DISABLE_AUTO_MEMORY"),
        (lambda t: t.replace("export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "true"),
         "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
        (lambda t: t.replace('export ANTHROPIC_AUTH_TOKEN="omlx"', 'export ANTHROPIC_AUTH_TOKEN=""'), "ANTHROPIC_AUTH_TOKEN"),
        (lambda t: t.replace("export CLAUDE_CONFIG_DIR=", "CLAUDE_CONFIG_DIR_UNUSED="), "CLAUDE_CONFIG_DIR"),
        (lambda t: t + "unset ANTHROPIC_DEFAULT_OPUS_MODEL\n", "ANTHROPIC_DEFAULT_OPUS_MODEL"),
    ],
)  # fmt: skip
def test_safety_variables_are_asserted_on_the_sourced_copy(tmp_path: Path, edit, complaint: str) -> None:  # noqa: ANN001
    template = edit(LIVE_ENV.read_text())
    path, patched = _patched(tmp_path, template)
    problems = runner.claude_env_problems(template, patched, runner.shell_env(path), "Nanbeige4.3-33B-oQ5", CLAUDE_BIN)
    assert any(complaint in p for p in problems), problems


def test_an_unnamed_line_change_is_caught() -> None:
    original = LIVE_ENV.read_text()
    tampered = original.replace("export API_TIMEOUT_MS=3000000", "export API_TIMEOUT_MS=1")
    problems = runner.claude_env_problems(original, tampered, {}, "M", CLAUDE_BIN)
    assert any("unnamed line changed" in p for p in problems)


def test_settings_copy_changes_only_the_model() -> None:
    text = LIVE_SETTINGS.read_text()
    patched = runner.patch_claude_settings(text, "Nanbeige4.3-33B-oQ5")
    assert json.loads(patched) == {**json.loads(text), "model": "Nanbeige4.3-33B-oQ5"}
    assert len(patched) - len(text) == len("Nanbeige4.3-33B-oQ5") - len(json.loads(text)["model"])
    with pytest.raises(runner.Refused):
        runner.patch_claude_settings('{"model": "a", "x": {"model": "b"}}', "M")


def test_write_claude_config(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    for rel, src in ((epochs.CLAUDE_ENV_COPY, LIVE_ENV), (epochs.CLAUDE_SETTINGS_COPY, LIVE_SETTINGS)):
        (vendor / rel).parent.mkdir(parents=True, exist_ok=True)
        (vendor / rel).write_bytes(src.read_bytes())
    env_file, home = runner.write_claude_config(vendor, tmp_path / "config", "Nanbeige4.3-33B-oQ5", CLAUDE_BIN)
    assert runner.shell_env(env_file)["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "Nanbeige4.3-33B-oQ5"
    assert [p.name for p in home.iterdir()] == ["settings.json"]
    env_copy = vendor / epochs.CLAUDE_ENV_COPY
    env_copy.write_text(env_copy.read_text().replace('export ANTHROPIC_API_KEY=""', ""))
    with pytest.raises(runner.Refused, match="ANTHROPIC_API_KEY"):
        runner.write_claude_config(vendor, tmp_path / "config2", "Nanbeige4.3-33B-oQ5", CLAUDE_BIN)
