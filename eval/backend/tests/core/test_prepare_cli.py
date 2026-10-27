from __future__ import annotations

import hashlib
import json
import os
import socket

import pytest

from modeleval import cli, paths, prepare
from modeleval.registry import parse_models

from .conftest import model_table

FAKE_OQ = '''
import argparse, json, pathlib, sys
ap = argparse.ArgumentParser()
ap.add_argument("--src"); ap.add_argument("--name"); ap.add_argument("--level")
ap.add_argument("--fix", action="append", default=[]); ap.add_argument("--no-register", action="store_true")
ap.add_argument("--allow-proxy", action="store_true"); ap.add_argument("--models")
a = ap.parse_args()
out = pathlib.Path(a.models) / "oq" / a.name
out.mkdir(parents=True)
(out / "config.json").write_text("{}")
(out / "agent-model-eval.json").write_text(json.dumps({"events": [{"step": "convert", "argv": sys.argv[1:]}]}))
print("converted", a.name, a.level, a.fix, a.no_register)
'''


@pytest.fixture
def nb(env, monkeypatch):
    """A Nanbeige-like entry: spec none, a convert level, a tokenizer fix, a cached source snapshot."""
    ckpt = env.root / "models/oq/NB-oQ5"
    m = parse_models({"models": {"nb": model_table(ckpt, served_id="NB-oQ5", spec="none", prepare={
        "convert": {"level": 5}, "tokenizer_fixes": ["drop-bos-post-processor"]},
        source={"hf_repo": "Org/NB", "revision": "c" * 40})}})["nb"]
    snap = prepare.snapshot_dir(m)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0" * 4096)
    # the fake writes models/oq/<name> relative to its own location: scripts/../models/oq
    (env.scripts / "oq_convert.py").write_text(FAKE_OQ + "# --no-register\n")
    monkeypatch.setattr(paths, "REPO", env.root)
    return m


def test_register_links_relative_and_is_idempotent(env, nb):
    nb.path.mkdir(parents=True)
    (nb.path / "config.json").write_text("{}")
    link = prepare.register(nb)
    assert os.readlink(link) == "../oq/NB-oQ5" and link.resolve() == nb.path.resolve()
    assert prepare.register(nb) == link
    assert prepare.status(nb)["link"] == "ok"


def test_register_refuses_a_foreign_link_and_a_listening_port(env, nb):
    nb.path.mkdir(parents=True)
    (nb.path / "config.json").write_text("{}")
    (paths.MODELS_DIR / "NB-oQ5").symlink_to(env.ckpt)
    with pytest.raises(prepare.PrepareError, match="points at"):
        prepare.register(nb)
    with socket.socket() as s:
        s.bind(("127.0.0.1", env.port))
        s.listen()
        with pytest.raises(prepare.PrepareError, match="listening"):
            prepare.register(nb)


def test_download_guards(env, nb):
    with pytest.raises(prepare.PrepareError, match="nothing to download"):
        prepare.download(env.model)
    cmd = prepare.download(nb, min_free_gb=0, dry_run=True)
    assert cmd[1:] == ["download", "Org/NB", "--revision", "c" * 40]
    env.lock.write_text("holder: x\npurpose: modeleval qwen speed (timed)\n")
    with pytest.raises(prepare.PrepareError, match="speed leg"):
        prepare.download(nb, min_free_gb=0, dry_run=True)
    env.lock.write_text("holder: x\npurpose: agent bench segment 3\n")   # free text: may be timed
    with pytest.raises(prepare.PrepareError, match="speed leg"):
        prepare.download(nb, min_free_gb=0, dry_run=True)
    env.lock.write_text("holder: x\npurpose: modeleval qwen capability, accuracy only (untimed)\n")
    assert prepare.download(nb, min_free_gb=0, dry_run=True)   # allowed during a capability leg


def test_convert_command_and_refusal_of_bench_home_writes(env, nb):
    cmd = prepare.convert_command(nb)
    assert cmd[1:3] == ["-I", "-B"] and "--fix" in cmd and cmd[-1] == "--no-register"
    assert cmd[cmd.index("--level") + 1] == "5"
    (env.scripts / "oq_convert.py").write_text("SETTINGS = '.bench-home/model_settings.json'\n")
    with pytest.raises(prepare.PrepareError, match="no-register"):
        prepare.convert_command(nb)


def test_convert_checks(env, nb):
    assert prepare.convert_checks(nb) > 0
    nb.path.mkdir(parents=True)
    with pytest.raises(prepare.PrepareError, match="never rewritten"):
        prepare.convert_checks(nb)
    nb.path.rmdir()
    with socket.socket() as s:
        s.bind(("127.0.0.1", env.port))
        s.listen()
        with pytest.raises(prepare.PrepareError, match="no model server"):
            prepare.convert_checks(nb)


def test_convert_under_the_lock_then_register(env, nb, capsys):
    out = prepare.convert(nb)
    assert (out / "config.json").exists() and (paths.MODELS_DIR / "NB-oQ5").is_symlink()
    calls = (env.root / "lock.log").read_text().splitlines()
    assert calls[0].startswith("acquire modeleval nb accuracy only (untimed), convert")
    assert calls[-1].startswith("release") and not env.lock.exists()
    printed = capsys.readouterr().out
    assert "LOCK-NOTICE" in printed and "converted NB-oQ5 5 ['drop-bos-post-processor'] True" in printed


def test_convert_releases_the_lock_when_the_script_fails(env, nb):
    (env.scripts / "oq_convert.py").write_text("import sys; sys.exit(7)  # --no-register\n")
    with pytest.raises(prepare.PrepareError, match="exited 7"):
        prepare.convert(nb)
    assert not env.lock.exists() and not (paths.MODELS_DIR / "NB-oQ5").exists()


def _toml(env) -> None:
    (env.root / "models.toml").write_text(f'''
[models.fake-1b]
display = "Fake"
served_id = "Fake-1B-mtp"
path = "{env.ckpt}"
source = {{ hf_repo = "org/Fake-1B", revision = "{'0' * 40}" }}
quant = "oQ5"
family = "fake_1b"
quant_dir = "oq5"
spec = "mtp"
vision = false
reference = true
sampling = {{ temperature = 1.0, top_p = 0.95, top_k = 20 }}
proxy_sampling = "inject:qwen3.8-gen-config"
server_log_expect = ["MTP path activated"]
''')


def test_cli_models_plan_and_dry_run(env, monkeypatch, capsys):
    monkeypatch.setattr(paths, "REPO", env.root)
    monkeypatch.delenv(cli.TOKENIZER_ENV, raising=False)   # main() may set it; undone after the test
    _toml(env)
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "fake-1b" in out and "Fake-1B-mtp" in out and " ok " in out
    assert cli.main(["plan", "fake-1b", "--legs", "speed"]) == 0
    assert "quiet window" in capsys.readouterr().out
    assert cli.main(["run", "fake-1b", "--dry-run", "--unverified"]) == 0
    assert "unverified True" in capsys.readouterr().out
    assert not env.lock.exists() and not paths.RESULTS.exists()
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1" and os.environ["GIT_OPTIONAL_LOCKS"] == "0"
    assert cli.main(["register", "nope"]) == 5


def test_selftest_hygiene_checks_run_read_only():
    rows = cli.hygiene_checks()
    names = [r[1] for r in rows]
    assert "models.toml" in names and any("quietlib" in n for n in names)
    assert all(r[0] in ("ok", "WARN", "FAIL") for r in rows)


def test_tokenizer_dir_defaults_to_the_reference_checkpoint(env, monkeypatch):
    """At startup $AGENTBENCH_TOKENIZER_DIR becomes the registry's reference checkpoint, but only when it holds
    the tokenizer files (else agentbench's HF-snapshot fallback stays open); a value already set wins."""
    from agentbench.tokenizer import FILES

    monkeypatch.setattr(paths, "REPO", env.root)
    _toml(env)                                     # fake-1b is the reference, at env.ckpt
    monkeypatch.delenv(cli.TOKENIZER_ENV, raising=False)
    cli.default_tokenizer_dir()
    assert cli.TOKENIZER_ENV not in os.environ
    for f in FILES:
        (env.ckpt / f).write_text("{}")
    assert cli.reference_tokenizer_dir() == env.ckpt
    cli.default_tokenizer_dir()
    assert os.environ[cli.TOKENIZER_ENV] == str(env.ckpt)
    monkeypatch.setenv(cli.TOKENIZER_ENV, str(env.root / "elsewhere"))
    cli.default_tokenizer_dir()
    assert os.environ[cli.TOKENIZER_ENV] == str(env.root / "elsewhere")
    (env.root / "models.toml").write_text("not [toml")
    assert cli.reference_tokenizer_dir() is None   # a broken registry is the command's problem, not startup's


def _fake_build(prompt: bytes):
    """agentbench's build() in miniature: a committed file, a generated prompt, the manifest, an extra prompt."""
    def build(out, seed, log=print, extras=True):
        files = {"system_prompt.md": b"S\n", "generated/prompts/cold-2000.json": prompt}
        for rel, data in files.items():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(data)
        manifest = {"generator_version": 1, "seed": seed, "pillow": "13.0.0", "tokenizer_snapshot": out.name,
                    "files": {rel: {"sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)} for rel, d in files.items()}}
        (out / "manifest.json").write_text(json.dumps(manifest))
        (out / "generated/prompts/m3-off-110000.json").write_text("{}")
        log("fitting ...")
        return dict(manifest, extra_prompts={"m3-off-110000": 110000})
    return build


def test_fixtures_command_installs_only_a_build_that_matches_the_manifest(tmp_path, monkeypatch):
    import agentbench.fixtures.build as fixtures_build
    import agentbench.tokenizer as tokenizer

    root = tmp_path / "fixtures"
    root.mkdir()
    _fake_build(b"x")(root, 7)                     # the committed state (another tokenizer directory name)
    committed = (root / "manifest.json").read_bytes()
    (root / "generated/prompts/cold-2000.json").unlink()
    monkeypatch.setattr(tokenizer, "snapshot_dir", lambda: tmp_path)
    monkeypatch.setattr(fixtures_build, "build", _fake_build(b"x"))
    lines: list[str] = []
    assert cli.build_fixtures(root, log=lines.append) == 0
    assert (root / "generated/prompts/cold-2000.json").read_bytes() == b"x"
    assert (root / "generated/prompts/m3-off-110000.json").exists()
    assert (root / "manifest.json").read_bytes() == committed   # committed files are never written
    assert lines[-1].startswith("fixtures check: OK") and "m3-off-110000 (110000 tokens)" in "\n".join(lines)

    monkeypatch.setattr(fixtures_build, "build", _fake_build(b"y"))   # drift: another prompt
    lines.clear()
    assert cli.build_fixtures(root, log=lines.append) == 1
    assert "MISMATCH generated/prompts/cold-2000.json: sha256 differs from the manifest" in lines
    assert (root / "generated/prompts/cold-2000.json").read_bytes() == b"x"   # left as it was

    def no_tokenizer():
        raise FileNotFoundError("no local Qwen3.8 tokenizer files")
    monkeypatch.setattr(tokenizer, "snapshot_dir", no_tokenizer)
    lines.clear()
    assert cli.build_fixtures(root, log=lines.append) == 5 and "no local Qwen3.8 tokenizer" in lines[0]
