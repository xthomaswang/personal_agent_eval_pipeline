from __future__ import annotations

import pytest

from modeleval import paths, registry
from modeleval.registry import RegistryError, parse_models

from .conftest import model_table


def _models(**tables):
    return parse_models({"models": tables})


def test_repo_registry_loads_three_models_with_one_reference():
    models = registry.load_models()
    assert set(models) == {"qwen3-8-27b-oq5", "nanbeige4-3-33b-oq5", "qwen3-8-flash-next-oq4e"}
    ref = registry.reference(models)
    assert ref.key == "qwen3-8-27b-oq5" and ref.spec == "mtp" and ref.quant == "oQ5"
    assert ref.prepare == {"convert": {"level": 5}}   # download + convert work for the reference on a fresh Mac
    assert ref.source["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    nb = models["nanbeige4-3-33b-oq5"]
    assert nb.spec == "none" and not nb.vision and nb.sampling["top_k"] == 0
    assert nb.source["revision"] == "c1093c501cc6460427c0ac634db7223d0450c0ee"
    assert nb.prepare == {"convert": {"level": 5}, "tokenizer_fixes": ["drop-bos-post-processor"]}
    fx = models["qwen3-8-flash-next-oq4e"]
    assert fx.model_settings == {"qwen4_ple_ssd_offload": True, "moe_expert_offload_enabled": False}
    assert fx.quant == "oQ4e" and fx.path == paths.HOME / ".omlx-0.7/models/Qwen3.8-Flash-Next-oQ4e-mtp"
    assert {k: (m.family, m.quant_dir) for k, m in models.items()} == {
        "qwen3-8-27b-oq5": ("qwen3.8_27b", "oq5"), "nanbeige4-3-33b-oq5": ("nanbeige4.3_33b", "oq5"),
        "qwen3-8-flash-next-oq4e": ("qwen3.8_flash_next", "oq4e")}
    assert ref.results_dir == paths.RESULTS / paths.machine_id() / "qwen3.8_27b" / "oq5" / "backend"
    for m in models.values():
        assert m.server_log_expect, m.key
        assert str(m.path).startswith(str(paths.HOME))   # ~ expanded on load
        assert m.to_dict()["path"].startswith("~/")


def test_repo_registry_file_has_no_absolute_home_paths():
    assert "/Users/" not in (paths.REPO / "models.toml").read_text()


def test_get_by_key_or_served_id(tmp_path):
    ms = _models(a=model_table(tmp_path))
    assert registry.get(ms, "a").key == "a"
    assert registry.get(ms, "Fake-1B-mtp").key == "a"
    with pytest.raises(RegistryError, match="unknown model"):
        registry.get(ms, "nope")


@pytest.mark.parametrize("change, message", [
    ({"spec": "dflash"}, "spec must be one of"),
    ({"sampling": {"temperature": 1.0, "top_p": 0.95}}, "sampling must have exactly"),
    ({"proxy_sampling": "inject:made-up"}, "proxy_sampling"),
    ({"source": {"hf_repo": "org/x", "revision": "main"}}, "40-hex revision"),
    ({"model_settings": {"max_tokens": 65536}}, "may not set"),
    ({"model_settings": {"mtp_enabled": True}}, "may not set"),
    ({"server_log_expect": ["Lightning (MTP"]}, "not a regex"),
    ({"path": "relative/dir"}, "path must be absolute"),
    ({"prepare": {"tokenizer_fixes": ["fix-everything"]}}, "unknown tokenizer fix"),
    ({"surprise": 1}, "unknown field"),
    ({"family": "Qwen3.8/27B"}, "family is one lower-case directory name"),
    ({"quant_dir": ".."}, "quant_dir is one lower-case directory name"),
    ({"quant_dir": "oQ5"}, "quant_dir is one lower-case directory name"),
])
def test_invalid_entries_are_refused(tmp_path, change, message):
    with pytest.raises(RegistryError, match=message):
        _models(a=model_table(tmp_path, **change))


def test_missing_field_is_refused(tmp_path):
    t = model_table(tmp_path)
    del t["quant"]
    with pytest.raises(RegistryError, match="missing 'quant'"):
        _models(a=t)


def test_exactly_one_reference_and_unique_served_ids(tmp_path):
    with pytest.raises(RegistryError, match="exactly one model"):
        _models(a=model_table(tmp_path), b=model_table(tmp_path, served_id="Other"))
    with pytest.raises(RegistryError, match="exactly one model"):
        _models(a=model_table(tmp_path, reference=False))
    with pytest.raises(RegistryError, match="duplicate served_id"):
        _models(a=model_table(tmp_path), b=model_table(tmp_path, reference=False))
    with pytest.raises(RegistryError, match="share a results directory"):
        _models(a=model_table(tmp_path), b=model_table(tmp_path, reference=False, served_id="Other"))
    assert len(_models(a=model_table(tmp_path), b=model_table(tmp_path, reference=False, served_id="Other",
                                                               quant_dir="oq8"))) == 2


def test_pins_absent_is_empty_and_epoch_id_forms(tmp_path):
    assert registry.load_pins(tmp_path / "pins.toml") == {}
    (tmp_path / "pins.toml").write_text('[epoch]\nid = "e1"\n')
    assert registry.epoch_id(registry.load_pins(tmp_path / "pins.toml")) == "e1"
    assert registry.epoch_id({"epoch": "e2"}) == "e2"
    assert registry.epoch_id({}) is None


def test_capability_reference_lookup_is_keyed_by_protocol_and_epoch():
    flash = "qwen3-8-flash-next-oq4e"
    assert registry.capability_reference(2, "4271c2f4a925", None) == flash   # capability-2
    assert registry.capability_reference(2, "8c545ce3471c", None) == flash   # capability-3
    assert registry.capability_reference(2, "7582825160ad", None) is None    # protocol 1's epoch has none in protocol 2
    assert registry.capability_reference(1, "8c545ce3471c", "legacy") == "legacy"   # protocol 1 keeps its reference
