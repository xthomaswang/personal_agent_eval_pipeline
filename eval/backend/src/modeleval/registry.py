"""Model registry: models.toml and pins.toml (DESIGN.md §3, docs/CONTRACTS.md).

The registry is the only place per-model choices live (served id, checkpoint, quantization,
speculative decoding, sampling, server-log expectations). Everything else in the pipeline is the
fixed setting of the protocol version, so a value that is not here is the same for every model.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from modeleval import paths

SPECS = ("mtp", "none")
SAMPLING_KEYS = ("temperature", "top_p", "top_k")
REQUIRED = ("display", "served_id", "path", "source", "quant", "family", "quant_dir", "spec", "vision", "sampling",
            "proxy_sampling")
OPTIONAL = ("reference", "model_settings", "server_log_expect", "prepare", "notes")
PROXY_PRESETS = ("passthrough", "strip", "inject:omlx-default", "inject:qwen3.8-gen-config")
TOKENIZER_FIXES = ("drop-bos-post-processor",)
# Per-model oMLX settings the pipeline writes itself or must never set (DESIGN §2): a per-model
# max_tokens would give Hermes and localmelo, which send none, a different budget per model.
FORBIDDEN_MODEL_SETTINGS = ("max_tokens", "max_context_window", "mtp_enabled", "temperature", "top_p", "top_k")
_KEY = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
# docs/benchmark/<family>/<quant_dir>/ (the results contract): lower case, one path component each
_DIR = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class RegistryError(ValueError):
    """models.toml or pins.toml does not satisfy the registry rules."""


@dataclass(frozen=True)
class Model:
    key: str
    display: str
    served_id: str
    path: Path
    quant: str
    family: str           # docs/benchmark/<family>/..., e.g. qwen3.8_27b
    quant_dir: str        # docs/benchmark/<family>/<quant_dir>/..., e.g. oq5 (the quantization as served)
    spec: Literal["mtp", "none"]
    vision: bool
    reference: bool
    sampling: dict
    proxy_sampling: str
    source: dict
    model_settings: dict = field(default_factory=dict)
    server_log_expect: list[str] = field(default_factory=list)
    prepare: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self, tilde: bool = True) -> dict:
        """JSON-ready dict; the path is written with ~ unless tilde=False (spec files in runs/)."""
        d = dataclasses.asdict(self)
        d["path"] = paths.tilde(self.path) if tilde else str(self.path)
        return d

    @property
    def link(self) -> Path:
        """The models/hf-bench entry oMLX discovers this model through (oMLX model id = its name)."""
        return paths.MODELS_DIR / self.served_id

    @property
    def results_dir(self) -> Path:
        """Where this model's verified runs go: docs/benchmark/<family>/<quant_dir>/backend/ (one <run-id>/
        directory per run)."""
        return paths.results_dir(self.family, self.quant_dir)


def _expand(p: str) -> Path:
    return paths.expand(p)   # ${name} from the config's [paths], then ~


def _check_model(key: str, raw: dict) -> list[str]:
    """All problems of one [models.<key>] table (empty = valid)."""
    errs = []
    if not _KEY.match(key):
        errs.append(f"{key}: key must be lowercase letters, digits, '.' or '-'")
    for k in REQUIRED:
        if k not in raw:
            errs.append(f"{key}: missing '{k}'")
    for k in raw:
        if k not in REQUIRED + OPTIONAL:
            errs.append(f"{key}: unknown field '{k}'")
    if errs:
        return errs
    if raw["spec"] not in SPECS:
        errs.append(f"{key}: spec must be one of {SPECS}, got {raw['spec']!r}")
    if not isinstance(raw["vision"], bool) or not isinstance(raw.get("reference", False), bool):
        errs.append(f"{key}: vision and reference must be booleans")
    s = raw["sampling"]
    if not isinstance(s, dict) or sorted(s) != sorted(SAMPLING_KEYS):
        errs.append(f"{key}: sampling must have exactly {SAMPLING_KEYS}")
    elif not (0 <= s["temperature"] <= 2 and 0 < s["top_p"] <= 1 and isinstance(s["top_k"], int) and s["top_k"] >= 0):
        errs.append(f"{key}: sampling values out of range: {s}")
    if raw["proxy_sampling"] not in PROXY_PRESETS:
        errs.append(f"{key}: proxy_sampling must be one of {PROXY_PRESETS}")
    src = raw["source"]
    if not isinstance(src, dict) or "hf_repo" not in src or not _SHA.match(str(src.get("revision", ""))):
        errs.append(f"{key}: source needs hf_repo and a 40-hex revision")
    for k in ("family", "quant_dir"):
        if not isinstance(raw[k], str) or not _DIR.match(raw[k]) or raw[k] in (".", ".."):
            errs.append(f"{key}: {k} is one lower-case directory name (docs/benchmark/<family>/<quant_dir>), "
                        f"got {raw[k]!r}")
    if "/" in raw["served_id"] or raw["served_id"] in (".", ".."):
        errs.append(f"{key}: served_id is a directory name, got {raw['served_id']!r}")
    if not str(raw["path"]).startswith(("~/", "/", "${")):
        errs.append(f"{key}: path must be absolute or start with ~/ or a ${{name}} of the config's [paths]")
    ms = raw.get("model_settings", {})
    bad = [k for k in ms if k in FORBIDDEN_MODEL_SETTINGS]
    if bad:
        errs.append(f"{key}: model_settings may not set {bad} (written by the pipeline or forbidden, DESIGN §2)")
    if raw["spec"] == "none" and ms.get("mtp_enabled"):
        errs.append(f"{key}: spec 'none' with mtp_enabled")
    for rx in raw.get("server_log_expect", []):
        try:
            re.compile(rx)
        except re.error as e:
            errs.append(f"{key}: server_log_expect {rx!r} is not a regex: {e}")
    prep = raw.get("prepare", {})
    for fix in prep.get("tokenizer_fixes", []):
        if fix not in TOKENIZER_FIXES:
            errs.append(f"{key}: unknown tokenizer fix {fix!r} (known: {TOKENIZER_FIXES})")
    conv = prep.get("convert")
    if conv is not None and not (isinstance(conv, dict) and 2 <= float(conv.get("level", 0)) <= 8):
        errs.append(f"{key}: prepare.convert needs a level between 2 and 8")
    return errs


def parse_models(data: dict) -> dict[str, Model]:
    """Validate a parsed models.toml and build the Model entries (exactly one reference)."""
    tables = data.get("models")
    if not isinstance(tables, dict) or not tables:
        raise RegistryError("models.toml has no [models.<key>] tables")
    errs = [e for key, raw in tables.items() for e in _check_model(key, raw)]
    if errs:
        raise RegistryError("; ".join(errs))
    refs = [k for k, raw in tables.items() if raw.get("reference")]
    if len(refs) != 1:
        raise RegistryError(f"exactly one model must have reference = true, found {refs or 'none'}")
    ids = [raw["served_id"] for raw in tables.values()]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RegistryError(f"duplicate served_id: {dup}")
    dirs = [(raw["family"], raw["quant_dir"]) for raw in tables.values()]
    dup_dirs = sorted({f"{f}/{q}" for f, q in dirs if dirs.count((f, q)) > 1})
    if dup_dirs:
        raise RegistryError(f"two models would share a results directory: {dup_dirs}")
    out = {}
    for key, raw in tables.items():
        out[key] = Model(
            key=key, display=raw["display"], served_id=raw["served_id"], path=_expand(raw["path"]),
            quant=raw["quant"], family=raw["family"], quant_dir=raw["quant_dir"], spec=raw["spec"], vision=raw["vision"], reference=raw.get("reference", False),
            sampling=dict(raw["sampling"]), proxy_sampling=raw["proxy_sampling"], source=dict(raw["source"]),
            model_settings=dict(raw.get("model_settings", {})),
            server_log_expect=list(raw.get("server_log_expect", [])),
            prepare=dict(raw.get("prepare", {})), notes=list(raw.get("notes", [])),
        )
    return out


def load_models(path: Path | None = None) -> dict[str, Model]:
    """The registry from models.toml (default: the repo's), validated."""
    path = path or paths.REPO / "models.toml"
    with open(path, "rb") as f:
        return parse_models(tomllib.load(f))


def reference(models: dict[str, Model]) -> Model:
    return next(m for m in models.values() if m.reference)


CAPABILITY_REFERENCES = {
    (2, "4271c2f4a925"): "qwen3-8-flash-next-oq4e",
    (2, "8c545ce3471c"): "qwen3-8-flash-next-oq4e",
}


def capability_reference(protocol: int, epoch: str | None, legacy_reference: str | None) -> str | None:
    """Resolve a capability reference without changing the historical speed or v1 reference."""
    if protocol == 1:
        return legacy_reference
    return CAPABILITY_REFERENCES.get((protocol, epoch))


def get(models: dict[str, Model], name: str) -> Model:
    """A model by registry key or served id, with a message that lists the keys."""
    if name in models:
        return models[name]
    for m in models.values():
        if m.served_id == name:
            return m
    raise RegistryError(f"unknown model {name!r}; known: {', '.join(models)}")


def load_pins(path: Path | None = None) -> dict:
    """pins.toml (the harness epoch, DESIGN §6.5); {} while no epoch has been created."""
    path = path or paths.REPO / "pins.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def epoch_id(pins: dict) -> str | None:
    """The current harness epoch id from pins.toml: `epoch = "<id>"` or `[epoch] id = "<id>"`."""
    e = pins.get("epoch")
    if isinstance(e, dict):
        e = e.get("id")
    return str(e) if e else None
