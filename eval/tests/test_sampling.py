"""The --sampling policy."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from bench.proxy import ProxyConfig, SamplingPolicy
from bench.proxy.__main__ import build_parser
from bench.proxy.sampling import DEFAULT_SAMPLING, apply_sampling, sampling_view
from tests.support import MODEL, FakeUpstream, ProxyHandle

CLIENT_SAMPLING = {"temperature": 0.2, "top_p": 0.8, "seed": 7, "presence_penalty": 0.5, "min_p": 0.05}


def test_parse() -> None:
    assert str(SamplingPolicy.parse("passthrough")) == "passthrough"
    assert str(SamplingPolicy.parse("strip")) == "strip"
    assert str(SamplingPolicy.parse("inject:qwen3.8-gen-config")) == "inject:qwen3.8-gen-config"
    assert str(SamplingPolicy.parse("INJECT:omlx-default")) == "inject:omlx-default"
    for bad in ("", "inject", "inject:unknown", "default"):
        with pytest.raises(ValueError):
            SamplingPolicy.parse(bad)


def test_the_bench_default_is_the_model_generation_config() -> None:
    config = ProxyConfig(upstream="http://127.0.0.1:8500", run_dir="/tmp/unused", run_id="r")
    assert config.sampling == DEFAULT_SAMPLING
    assert str(build_parser().parse_args([]).sampling) == "inject:qwen3.8-gen-config"
    assert DEFAULT_SAMPLING.values == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert SamplingPolicy.parse("inject:omlx-default").values == {"temperature": 1.0, "top_p": 0.95, "top_k": 0}


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("passthrough", CLIENT_SAMPLING),
        ("strip", {}),
        ("inject:qwen3.8-gen-config", {"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
        ("inject:omlx-default", {"temperature": 1.0, "top_p": 0.95, "top_k": 0}),
    ],
)
def test_apply_sampling(policy: str, expected: dict) -> None:
    body = {"model": MODEL, "messages": [], "max_tokens": 100, **CLIENT_SAMPLING, "repetition_penalty": 1.1, "top_k": 40}
    if policy == "passthrough":
        expected = {**CLIENT_SAMPLING, "repetition_penalty": 1.1, "top_k": 40}
    apply_sampling(body, SamplingPolicy.parse(policy))
    assert sampling_view(body) == expected
    assert body["max_tokens"] == 100 and body["model"] == MODEL


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_default_policy_end_to_end(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle], path: str) -> None:
    proxy = make_proxy(sampling=DEFAULT_SAMPLING)
    body = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], **CLIENT_SAMPLING}
    response = httpx.post(proxy.url + path, json=body, timeout=10)
    assert response.status_code == 200
    sent = upstream.requests[-1]["json"]
    assert sampling_view(sent) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert sent["max_tokens"] == 64
    rec = proxy.record()
    assert rec["sampling_policy"] == "inject:qwen3.8-gen-config"
    assert rec["sampling_original"] == CLIENT_SAMPLING
    assert rec["sampling_forwarded"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def test_strip_and_passthrough_end_to_end(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **CLIENT_SAMPLING}
    strip = make_proxy(sampling=SamplingPolicy.parse("strip"))
    httpx.post(strip.url + "/v1/chat/completions", json=body, timeout=10)
    assert sampling_view(upstream.requests[-1]["json"]) == {}
    assert strip.record()["sampling_forwarded"] == {}

    passthrough = make_proxy()
    httpx.post(passthrough.url + "/v1/chat/completions", json=body, timeout=10)
    assert sampling_view(upstream.requests[-1]["json"]) == CLIENT_SAMPLING
    rec = passthrough.record()
    assert rec["sampling_policy"] == "passthrough" and rec["sampling_original"] == rec["sampling_forwarded"]


def test_count_tokens_has_no_sampling(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy(sampling=DEFAULT_SAMPLING)
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    httpx.post(proxy.url + "/v1/messages/count_tokens", json=body, timeout=10)
    assert "temperature" not in upstream.requests[-1]["json"]
    rec = proxy.record()
    assert rec["sampling_original"] is None and rec["sampling_forwarded"] is None


def test_random_seed_is_fresh_per_request_and_recorded(
    upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]
) -> None:
    proxy = make_proxy(sampling=DEFAULT_SAMPLING, random_seed=True)
    body = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], **CLIENT_SAMPLING}
    for _ in range(3):
        httpx.post(proxy.url + "/v1/chat/completions", json=body, timeout=10)
    seeds = [r["json"]["seed"] for r in upstream.requests]
    assert len(set(seeds)) == 3 and all(isinstance(s, int) and 0 <= s < 2**31 for s in seeds)
    records = proxy.records()
    assert [r["sampling_forwarded"]["seed"] for r in records] == seeds  # the client's seed 7 is replaced
    assert {r["sampling_policy"] for r in records} == {"inject:qwen3.8-gen-config+random-seed"}
    assert build_parser().parse_args(["--random-seed"]).random_seed is True
    assert build_parser().parse_args([]).random_seed is False

    httpx.post(proxy.url + "/v1/messages", json=body, timeout=10)  # Anthropic requests have no seed field
    assert "seed" not in upstream.requests[-1]["json"]
