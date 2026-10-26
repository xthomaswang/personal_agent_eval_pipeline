"""Image hashes, sizes and images_new."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable

import httpx

from bench.proxy.images import describe_image, image_size
from tests.support import MODEL, FakeUpstream, ProxyHandle, b64, jpeg_header, png


def webp_vp8x(width: int, height: int) -> bytes:
    payload = b"VP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return b"RIFF" + struct.pack("<I", 4 + len(payload)) + b"WEBP" + payload


def webp_vp8l(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    payload = b"VP8L" + struct.pack("<I", 5) + b"\x2f" + bits.to_bytes(4, "little")
    return b"RIFF" + struct.pack("<I", 4 + len(payload)) + b"WEBP" + payload


def webp_vp8(width: int, height: int) -> bytes:
    frame = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height)
    payload = b"VP8 " + struct.pack("<I", len(frame)) + frame
    return b"RIFF" + struct.pack("<I", 4 + len(payload)) + b"WEBP" + payload


def test_image_size_from_headers() -> None:
    assert image_size(png(1280, 720)) == (1280, 720)
    assert image_size(jpeg_header(1280, 800)) == (1280, 800)
    assert image_size(b"GIF89a" + struct.pack("<HH", 640, 480) + b"\x00" * 8) == (640, 480)
    assert image_size(webp_vp8x(1280, 731)) == (1280, 731)
    assert image_size(webp_vp8l(300, 200)) == (300, 200)
    assert image_size(webp_vp8(1024, 768)) == (1024, 768)
    assert image_size(b"BM" + b"\x00" * 16 + struct.pack("<ii", 800, -600)) == (800, 600)
    assert image_size(b"not an image") is None


def test_describe_image_for_both_formats() -> None:
    data = png(64, 32, fill=7)
    sha = hashlib.sha256(data).hexdigest()
    anthropic = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(data)}}
    openai = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(data)}"}}
    assert describe_image(anthropic) == describe_image(openai) == (sha, (64, 32))
    # unpadded base64 is accepted too
    unpadded = {"type": "image", "source": {"type": "base64", "data": b64(data).rstrip("=")}}
    assert describe_image(unpadded) == (sha, (64, 32))
    # images by reference cannot be identified without fetching them
    assert describe_image({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}) == (None, None)
    assert describe_image({"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}) == (None, None)


def anthropic_image(data: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(data)}}


def openai_image(data: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(data)}"}}


def test_images_new_counts_first_appearances_in_the_run(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    shot1, shot2 = png(1280, 720, fill=1), png(1280, 800, fill=2)
    tools = [{"name": "screenshot", "input_schema": {"type": "object"}}]

    def turn(*images: bytes) -> dict:
        content = [anthropic_image(img) for img in images] + [{"type": "text", "text": "next"}]
        return {"model": MODEL, "max_tokens": 100, "tools": tools, "messages": [{"role": "user", "content": content}]}

    url = proxy.url + "/v1/messages"
    httpx.post(url, json=turn(shot1), timeout=10)
    # count_tokens sees shot2 first, but it is not a generation request
    httpx.post(proxy.url + "/v1/messages/count_tokens", json=turn(shot1, shot2), timeout=10)
    httpx.post(url, json=turn(shot1, shot2), timeout=10)  # history plus one new screenshot
    httpx.post(url, json=turn(shot1, shot2, shot2), timeout=10)  # nothing new
    first, counted, second, third = proxy.records()

    assert first["images"] == 1 and first["images_new"] == 1
    assert first["image_sha256"] == [hashlib.sha256(shot1).hexdigest()]
    assert first["image_wh"] == [[1280, 720]] and first["images_new_long_side"] == [1280]
    assert counted["endpoint"] == "count_tokens" and counted["images"] == 2 and counted["images_new"] is None
    assert second["images"] == 2 and second["images_new"] == 1 and second["images_new_long_side"] == [1280]
    assert second["image_wh"] == [[1280, 720], [1280, 800]]
    assert third["images"] == 3 and third["images_new"] == 0 and third["images_new_long_side"] == []
    # content stays out of the record
    assert b64(shot1)[:40] not in proxy.calls_path.read_text()


def test_images_new_is_tracked_per_run_id(upstream: FakeUpstream, proxy: ProxyHandle) -> None:
    shot = png(1280, 720, fill=3)
    body = {"model": MODEL, "messages": [{"role": "user", "content": [openai_image(shot)]}]}
    url = proxy.url + "/v1/chat/completions"
    httpx.post(url, json=body, timeout=10)
    httpx.post(url, json=body, timeout=10, headers={"X-Bench-Run-Id": "other-run"})
    httpx.post(url, json=body, timeout=10)
    assert [r["images_new"] for r in proxy.records()] == [1, 1, 0]


def test_requests_without_images(upstream: FakeUpstream, make_proxy: Callable[..., ProxyHandle]) -> None:
    proxy = make_proxy()
    httpx.post(proxy.url + "/v1/chat/completions", json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}, timeout=10)
    rec = proxy.record()
    assert (rec["images"], rec["image_sha256"], rec["image_wh"], rec["images_new"]) == (0, [], [], 0)
