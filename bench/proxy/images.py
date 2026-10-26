"""Identify the images in a request: sha256 of the image bytes and width x height.

Only inline images (base64 or data URLs) can be identified; an image given
by URL or file id is counted but has no hash or size. Sizes come from the
file header (PNG, JPEG, GIF, WebP, BMP), so no image library is needed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any
from urllib.parse import unquote_to_bytes

ImageInfo = tuple[str | None, tuple[int, int] | None]

_JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def describe_image(block: dict[str, Any]) -> ImageInfo:
    """Return ``(sha256 hex, (width, height))`` for an image content block."""
    data = _image_bytes(block)
    if data is None:
        return None, None
    return hashlib.sha256(data).hexdigest(), image_size(data)


def _image_bytes(block: dict[str, Any]) -> bytes | None:
    if block.get("type") == "image":  # Anthropic
        source = block.get("source")
        if isinstance(source, dict) and source.get("type") == "base64" and isinstance(source.get("data"), str):
            return _b64decode(source["data"])
        return None
    # OpenAI: {"type": "image_url", "image_url": {"url": ...}} or a bare URL
    # string; the Responses API's input_image carries the URL directly.
    url = block.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" in header:
            return _b64decode(payload)
        return unquote_to_bytes(payload)
    return None


def _b64decode(text: str) -> bytes | None:
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        return None


def image_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from the header of a PNG, JPEG, GIF, WebP or BMP file."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        return _jpeg_size(data)
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_size(data)
    if data[:2] == b"BM" and len(data) >= 26:
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = int.from_bytes(data[22:26], "little", signed=True)
        return abs(width), abs(height)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # markers without a length
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image or start of scan before any frame header
            return None
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in _JPEG_SOF:
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        i += 2 + length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X" and len(data) >= 30:
        return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
    return None
