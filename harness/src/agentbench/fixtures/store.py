"""Load fixtures built by `agentbench fixtures build` and assemble request messages.

Message templates reference screenshots as {"type": "image_ref", "image": "<id>"}.
`resolve()` turns them into OpenAI `image_url` parts (base64 PNG data URIs), Anthropic
`image` blocks, or, in --text-only mode, the screenshot's text equivalent (an
accessibility-tree description with exactly the same token count as the image).
"""

from __future__ import annotations

import base64
import copy
import json
import re
import secrets
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .build import GENERATOR_VERSION, NONCE, REQUEST_ID, ZERO12, nonce_line, nonce_tool_prefix


class FixturesMissing(RuntimeError):
    pass


def default_root() -> Path:
    return Path(__file__).resolve().parents[4] / "fixtures"


def new_digits(n: int = 12) -> str:
    """Fresh random digits for nonces and request ids (one token per digit)."""
    return "".join(str(secrets.randbelow(10)) for _ in range(n))


class Fixtures:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else default_root()
        mf = self.root / "manifest.json"
        if not mf.exists():
            raise FixturesMissing(f"no fixtures at {self.root}; run `agentbench fixtures build` first")
        self.manifest = json.loads(mf.read_text())
        if self.manifest.get("generator_version") != GENERATOR_VERSION:
            raise FixturesMissing("fixtures were built by another generator version; rebuild them")
        if not (self.root / "generated").exists():
            raise FixturesMissing(f"{self.root}/generated is missing; run `agentbench fixtures build`")
        self.system_prompt = (self.root / "system_prompt.md").read_text(encoding="utf-8").rstrip("\n")
        self.tools_openai: list[dict[str, Any]] = self._json("tools.openai.json")
        self.tools_anthropic: list[dict[str, Any]] = self._json("tools.anthropic.json")
        self.m1 = self._json("m1_m5.json")
        self.m4 = self._json("m4_trajectory.json")
        self.m7 = self._json("m7_suite.json")
        self.gate = self._json("gate.json")
        self.images: dict[str, dict[str, Any]] = self.manifest["images"]
        self.sizes: dict[str, Any] = self.manifest["sizes"]

    def _json(self, rel: str) -> Any:
        return json.loads((self.root / rel).read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- images
    @lru_cache(maxsize=None)
    def image_b64(self, image_id: str) -> str:
        return base64.b64encode((self.root / self.images[image_id]["png"]).read_bytes()).decode("ascii")

    @lru_cache(maxsize=None)
    def image_text(self, image_id: str) -> str:
        return (self.root / self.images[image_id]["text"]).read_text(encoding="utf-8")

    def resolve_content(self, content: Any, *, text_only: bool, api: str = "openai") -> Any:
        if not isinstance(content, list):
            return content
        out = []
        for part in content:
            if part.get("type") != "image_ref":
                out.append(dict(part))
            elif text_only:
                out.append({"type": "text", "text": self.image_text(part["image"])})
            elif api == "anthropic":
                out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": self.image_b64(part["image"])}})
            else:
                out.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + self.image_b64(part["image"])}})
        return out

    def resolve(self, messages: list[dict[str, Any]], *, text_only: bool) -> list[dict[str, Any]]:
        return [dict(m, content=self.resolve_content(m.get("content"), text_only=text_only)) for m in messages]

    @staticmethod
    def screenshot_message(image_id: str, caption: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"type": "image_ref", "image": image_id}, {"type": "text", "text": caption}]}

    # ---------------------------------------------------------------- prompts
    @lru_cache(maxsize=None)
    def prompt(self, name: str) -> dict[str, Any]:
        """A generated prompt; sizes that were not pre-built are generated on the fly with
        the same deterministic procedure (needs the local tokenizer)."""
        path = self.root / "generated" / "prompts" / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        m = re.fullmatch(r"(cold|m3-off|m3-on)-(\d+)", name)
        if not m:
            raise FixturesMissing(f"no generated prompt {name}; available: {self.available_prompts()}")
        from ..tokenizer import default_prompt
        from .build import make_prompt
        print(f"note: {name} is not pre-built; generating it now (not covered by the manifest)", file=sys.stderr)
        return make_prompt(default_prompt(), m.group(1), int(m.group(2)), self.manifest["seed"])

    def available_prompts(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "generated" / "prompts").glob("*.json"))

    def prompt_messages(self, name: str, nonce: str | None) -> list[dict[str, Any]]:
        """Messages of a generated prompt with its nonce placeholder filled in: fresh
        digits to bust prefix caches, or zeros (same token count) to allow reuse."""
        msgs = copy.deepcopy(self.prompt(name)["messages"])
        for m in msgs:
            if isinstance(m.get("content"), str) and NONCE in m["content"]:
                m["content"] = m["content"].replace(NONCE, nonce_line(nonce or ZERO12))
        return msgs

    # ---------------------------------------------------------------- agent
    def agent_system(self) -> dict[str, Any]:
        return {"role": "system", "content": self.system_prompt}

    def tools(self, nonce: str | None = None, api: str = "openai") -> list[dict[str, Any]]:
        """Agent tools; with a nonce, the first tool's description starts with it (the Qwen
        template renders the tools block before the system prompt)."""
        tools = copy.deepcopy(self.tools_anthropic if api == "anthropic" else self.tools_openai)
        if nonce is not None:
            first = tools[0] if api == "anthropic" else tools[0]["function"]
            first["description"] = nonce_tool_prefix(nonce) + first["description"]
        return tools

    def tool_result(self, i: int, request_id: str | None = None) -> str:
        pool = self.m1["tool_results"]
        return pool[i % len(pool)].replace(REQUEST_ID, request_id or new_digits())

    def schemas(self) -> dict[str, dict[str, Any]]:
        return {t["function"]["name"]: t["function"]["parameters"] for t in self.tools_openai}


def with_nonce_at_start(messages: list[dict[str, Any]], nonce: str) -> list[dict[str, Any]]:
    """Requests without tools: put the nonce at the very start of the first message."""
    msgs = copy.deepcopy(messages)
    first = msgs[0]
    if isinstance(first.get("content"), str):
        first["content"] = nonce_line(nonce) + first["content"]
    else:
        first["content"] = [{"type": "text", "text": nonce_line(nonce)}] + list(first.get("content") or [])
    return msgs
