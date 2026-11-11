#!/usr/bin/env python3
"""The forms tap: the web suite's tap in its ``dom`` mode, with the three tools that forms need.

    python3 tap.py --mode dom --log calls.jsonl --browser-url http://127.0.0.1:9222

It serves web/tap.py's 13 tools of chrome-devtools-mcp, plus:

- ``fill_form``: several fields in one call;
- ``upload_file``: only files under ``$FORMS_TAP_UPLOAD_ROOT``, the run's workspace;
- ``take_screenshot``: the page as an image, which the picture grid cannot do without. It is always a JPEG, and its
  ``filePath`` is taken out, so nothing is written to disk.

``navigate_page`` and ``new_page`` open only ``$FORMS_TAP_SITE``, the run's fixture site, so a run stays on this
computer. Everything else (the log, the refusals, the child's life) is web/tap.py's. Like it, this file uses the
standard library only.

melo hides a password the person gives in their message: the model reads and types the handle ``[password-1]``, and only
melo's own browser tool types the value. These runs reach Chrome through this tap instead, so the tap does that step:
``fill``, ``fill_form`` and ``type_text`` get ``$FORMS_TAP_PASSWORD`` in place of the handle, and the log keeps the handle.
"""

from __future__ import annotations

import os
import re
import sys
import urllib.parse
from typing import Any

AGENT_ABILITY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AGENT_ABILITY not in sys.path:
    sys.path.insert(0, AGENT_ABILITY)

from web import tap as base  # noqa: E402

UPLOAD_ROOT_ENV = "FORMS_TAP_UPLOAD_ROOT"
SITE_ENV = "FORMS_TAP_SITE"
PASSWORD_ENV = "FORMS_TAP_PASSWORD"
PASSWORD_HANDLE = "[password-1]"
# Newer melo builds number the handle with six digits picked per process (``[password-482913]``); both forms are taken.
PASSWORD_HANDLE_RE = re.compile(r"\[password-\d+\]")
TYPING_TOOLS = ("fill", "fill_form", "type_text")
EXTRA_TOOLS = ("fill_form", "upload_file", "take_screenshot")
TOOLS = (*base.DOM_TOOLS, *EXTRA_TOOLS)
JPEG_QUALITY = 70
# chrome-devtools-mcp reads files only under the system's temp folder unless told otherwise, and a run's workspace is not
# there. upload_refusal keeps the same limit, to the run's workspace.
CHILD_FLAGS = ("--allow-unrestricted-paths",)


def origin(url: Any) -> tuple[str, str, int | None] | None:
    try:
        parts = urllib.parse.urlsplit(str(url))
        return parts.scheme, parts.hostname or "", parts.port
    except ValueError:
        return None


def upload_refusal(paths: Any, root: str | None) -> str | None:
    """Why an upload_file call may not go on, or None. Every path must be a file under the workspace."""
    if not root:
        return "there is no upload folder in this run"
    if not isinstance(paths, list) or not paths:
        return "filePaths must be a list of paths"
    folder = os.path.realpath(root)
    for path in paths:
        real = os.path.realpath(os.path.join(folder, str(path)))
        if os.path.commonpath([folder, real]) != folder:
            return f"only files in {root} can be uploaded"
    return None


def with_password(value: Any, password: str) -> Any:
    """*value*, a tool's arguments, with the handle in its strings replaced by *password*."""
    if isinstance(value, str):
        return PASSWORD_HANDLE_RE.sub(lambda _: password, value)
    if isinstance(value, dict):
        return {k: with_password(v, password) for k, v in value.items()}
    if isinstance(value, list):
        return [with_password(v, password) for v in value]
    return value


class FormsTap(base.Tap):
    def _call_dom(self, name: str, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        if name in ("navigate_page", "new_page") and "url" in args:
            site = os.environ.get(SITE_ENV)
            if site and origin(args["url"]) != origin(site):
                raise base.Refused(f"only {site} can be opened in this task")
        if name == "upload_file":
            problem = upload_refusal(args.get("filePaths"), os.environ.get(UPLOAD_ROOT_ENV))
            if problem:
                raise base.Refused(problem)
        if name == "take_screenshot":
            return self._screenshot(args, detail)
        password = os.environ.get(PASSWORD_ENV)
        if password and name in TYPING_TOOLS:
            args = with_password(args, password)
        return super()._call_dom(name, args, detail)

    def _screenshot(self, args: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
        assert self.child is not None
        if "filePath" in args:
            raise base.Refused("take_screenshot.filePath is not available")
        raw = self.child.call_tool("take_screenshot", {**args, "format": "jpeg", "quality": JPEG_QUALITY})
        if raw.get("isError"):
            return {"content": raw.get("content", []), "isError": True}
        image = base.first_image(raw)
        if image is None:
            return base.text_result("Error: the browser returned no screenshot", True)
        detail.update(image_bytes_raw=len(image.get("data", "")) * 3 // 4)
        return {"content": [image]}


def install() -> None:
    """Make web/tap.py serve this suite's tools: its ``main`` builds the tap and reads these names when it runs."""
    base.DOM_TOOLS = TOOLS
    base.DOM_DROPPED_PARAMS = {**base.DOM_DROPPED_PARAMS, "take_screenshot": ("filePath",)}
    base.COMMON_CHILD_FLAGS = (*base.COMMON_CHILD_FLAGS, *CHILD_FLAGS)
    base.Tap = FormsTap


def main(argv: list[str] | None = None) -> int:
    install()
    return base.main(argv)


if __name__ == "__main__":
    sys.exit(main())
