"""The forms tap (forms/tap.py): its tools, the upload folder, the site guard and the screenshot. A fake child stands in
for chrome-devtools-mcp, as in test_web_tap.py, so no browser and no node are needed."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import test_web_tap
from forms import arms
from forms import tap as forms_tap

SITE = "http://127.0.0.1:50123"


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[test_web_tap.Rig]:
    tools = {
        **test_web_tap.CHILD_TOOLS,
        "fill_form": {"elements": {"type": "array"}},
        "upload_file": {"uid": {"type": "string"}, "filePaths": {"type": "array"}},
        "take_screenshot": {
            "format": {"type": "string"},
            "quality": {"type": "number"},
            "filePath": {"type": "string"},
        },
    }
    monkeypatch.setattr(test_web_tap, "TAP", arms.TAP)
    monkeypatch.setattr(test_web_tap, "CHILD_TOOLS", tools)
    rig = test_web_tap.Rig(tmp_path)
    yield rig
    for client in rig.clients:
        if client.proc.poll() is None:
            client.proc.kill()


def start(rig: test_web_tap.Rig, workspace: Path) -> test_web_tap.Client:
    workspace.mkdir(exist_ok=True)
    return rig.start("dom", **{forms_tap.UPLOAD_ROOT_ENV: str(workspace), forms_tap.SITE_ENV: SITE})


def refused(result: dict[str, Any]) -> bool:
    return result.get("isError") is True and test_web_tap.text_of(result).startswith("Refused:")


def test_the_tools_are_the_web_arm_and_three_for_forms(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    tools = start(rig, tmp_path / "ws").tools()
    assert [t["name"] for t in tools] == [*test_web_tap.tap.DOM_TOOLS, "fill_form", "upload_file", "take_screenshot"]
    assert list(forms_tap.TOOLS) == [t["name"] for t in tools]
    shot = next(t for t in tools if t["name"] == "take_screenshot")
    assert "filePath" not in shot["inputSchema"]["properties"]


def test_evaluate_script_and_click_at_stay_out(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    client = start(rig, tmp_path / "ws")
    assert refused(client.call("evaluate_script", function="() => 1"))
    assert refused(client.call("click_at", x=1, y=2))
    assert rig.child_call_names() == []


def test_a_file_in_the_workspace_can_be_uploaded(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    client = start(rig, tmp_path / "ws")
    path = tmp_path / "ws" / "cv.pdf"
    path.write_bytes(b"x")
    assert not client.call("upload_file", uid="1_2", filePaths=[str(path)]).get("isError")
    assert not client.call("upload_file", uid="1_2", filePaths=["cv.pdf"]).get("isError")
    assert rig.child_call_names() == ["upload_file", "upload_file"]


def test_a_file_elsewhere_is_refused(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    client = start(rig, tmp_path / "ws")
    (tmp_path / "secret").write_text("x")
    os.symlink(tmp_path / "secret", tmp_path / "ws" / "link.pdf")
    for path in (str(tmp_path / "secret"), "../secret", "/etc/hosts", "link.pdf"):
        assert refused(client.call("upload_file", uid="1_2", filePaths=[path])), path
    assert refused(client.call("upload_file", uid="1_2", filePaths=[]))
    assert refused(client.call("upload_file", uid="1_2", filePaths="/etc/hosts"))
    assert rig.child_call_names() == []
    assert [e["status"] for e in rig.log_entries() if e.get("event") == "call"] == ["refused"] * 6


def test_a_run_without_a_workspace_uploads_nothing() -> None:
    assert forms_tap.upload_refusal(["a.pdf"], None) == "there is no upload folder in this run"


def test_only_the_site_can_be_opened(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    client = start(rig, tmp_path / "ws")
    assert not client.call("navigate_page", url=f"{SITE}/forum/").get("isError")
    assert not client.call("new_page", url=f"{SITE}/shop/cart").get("isError")
    assert not client.call("navigate_page", type="back").get("isError")
    for url in (
        "https://www.example.com/",
        "http://127.0.0.1:50124/",
        "http://127.0.0.1:8500/v1/models",
        "about:blank",
    ):
        assert refused(client.call("navigate_page", url=url)), url
        assert refused(client.call("new_page", url=url)), url
    assert rig.child_call_names() == ["navigate_page", "new_page", "navigate_page"]


def test_a_screenshot_is_one_jpeg_and_nothing_is_written(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    client = start(rig, tmp_path / "ws")
    result = client.call("take_screenshot")
    assert [b["type"] for b in result["content"]] == ["image"] and result["content"][0]["mimeType"] == "image/jpeg"
    assert rig.child_calls()[-1] == {"name": "take_screenshot", "arguments": {"format": "jpeg", "quality": 70}}
    assert refused(client.call("take_screenshot", filePath="/tmp/page.png"))
    assert client.call("take_screenshot", format="png")["content"][0]["mimeType"] == "image/jpeg"


def test_the_text_tools_still_get_urls_cut_to_hosts(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    text = test_web_tap.text_of(start(rig, tmp_path / "ws").call("take_snapshot"))
    assert 'url="www.example.com"' in text and "sxsrf" not in text


def test_every_call_is_logged(rig: test_web_tap.Rig, tmp_path: Path) -> None:
    client = start(rig, tmp_path / "ws")
    client.call("fill_form", elements=[{"uid": "1_2", "value": "x"}])
    client.close()
    calls = [e for e in rig.log_entries() if e.get("event") == "call"]
    assert [(c["tool"], c["status"]) for c in calls] == [("fill_form", "ok")]
