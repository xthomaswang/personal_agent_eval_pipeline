"""The forms suite against the real thing, opt-in. CPU only: no model server, no window on the screen.

    cd eval
    FORMS_REAL_CHROME=1 uv run pytest -q agent_ability/tests/test_forms_real.py -k chrome  # a scripted client, no melo
    FORMS_REAL_MELO=1 uv run pytest -q agent_ability/tests/test_forms_real.py -k melo      # real melo, a stub model

The first drives every task through the forms tap on a headless Chrome and proves that the pages, the tools and the
checks fit together: each task can be done with the tools the agent has, and the site's records pass it. The second runs
the whole driver with the pinned melo, the real proxy and Chrome, with a local stub in the model's place.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import test_web_tap
from forms import arms, fixture, tasks
from forms import run as forms_run
from forms import tap as forms_tap

REAL_CHROME = os.environ.get("FORMS_REAL_CHROME") == "1"
REAL_MELO = os.environ.get("FORMS_REAL_MELO") == "1"


def text_of(result: dict[str, Any]) -> str:
    return "\n".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")


def find_uid(snapshot: str, role: str, name: str) -> str:
    found = re.search(rf'uid=(\S+) {role} "{re.escape(name)}', snapshot)
    assert found, f"no {role} {name!r} in:\n{snapshot}"
    return found.group(1)


class Driver:
    """What an agent would do, written out: read the page, then act on what it shows."""

    def __init__(self, client: test_web_tap.Client) -> None:
        self.client = client

    def snapshot(self) -> str:
        return text_of(self.client.call("take_snapshot"))

    def click(self, role: str, name: str) -> None:
        result = self.client.call("click", uid=find_uid(self.snapshot(), role, name))
        assert not result.get("isError"), result

    def fill(self, *pairs: tuple[str, str, str]) -> None:
        snapshot = self.snapshot()
        elements = [{"uid": find_uid(snapshot, role, name), "value": value} for role, name, value in pairs]
        result = self.client.call("fill_form", elements=elements)
        assert not result.get("isError"), result


def sign_in(d: Driver, account: tuple[str, str]) -> None:
    d.fill(("textbox", "Username", account[0]), ("textbox", "Password", account[1]))
    d.click("button", "Sign in")


def solve_signin(d: Driver, _workspace: Path) -> str:
    sign_in(d, fixture.TASKMERE_ACCOUNT)
    assert "Signed in as" in d.snapshot()
    return "Done."


def apply_step_two(d: Driver, school: str, degree: str, date: str, heard: str, gender: str, sponsor: str) -> None:
    d.fill(
        ("textbox", "School", school),
        ("combobox", "Degree", degree),
        ("textbox", "Graduation date", date),
        ("combobox", "How did you hear about us?", heard),
        ("combobox", "Gender", gender),
    )
    snapshot = d.snapshot()
    radios = re.findall(r'uid=(\S+) radio "(Yes|No)"', snapshot)
    assert [name for _, name in radios] == ["Yes", "No", "Yes", "No"], snapshot
    d.client.call("click", uid=radios[0][0])
    d.client.call("click", uid=radios[2 if sponsor == "yes" else 3][0])
    d.click("checkbox", "I consent")
    d.click("button", "Submit application")
    assert "Application received" in d.snapshot()


def solve_job_two_step(d: Driver, _workspace: Path) -> str:
    d.click("link", "Apply for this job")
    d.fill(
        ("textbox", "Legal first name", "Katherine"),
        ("textbox", "Legal last name", "Lin"),
        ("textbox", "Preferred first name", "Kate"),
        ("textbox", "Email", "kate.lin@example.com"),
        ("textbox", "Phone", "(312) 555-0147"),
        ("textbox", "Current location", "Evanston"),
        ("textbox", "Website / portfolio", "https://katelin.example.com"),
    )
    d.click("button", "Continue")
    apply_step_two(
        d, "Lakeshore State University", "Master's", "05/2019", "Employee referral", "Decline to self-identify", "no"
    )
    return "Done."


def solve_job_upload(d: Driver, workspace: Path) -> str:
    d.click("link", "Apply for this job")
    d.fill(
        ("textbox", "Legal first name", "Nadia"),
        ("textbox", "Legal last name", "Haddad"),
        ("textbox", "Email", "nadia.haddad@example.com"),
        ("textbox", "Phone", "(617) 555-0182"),
        ("textbox", "Current location", "Somerville"),
    )
    result = d.client.call(
        "upload_file", uid=find_uid(d.snapshot(), "button", "Attach"), filePaths=[str(workspace / tasks.RESUME_NAME)]
    )
    assert not result.get("isError"), result
    assert tasks.RESUME_NAME in d.snapshot()
    d.click("button", "Continue")
    apply_step_two(d, "Calder Bay University", "PhD", "12/2024", "University career fair", "Female", "yes")
    return "Done."


def solve_checkout(d: Driver, _workspace: Path) -> str:
    d.click("link", "Check out")
    d.fill(
        ("textbox", "Email", "priya.raman@example.com"),
        ("textbox", "First name", "Priya"),
        ("textbox", "Last name", "Raman"),
        ("textbox", "Address", "418 Alder Court"),
        ("textbox", "Apartment", "Apartment 3B"),
        ("textbox", "City", "Portland"),
        ("combobox", "State", "Oregon"),
        ("textbox", "ZIP code", "97205"),
        ("textbox", "Phone", "(503) 555-0139"),
    )
    d.click("checkbox", "Email me with news")
    d.click("radio", "Express")
    d.click("button", "Continue to payment")
    d.fill(("textbox", "Discount code or gift card", fixture.GIFT_CODE))
    d.click("button", "Apply")
    assert "$124.00" in d.snapshot()
    d.click("radio", "Bank transfer")
    d.click("checkbox", "I agree")
    d.click("button", "Place order")
    assert "Thank you for your order" in d.snapshot()
    return "Done."


def solve_search(d: Driver, _workspace: Path) -> str:
    d.fill(
        ("searchbox", "Search the catalog", "hex bolt"),
        ("combobox", "Material", "Stainless steel A4"),
        ("combobox", "Length", "40 mm"),
    )
    d.click("button", "Search")
    d.click("link", "Hex bolt M8 x 40")
    found = re.search(r'"([A-Z]{2}-\d{5}-[A-Z])"', d.snapshot())
    assert found, d.snapshot()
    return f"FINAL ANSWER: {found.group(1)}"


def solve_checkin(d: Driver, _workspace: Path) -> str:
    d.click("link", "Sign in")
    sign_in(d, fixture.FORUM_ACCOUNT)
    d.click("radio", "Focused")
    d.fill(("textbox", "Note for today", "Thesis draft, chapter 2"))
    d.click("button", "Check in")
    assert "Checked in for today" in d.snapshot()
    return "Done."


def solve_settings(d: Driver, _workspace: Path) -> str:
    d.click("switch", "Product updates")
    d.click("switch", "Mobile push")
    d.fill(
        ("combobox", "Digest frequency", "Daily"),
        ("textbox", "From", "22:30"),
        ("textbox", "To", "07:00"),
        ("combobox", "Time zone", "(UTC+01:00) Berlin"),
    )
    d.click("button", "Save changes")
    assert "were saved" in d.snapshot()
    return "Done."


def solve_picture_grid(d: Driver, _workspace: Path) -> str:
    shot = d.client.call("take_screenshot")
    assert [b["type"] for b in shot["content"]] == ["image"] and shot["content"][0]["mimeType"] == "image/jpeg"
    for number in tasks.TILES_WITH_A_MUG:
        d.click("button", f"Photo {number}")
    d.click("button", "Use selected photos")
    assert "4 photos are on your listing" in d.snapshot()
    return "Done."


SOLUTIONS: dict[str, Callable[[Driver, Path], str]] = {
    "signin": solve_signin,
    "job_two_step": solve_job_two_step,
    "job_upload": solve_job_upload,
    "checkout": solve_checkout,
    "search": solve_search,
    "checkin": solve_checkin,
    "settings": solve_settings,
    "picture_grid": solve_picture_grid,
}


def test_every_task_has_a_scripted_solution() -> None:
    assert list(SOLUTIONS) == [t.id for t in tasks.TASKS]


@pytest.mark.skipif(not REAL_CHROME, reason="set FORMS_REAL_CHROME=1: starts the installed Chrome (headless)")
@pytest.mark.parametrize("task", tasks.TASKS, ids=lambda t: t.id)
def test_real_chrome_a_scripted_client_passes_the_task(
    task: tasks.Task, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_web_tap, "TAP", arms.TAP)
    workspace = tmp_path / "workspace"
    tasks.prepare_workspace(task, workspace)
    site = fixture.Site(tmp_path / "site")
    env = {
        **os.environ,
        **arms.web_arms.tap_env(),
        forms_tap.UPLOAD_ROOT_ENV: str(workspace),
        forms_tap.SITE_ENV: site.base_url,
    }
    try:
        with forms_run.forms_chrome(tmp_path / "chrome", site.url(task.start)) as browser:
            assert browser.meta["headless"] is True
            assert "--headless=new" in browser.meta["command"].split()
            argv = ["--mode", "dom", "--log", str(tmp_path / "tap.jsonl"), "--browser-url", browser.browser_url]
            client = test_web_tap.Client(argv, env)
            assert [t["name"] for t in client.tools()] == list(forms_tap.TOOLS)
            reply = SOLUTIONS[task.id](Driver(client), workspace)
            client.close()
    finally:
        site.stop()
    outcome = tasks.check(task, site.submissions, reply)
    assert outcome["passed"], (outcome, site.submissions)


# --- real melo, with a stub in the model's place ---


class StubModel:
    """An OpenAI-compatible server on a free port: plays a list of steps, and keeps every chat request.

    A step is a function of the latest tool result: it returns the reply's text, or a (tool, arguments) pair. A step of
    None hangs until the server stops, like a model that never answers. T-off's requests get no reasoning, as a server
    would behave; the others get a little of it, with reasoning tokens in the usage.
    """

    def __init__(self, steps: list[Callable[[str], Any] | None]) -> None:
        self.steps = steps
        self.requests: list[dict[str, Any]] = []
        self.done = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def send(self, value: dict[str, Any]) -> None:
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self.send({"object": "list", "data": [{"id": arms.DEFAULT_MODEL, "object": "model"}]})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path.endswith("/embeddings"):
                    items = body.get("input") if isinstance(body.get("input"), list) else [body.get("input")]
                    self.send(
                        {"object": "list", "data": [{"index": i, "embedding": [0.1] * 16} for i in range(len(items))]}
                    )
                    return
                owner.requests.append(body)
                reply = owner.reply(body)
                if reply is None:
                    owner.done.wait(120)
                    return
                self.send(reply)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def reply(self, body: dict[str, Any]) -> dict[str, Any] | None:
        if not body.get("tools"):  # a reflection or a note: no step of the script
            return self.message({"role": "assistant", "content": "ok"}, body)
        if not self.steps:
            return self.message({"role": "assistant", "content": "All done."}, body)
        step = self.steps.pop(0)
        if step is None:
            return None
        last = next((m for m in reversed(body["messages"]) if m.get("role") == "tool"), {})
        content = last.get("content")
        latest = content if isinstance(content, str) else json.dumps(content)
        action = step(latest)
        if isinstance(action, str):
            return self.message({"role": "assistant", "content": action}, body)
        name, arguments = action
        call = {
            "id": f"call_{len(self.requests)}",
            "type": "function",
            "function": {"name": f"{arms.TOOL_PREFIX}{name}", "arguments": json.dumps(arguments)},
        }
        return self.message({"role": "assistant", "content": None, "tool_calls": [call]}, body)

    @staticmethod
    def message(message: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        off = (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
        reasoning = 0 if off else 7
        if reasoning:
            message = {**message, "reasoning_content": "Let me look at the page."}
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
        }

    def stop(self) -> None:
        self.done.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def signin_steps() -> list[Callable[[str], Any] | None]:
    def elements(latest: str) -> tuple[str, dict[str, Any]]:
        fields = [
            {"uid": find_uid(latest, "textbox", "Username"), "value": fixture.TASKMERE_ACCOUNT[0]},
            {"uid": find_uid(latest, "textbox", "Password"), "value": forms_tap.PASSWORD_HANDLE},
        ]
        return "fill_form", {"elements": fields}

    return [
        lambda _: ("take_snapshot", {}),
        elements,
        lambda _: ("take_snapshot", {}),
        lambda latest: ("click", {"uid": find_uid(latest, "button", "Sign in")}),
        lambda _: "Signed in.",
    ]


@pytest.fixture
def stub_factory() -> Iterator[Callable[[list[Any]], StubModel]]:
    made: list[StubModel] = []

    def make(steps: list[Any]) -> StubModel:
        made.append(StubModel(steps))
        return made[-1]

    yield make
    for stub in made:
        stub.stop()


def study(tmp_path: Path, stub: StubModel, **kw: Any) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "study"
    kw.setdefault("arm_names", ("T-as-is", "T-off"))
    kw.setdefault("chosen", [tasks.BY_ID["signin"]])
    code = forms_run.run_study(out, repeats=1, gpu_lock=False, base_url=stub.url, **kw)
    return code, json.loads((out / "forms.json").read_text())


@pytest.mark.skipif(not REAL_MELO, reason="set FORMS_REAL_MELO=1: runs the pinned melo and Chrome, with a stub model")
def test_real_melo_signs_in_on_both_arms_and_only_t_off_turns_thinking_off(
    tmp_path: Path, stub_factory: Callable[[list[Any]], StubModel]
) -> None:
    stub = stub_factory(signin_steps() + signin_steps())  # the same script for both arms
    code, state = study(tmp_path, stub)
    assert code == 0 and state["complete"]
    runs = {r["arm"]: r for r in state["runs"].values()}
    for record in runs.values():
        assert record["verdict"] == "pass", record
        assert record["end_reason"] == "completed" and record["melo_status"] == "completed"
        assert record["headless"] is True and "--headless=new" in record["chrome_command"].split()
        assert record["melo_chrome_profile"] is False
        assert record["arm_applied"] is True
        assert record["model_share"] is not None and record["n_tool_tap"] >= 4
        assert record["refused_calls"] == 0, record
        assert record["foreign_tools"] == [], record
    assert runs["T-off"]["thinking_off_calls"] == runs["T-off"]["n_calls"]
    assert runs["T-off"]["reasoning_chars"] == 0
    assert runs["T-as-is"]["thinking_off_calls"] == 0
    assert runs["T-as-is"]["reasoning_tokens"] == 7 * runs["T-as-is"]["n_calls"]
    kwargs = [r.get("chat_template_kwargs") for r in stub.requests]
    assert {json.dumps(k, sort_keys=True) for k in kwargs} == {"null", '{"enable_thinking": false}'}
    report = (tmp_path / "study" / "report.md").read_text()
    assert "T-off minus T-as-is" in report and "reasoning tokens" in report


@pytest.mark.skipif(not REAL_MELO, reason="set FORMS_REAL_MELO=1: runs the pinned melo and Chrome, with a stub model")
def test_real_melo_a_question_ends_the_run_as_a_failure(
    tmp_path: Path, stub_factory: Callable[[list[Any]], StubModel]
) -> None:
    stub = stub_factory([lambda _: "Which account should I use?"])
    code, state = study(tmp_path, stub, arm_names=("T-off",))
    assert code == 0
    (record,) = state["runs"].values()
    assert record["verdict"] == "fail" and record["end_reason"] == "asked_person"


@pytest.mark.skipif(not REAL_MELO, reason="set FORMS_REAL_MELO=1: runs the pinned melo and Chrome, with a stub model")
def test_real_melo_a_model_that_stops_answering_ends_the_run_in_bounded_time(
    tmp_path: Path, stub_factory: Callable[[list[Any]], StubModel]
) -> None:
    stub = stub_factory([lambda _: ("take_snapshot", {}), None])
    started = time.monotonic()
    code, state = study(tmp_path, stub, arm_names=("T-off",), wall_cap_s=40.0)
    assert code == 0 and time.monotonic() - started < 90
    (record,) = state["runs"].values()
    assert record["verdict"] == "fail" and record["end_reason"] == "timeout", record
    assert record["wall_s"] < 40


@pytest.mark.skipif(not REAL_MELO, reason="set FORMS_REAL_MELO=1: runs the pinned melo and Chrome, with a stub model")
def test_real_melo_a_screenshot_reaches_the_model_as_an_image(
    tmp_path: Path, stub_factory: Callable[[list[Any]], StubModel]
) -> None:
    stub = stub_factory([lambda _: ("take_screenshot", {}), lambda _: "I see the tiles."])
    code, state = study(tmp_path, stub, arm_names=("T-off",), chosen=[tasks.BY_ID["picture_grid"]])
    assert code == 0
    (record,) = state["runs"].values()
    assert record["n_images"] == 1 and record["tools"] == {"take_screenshot": 1}
    parts = [p for m in stub.requests[-1]["messages"] if isinstance(m.get("content"), list) for p in m["content"]]
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    assert len(urls) == 1 and urls[0].startswith("data:image/jpeg;base64,")


@pytest.mark.skipif(not REAL_MELO, reason="set FORMS_REAL_MELO=1: runs the pinned melo and Chrome, with a stub model")
def test_real_melo_sigterm_ends_the_process_tree_with_exit_3_and_a_rerun_completes(
    tmp_path: Path, stub_factory: Callable[[list[Any]], StubModel]
) -> None:
    stub = stub_factory([None])
    out = tmp_path / "study"
    command = [
        sys.executable, str(arms.HERE / "run.py"), "--run-dir", str(out), "--no-gpu-lock", "--base-url", stub.url,
        "--tasks", "signin", "--arms", "T-off", "--repeats", "1",
    ]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if not k.startswith("NIGHT_")}
    driver = subprocess.Popen(command, env=env, cwd=arms.AGENT_ABILITY.parent, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 40
    while not stub.requests and time.monotonic() < deadline:
        time.sleep(0.2)
    assert stub.requests, "melo never reached the stub"
    driver.send_signal(signal.SIGTERM)
    assert driver.wait(timeout=40) == forms_run.STOPPED
    time.sleep(1)
    assert [p.command for p in forms_run.chrome.process_table() if str(out) in p.command] == []
    run_id = "001__signin__r1__T-off"
    assert (out / "runs" / f"{run_id}.dropped-1").is_dir()
    assert not (out / "ws" / ".lock").exists()
    stub.steps = signin_steps()
    assert subprocess.run(command, env=env, cwd=arms.AGENT_ABILITY.parent, timeout=60, check=False).returncode == 0
    assert json.loads((out / "runs" / run_id / "record.json").read_text())["verdict"] == "pass"


@pytest.mark.skipif(not REAL_MELO, reason="set FORMS_REAL_MELO=1: runs the pinned melo")
def test_real_melo_refuses_extra_body_as_a_setting(tmp_path: Path) -> None:
    """Why T-off needs a config.toml: if this ever works, arms.py can say so more simply."""
    done = subprocess.run(
        [str(arms.LOCALMELO / ".venv" / "bin" / "melo"), "-X", 'vllm.extra_body={"a":1}', "config", "keys"],
        capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": str(tmp_path)},
    )  # fmt: skip
    assert done.returncode == 2 and "unknown setting 'vllm.extra_body'" in done.stderr
