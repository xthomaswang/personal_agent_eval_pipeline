"""The forms fixture site and the checks (forms/fixture.py, forms/tasks.py), over plain http: no Chrome, no model.

``Browser`` is a client with cookies, and ``solve`` the right submissions of each task. test_forms_run.py uses both as
the fake agent. The same tasks through the tools the agent has are in test_forms_real.py.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

import pytest
from forms import fixture, tasks

BOUNDARY = "----formsboundary"
JOB_DA = {
    "first_name": "Katherine", "last_name": "Lin", "preferred_name": "Kate", "email": "kate.lin@example.com",
    "phone_country": "+1", "phone": "(312) 555-0147", "city": "Evanston", "linkedin": "",
    "website": "https://katelin.example.com",
}  # fmt: skip
JOB_DA_2 = {
    "school": "Lakeshore State University", "degree": "Master's", "grad_date": "05/2019", "authorized": "yes",
    "sponsorship": "no", "hear_about": "Employee referral", "gender": "Decline to self-identify", "consent": "on",
    "action": "submit",
}  # fmt: skip
JOB_PM = {
    "first_name": "Nadia", "last_name": "Haddad", "preferred_name": "", "email": "nadia.haddad@example.com",
    "phone_country": "+1", "phone": "(617) 555-0182", "city": "Somerville", "linkedin": "", "website": "",
}  # fmt: skip
JOB_PM_2 = {
    "school": "Calder Bay University", "degree": "PhD", "grad_date": "12/2024", "authorized": "yes",
    "sponsorship": "yes", "hear_about": "University career fair", "gender": "Female", "consent": "on",
    "action": "submit",
}  # fmt: skip
SHIPPING = {
    "email": "priya.raman@example.com", "first_name": "Priya", "last_name": "Raman", "company": "",
    "address": "418 Alder Court", "address2": "Apartment 3B", "city": "Portland", "state": "Oregon", "zip": "97205",
    "phone": "(503) 555-0139", "delivery": "express",
}  # fmt: skip


class Browser:
    """An http client that keeps cookies, never through a system proxy, and does not raise on an error status."""

    def __init__(self, site: fixture.Site) -> None:
        self.site = site
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(CookieJar())
        )

    def _open(self, request: urllib.request.Request) -> tuple[int, str]:
        try:
            with self.opener.open(request, timeout=10) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def get(self, path: str) -> tuple[int, str]:
        return self._open(urllib.request.Request(self.site.url(path)))

    def post(self, path: str, fields: Mapping[str, str]) -> tuple[int, str]:
        data = urllib.parse.urlencode(fields).encode()
        return self._open(urllib.request.Request(self.site.url(path), data=data))

    def post_multipart(
        self, path: str, fields: Mapping[str, str], name: str, filename: str, data: bytes
    ) -> tuple[int, str]:
        parts = [
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields.items()
        ]
        head = f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        head += "Content-Type: application/pdf\r\n\r\n"
        body = b"".join(parts) + head.encode() + data + f"\r\n--{BOUNDARY}--\r\n".encode()
        request = urllib.request.Request(self.site.url(path), data=body)
        request.add_header("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
        return self._open(request)


def solve(task_id: str, browser: Browser) -> str:
    """The right submissions of a task, and the reply that goes with them."""
    if task_id == "signin":
        browser.post("/taskmere/signin", dict(zip(("username", "password"), fixture.TASKMERE_ACCOUNT, strict=True)))
    elif task_id == "job_two_step":
        browser.post("/careers/apply/DA-2047/1", JOB_DA)
        browser.post("/careers/apply/DA-2047/2", JOB_DA_2)
    elif task_id == "job_upload":
        data = tasks.BY_ID["job_upload"].workspace_files[tasks.RESUME_NAME]
        browser.post_multipart("/careers/apply/PM-3310/1", JOB_PM, "resume", tasks.RESUME_NAME, data)
        browser.post("/careers/apply/PM-3310/2", JOB_PM_2)
    elif task_id == "checkout":
        browser.post("/shop/checkout", SHIPPING)
        browser.post("/shop/checkout/payment", {"gift_code": fixture.GIFT_CODE, "action": "apply"})
        browser.post("/shop/checkout/payment", {"payment": "bank_transfer", "terms": "on", "action": "place"})
    elif task_id == "search":
        return f"The part number is below.\nFINAL ANSWER: {tasks.TARGET_PART.mpn}"
    elif task_id == "checkin":
        browser.post("/forum/signin", dict(zip(("username", "password"), fixture.FORUM_ACCOUNT, strict=True)))
        browser.post("/forum/checkin", {"mood": "Focused", "note": "Thesis draft, chapter 2"})
    elif task_id == "settings":
        browser.get("/taskmere/open")
        browser.post("/taskmere/settings/notifications", dict(tasks.BY_ID["settings"].expected))
    elif task_id == "picture_grid":
        browser.post("/market/listing/photos", {"selected": ",".join(map(str, tasks.TILES_WITH_A_MUG))})
    else:
        raise ValueError(task_id)
    return "Done."


@pytest.fixture
def site(tmp_path: Path) -> Iterator[fixture.Site]:
    running = fixture.Site(tmp_path / "site")
    yield running
    running.stop()


@pytest.fixture
def browser(site: fixture.Site) -> Browser:
    return Browser(site)


def rows(site: fixture.Site, kind: str) -> list[dict[str, Any]]:
    return [s for s in site.submissions if s["kind"] == kind]


# --- the site ---


def test_the_site_is_on_loopback_on_a_free_port_outside_the_reserved_ones(site: fixture.Site) -> None:
    assert site.base_url.startswith("http://127.0.0.1:")
    assert not any(site.port in ports for ports in fixture.RESERVED_PORTS)
    assert any(8500 in ports for ports in fixture.RESERVED_PORTS)
    assert all(any(p in ports for ports in fixture.RESERVED_PORTS) for p in (8400, 8799, 8955, 11434, 11500))


def test_a_reserved_port_is_given_back(monkeypatch: pytest.MonkeyPatch) -> None:
    given = iter([8500, 49999])
    closed = []

    class FakeServer:
        def __init__(self, *_: Any) -> None:
            self.server_address = ("127.0.0.1", next(given))
            self.daemon_threads = False

        def server_close(self) -> None:
            closed.append(self.server_address[1])

        def serve_forever(self, **_: Any) -> None:
            pass

    monkeypatch.setattr(fixture, "ThreadingHTTPServer", FakeServer)
    assert fixture.Site().port == 49999 and closed == [8500]


@pytest.mark.parametrize("task", tasks.TASKS, ids=lambda t: t.id)
def test_each_task_starts_on_a_page_the_site_serves(task: tasks.Task, browser: Browser) -> None:
    status, text = browser.get(task.start)
    assert status == 200 and "<title>" in text and "{{" not in text


@pytest.mark.parametrize(
    "path",
    [
        "/taskmere/signin", "/careers/apply/DA-2047/1", "/careers/apply/PM-3310/1", "/shop/cart", "/shop/checkout",
        "/supply/search?q=hex", "/supply/item/FS-10243", "/forum/", "/forum/signin", "/market/listing/photos",
        "/static/style.css", "/static/site.js", "/static/tiles/tile-01.svg",
    ],
)  # fmt: skip
def test_every_page_renders_with_no_mark_left(path: str, browser: Browser) -> None:
    status, text = browser.get(path)
    assert status == 200 and "{{" not in text


def test_a_file_outside_the_pages_or_of_another_type_is_not_served(browser: Browser) -> None:
    assert browser.get("/static/../fixture.py")[0] == 404
    assert browser.get("/static/signin.html")[0] == 404
    assert browser.get("/nothing")[0] == 404


def test_every_picture_is_there_and_names_nothing(browser: Browser) -> None:
    status, text = browser.get("/market/listing/photos")
    assert status == 200
    assert re.findall(r'aria-label="(Photo \d+)"', text) == [f"Photo {n}" for n in range(1, fixture.TILE_COUNT + 1)]
    assert "mug" not in text.lower() and "captcha" not in text.lower() and "robot" not in text.lower()
    for n in range(1, fixture.TILE_COUNT + 1):
        status, picture = browser.get(f"/static/tiles/tile-{n:02d}.svg")
        assert status == 200 and "<svg" in picture and "mug" not in picture.lower()


def test_a_wrong_password_is_refused_and_no_password_is_kept(site: fixture.Site, browser: Browser) -> None:
    _, text = browser.post("/taskmere/signin", {"username": "rowan.ellis", "password": "guess-1234"})
    assert "do not match" in text
    assert rows(site, "taskmere.signin")[0]["status"] == "refused"
    browser.post("/taskmere/signin", dict(zip(("username", "password"), fixture.TASKMERE_ACCOUNT, strict=True)))
    assert "Signed in as rowan.ellis" in browser.get("/taskmere/home")[1]
    logged = (site.out / "submissions.jsonl").read_text()
    assert "guess-1234" not in logged and fixture.TASKMERE_ACCOUNT[1] not in logged


def test_a_page_behind_a_sign_in_sends_you_there(browser: Browser) -> None:
    assert "Sign in to Taskmere" in browser.get("/taskmere/settings/notifications")[1]
    assert "Sign in to check in" in browser.get("/forum/")[1]


def test_each_run_has_its_own_state(tmp_path: Path) -> None:
    first = fixture.Site()
    solve("settings", Browser(first))
    first.stop()
    second = fixture.Site()
    assert second.submissions == [] and second.settings == fixture.DEFAULT_SETTINGS
    second.stop()


def test_step_one_needs_its_fields_and_keeps_what_was_typed(site: fixture.Site, browser: Browser) -> None:
    _, text = browser.post("/careers/apply/DA-2047/1", {"first_name": "Katherine"})
    assert "Please fill in: Legal last name, Email, Phone." in text and '"first_name": "Katherine"' in text
    assert site.submissions == []


def test_step_two_needs_step_one_first(browser: Browser) -> None:
    assert "Your details" in browser.get("/careers/apply/DA-2047/2")[1]


def test_an_application_without_its_answers_is_refused(site: fixture.Site, browser: Browser) -> None:
    browser.post("/careers/apply/DA-2047/1", JOB_DA)
    _, text = browser.post("/careers/apply/DA-2047/2", {"action": "submit", "authorized": "yes"})
    assert "Please answer: Visa sponsorship, Consent to data processing." in text
    assert [s["status"] for s in site.submissions] == ["refused"]


def test_a_draft_submits_nothing(site: fixture.Site, browser: Browser) -> None:
    browser.post("/careers/apply/DA-2047/1", JOB_DA)
    _, text = browser.post("/careers/apply/DA-2047/2", {**JOB_DA_2, "action": "draft"})
    assert "Your draft was saved" in text and site.submissions == []


def test_a_second_application_is_a_duplicate(site: fixture.Site, browser: Browser) -> None:
    solve("job_two_step", browser)
    browser.post("/careers/apply/DA-2047/2", {**JOB_DA_2, "school": "Another University"})
    assert [s["status"] for s in site.submissions] == ["accepted", "duplicate"]
    assert tasks.check(tasks.BY_ID["job_two_step"], site.submissions)["passed"]


def test_the_upload_arrives_byte_for_byte(site: fixture.Site, browser: Browser) -> None:
    data = bytes(range(256)) * 40 + b"\r\n--not-the-boundary\r\n" + b"\x00\xff\r\n"
    browser.post_multipart("/careers/apply/PM-3310/1", JOB_PM, "resume", "resume.pdf", data)
    browser.post("/careers/apply/PM-3310/2", JOB_PM_2)
    (row,) = site.submissions
    assert row["files"] == [
        {"field": "resume", "name": "resume.pdf", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    ]


def test_the_resume_is_required_and_of_the_right_type(site: fixture.Site, browser: Browser) -> None:
    _, text = browser.post("/careers/apply/PM-3310/1", JOB_PM)
    assert "Resume/CV" in text and "Please fill in" in text
    _, text = browser.post_multipart("/careers/apply/PM-3310/1", JOB_PM, "resume", "photo.png", b"x")
    assert "must be a PDF, DOC or DOCX" in text
    assert "Background and eligibility" not in text


def test_a_job_without_a_resume_field_has_none(browser: Browser) -> None:
    text = browser.get("/careers/apply/DA-2047/1")[1]
    assert "Resume/CV" not in text and "multipart" not in text
    assert 'enctype="multipart/form-data"' in browser.get("/careers/apply/PM-3310/1")[1]


def test_the_checkout_totals_the_order(site: fixture.Site, browser: Browser) -> None:
    assert "$137.00" in browser.get("/shop/cart")[1]
    assert "Shipping" in browser.get("/shop/checkout/payment")[1]  # no address yet: back to shipping
    browser.post("/shop/checkout", SHIPPING)
    assert "$149.00" in browser.get("/shop/checkout/payment")[1]
    _, text = browser.post("/shop/checkout/payment", {"gift_code": "NOPE", "action": "apply"})
    assert "not valid" in text
    _, text = browser.post("/shop/checkout/payment", {"gift_code": fixture.GIFT_CODE, "action": "apply"})
    assert "$124.00" in text and "was applied" in text
    _, text = browser.post("/shop/checkout/payment", {"action": "place"})
    assert "Please complete: Payment method, Terms of sale." in text
    assert [s["status"] for s in site.submissions] == ["refused"]


def test_a_code_typed_but_not_applied_counts_when_the_order_is_placed(site: fixture.Site, browser: Browser) -> None:
    browser.post("/shop/checkout", SHIPPING)
    fields = {"gift_code": fixture.GIFT_CODE, "payment": "bank_transfer", "terms": "on", "action": "place"}
    assert "$124.00" in browser.post("/shop/checkout/payment", fields)[1]
    assert site.submissions[-1]["fields"]["total"] == "124.00"


def test_the_search_filters_and_only_the_item_shows_its_part_number(browser: Browser) -> None:
    _, text = browser.get("/supply/search?q=hex+bolt&material=Stainless+steel+A4&length=40")
    names = re.findall(r'href="/supply/item/([\w-]+)"', text)
    assert names == ["FS-10233", "FS-10243", "FS-10251"]
    assert not any(p.mpn in text for p in fixture.CATALOG)
    assert tasks.TARGET_PART.mpn in browser.get(f"/supply/item/{tasks.TARGET_PART.sku}")[1]
    assert sum(1 for p in fixture.CATALOG if (p.thread, p.length, p.material) == ("M8", 40, "Stainless steel A4")) == 2
    assert browser.get("/supply/item/NOPE")[0] == 404


def test_the_check_in_needs_a_sign_in_and_a_mood_and_happens_once(site: fixture.Site, browser: Browser) -> None:
    assert "Sign in to Harborline" in browser.post("/forum/checkin", {"mood": "Busy"})[1]
    assert site.submissions == []
    browser.post("/forum/signin", dict(zip(("username", "password"), fixture.FORUM_ACCOUNT, strict=True)))
    assert "Please choose" in browser.post("/forum/checkin", {})[1]
    assert "Checked in for today" in browser.post("/forum/checkin", {"mood": "Focused"})[1]
    assert "already checked in" in browser.post("/forum/checkin", {"mood": "Tired"})[1]
    assert [s["status"] for s in rows(site, "forum.checkin")] == ["refused", "accepted", "duplicate"]


def test_settings_show_what_was_saved_and_refuse_a_bad_time(site: fixture.Site, browser: Browser) -> None:
    browser.get("/taskmere/open")
    fields = {**fixture.DEFAULT_SETTINGS, "quiet_from": "25:00"}
    _, text = browser.post("/taskmere/settings/notifications", fields)
    assert "HH:MM" in text and site.settings == fixture.DEFAULT_SETTINGS
    _, text = browser.post("/taskmere/settings/notifications", {**fixture.DEFAULT_SETTINGS, "sms": "on"})
    assert "were saved" in text and site.settings["sms"] == "on"
    assert 'aria-checked="true" aria-labelledby="t_sms"' in text


def test_the_picker_refuses_nothing_chosen(site: fixture.Site, browser: Browser) -> None:
    assert "at least one" in browser.post("/market/listing/photos", {"selected": ""})[1]
    assert "3 photos" in browser.post("/market/listing/photos", {"selected": "1,2,2,3,x"})[1]
    assert site.submissions[-1]["fields"]["selected"] == [1, 2, 3]


# --- the checks ---


def test_the_tasks_are_the_eight_of_the_set() -> None:
    assert [t.id for t in tasks.TASKS] == [
        "signin", "job_two_step", "job_upload", "checkout", "search", "checkin", "settings", "picture_grid",
    ]  # fmt: skip


@pytest.mark.parametrize("task", tasks.TASKS, ids=lambda t: t.id)
def test_the_right_submissions_pass_and_nothing_fails(task: tasks.Task, site: fixture.Site, browser: Browser) -> None:
    assert not tasks.check(task, [], "")["passed"]
    reply = solve(task.id, browser)
    outcome = tasks.check(task, site.submissions, reply)
    assert outcome["passed"], outcome


@pytest.mark.parametrize(
    ("task", "field", "wrong"),
    [
        ("job_two_step", "sponsorship", "yes"),
        ("job_two_step", "grad_date", "May 2019"),
        ("job_two_step", "linkedin", "https://example.com/in/kate"),
        ("job_upload", "gender", "Male"),
        ("checkout", "delivery", "overnight"),
        ("checkout", "news", "on"),
        ("checkin", "mood", "Tired"),
        ("settings", "language", "German"),
        ("settings", "weekly_digest", "on"),
        ("picture_grid", "selected", [2, 5, 8]),
        ("picture_grid", "selected", [2, 5, 8, 10, 1]),
    ],
)
def test_one_wrong_field_fails_the_task(
    task: str, field: str, wrong: Any, site: fixture.Site, browser: Browser
) -> None:
    solve(task, browser)
    row = next(s for s in site.submissions if s["kind"] == tasks.BY_ID[task].kind and s["status"] == "accepted")
    changed = [{**row, "fields": {**row["fields"], field: wrong}}]
    outcome = tasks.check(tasks.BY_ID[task], changed)
    assert not outcome["passed"] and list(outcome["differences"]) == [field]


def test_harmless_differences_in_form_pass() -> None:
    task = tasks.BY_ID["job_two_step"]
    fields = {
        **task.expected,
        "email": "Kate.Lin@Example.com ",
        "phone": "312-555-0147",
        "school": "Lakeshore  State University",
    }
    assert tasks.check(task, [{"kind": task.kind, "status": "accepted", "fields": fields, "files": []}])["passed"]


def test_a_task_counts_only_its_own_job() -> None:
    task = tasks.BY_ID["job_upload"]
    other = tasks.BY_ID["job_two_step"]
    row = {"kind": other.kind, "status": "accepted", "fields": dict(other.expected), "files": []}
    outcome = tasks.check(task, [row])
    assert not outcome["passed"] and outcome["reason"] == "nothing was submitted"


def test_the_upload_is_checked_by_name_and_hash(site: fixture.Site, browser: Browser) -> None:
    task = tasks.BY_ID["job_upload"]
    solve("job_upload", browser)
    row = site.submissions[-1]
    assert tasks.check(task, [row])["passed"]
    assert tasks.check(task, [{**row, "files": []}])["upload_problem"] == "no file arrived"
    named = [{**row["files"][0], "name": "cv.pdf"}]
    assert "cv.pdf" in tasks.check(task, [{**row, "files": named}])["upload_problem"]
    hashed = [{**row["files"][0], "sha256": "0" * 64}]
    assert "sha256" in tasks.check(task, [{**row, "files": hashed}])["upload_problem"]


def test_the_resume_is_a_pdf_with_a_fixed_hash() -> None:
    data = tasks.resume_bytes()
    assert data.startswith(b"%PDF-1.4") and data.endswith(b"%%EOF\n") and data == tasks.resume_bytes()


def test_which_submission_decides() -> None:
    def row(task: tasks.Task, **changes: Any) -> dict[str, Any]:
        return {"kind": task.kind, "status": "accepted", "fields": {**task.expected, **changes}, "files": []}

    order = tasks.BY_ID["checkout"]
    assert not tasks.check(order, [row(order, zip="00000"), row(order)])["passed"]  # the first stands
    settings = tasks.BY_ID["settings"]
    assert tasks.check(settings, [row(settings, sms="on"), row(settings)])["passed"]  # the last stands
    signin = tasks.BY_ID["signin"]
    assert tasks.check(signin, [row(signin, username="other"), row(signin)])["passed"]  # any
    refused = {**row(order), "status": "refused"}
    assert tasks.check(order, [refused])["reason"] == "the site refused all 1 submissions"


@pytest.mark.parametrize(
    ("reply", "passed"),
    [
        (f"FINAL ANSWER: {tasks.TARGET_PART.mpn}", True),
        (f"**FINAL ANSWER:** `{tasks.TARGET_PART.mpn}`.", True),
        (f"It is {tasks.TARGET_PART.mpn}", False),
        (f"FINAL ANSWER: {tasks.TARGET_PART.mpn.lower()}", False),
        ("FINAL ANSWER: HC-19382-P", False),
        ("FINAL ANSWER: unknown", False),
        ("", False),
    ],
)
def test_the_search_answer_is_an_exact_match(reply: str, passed: bool) -> None:
    assert tasks.check(tasks.BY_ID["search"], [], reply)["passed"] is passed


def test_the_workspace_holds_the_task_files(tmp_path: Path) -> None:
    task = tasks.BY_ID["job_upload"]
    tasks.prepare_workspace(task, tmp_path / "ws")
    assert (tmp_path / "ws" / tasks.RESUME_NAME).read_bytes() == tasks.resume_bytes()
    assert str(tmp_path / "ws" / tasks.RESUME_NAME) in task.text(tmp_path / "ws")


def test_the_prompts_name_what_is_checked_and_no_task_needs_a_person() -> None:
    for task in tasks.TASKS:
        text = task.text(Path("/ws"))
        assert "do not ask me anything" in text and "{" not in text
    assert fixture.TASKMERE_ACCOUNT[1] in tasks.BY_ID["signin"].text(Path("/ws"))
    assert fixture.FORUM_ACCOUNT[1] in tasks.BY_ID["checkin"].text(Path("/ws"))


def test_the_records_are_json_lines(site: fixture.Site, browser: Browser) -> None:
    solve("checkin", browser)
    lines = (site.out / "submissions.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["forum.signin", "forum.checkin"]
    assert any(
        json.loads(line)["path"] == "/forum/checkin" for line in (site.out / "requests.jsonl").read_text().splitlines()
    )
