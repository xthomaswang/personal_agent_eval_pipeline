"""The eight tasks of the forms suite: where each starts, what the agent is told, and what passes it.

A task passes on what the site recorded (fixture.py), never on what the agent says it did: the fields of the deciding
submission must equal ``expected``, and an upload must match by file name and sha256. The one exception is ``search``,
which asks for a value: the reply's ``FINAL ANSWER:`` line must match exactly, markup and a final full stop aside. No
model grades anything.

Which submission decides is the task's ``pick``. ``first`` is for what cannot be taken back (an application, an order, a
check-in; the site refuses a second one), ``last`` for what may be corrected (settings, a listing's photos), and ``any``
for a sign-in, where a wrong password before the right one does no harm. Rows the site refused do not count.

Fields compare as ``normalize`` says: ``text`` (spaces trimmed and collapsed), ``email`` (also lower case), ``digits``
(digits only, for a phone number), and ``exact`` otherwise. A field not named in ``expected`` is not compared, and a
field named with ``""`` must be empty.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forms import fixture
from web import answer as web_answer

CLOSING = (
    "Finish the whole task yourself and do not ask me anything: nobody can answer you. Everything here is a test "
    "site with made-up data. When you are done, say so in one short line."
)
RESUME_NAME = "nadia_haddad_resume.pdf"


def resume_bytes() -> bytes:
    """A small, valid PDF: the file of the upload task. Fixed, so that its sha256 is too."""
    text = b"BT /F1 18 Tf 72 720 Td (Nadia Haddad - Resume) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(text), text),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    table = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, table)
    return out


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    start: str  # the path Chrome opens before the run
    prompt: str  # ``{workspace}`` is the run's folder
    kind: str | None = None  # the kind of the deciding submission; None for an answer
    pick: str = "first"
    expected: Mapping[str, Any] = field(default_factory=dict)
    normalize: Mapping[str, str] = field(default_factory=dict)
    upload: Mapping[str, str] | None = None  # the field, file name and sha256 of the file that must arrive
    workspace_files: Mapping[str, bytes] = field(default_factory=dict)
    answer: str | None = None

    def text(self, workspace: Path) -> str:
        return f"{self.prompt.format(workspace=workspace)}\n\n{CLOSING}"


CAREER_NORMALIZE = {
    "email": "email",
    "phone": "digits",
    "first_name": "text",
    "last_name": "text",
    "preferred_name": "text",
    "city": "text",
    "linkedin": "text",
    "website": "text",
    "school": "text",
    "grad_date": "text",
}
SHIPPING_NORMALIZE = {
    "email": "email",
    "phone": "digits",
    "first_name": "text",
    "last_name": "text",
    "company": "text",
    "address": "text",
    "address2": "text",
    "city": "text",
    "zip": "text",
    "gift_code": "text",
}
TILES_WITH_A_MUG = [2, 5, 8, 10]
TARGET_PART = next(p for p in fixture.CATALOG if p.sku == "FS-10243")

TASKS: tuple[Task, ...] = (
    Task(
        id="signin",
        title="Sign in with a fixture account",
        start="/taskmere/",
        prompt=(
            "A Chrome browser is open on the Taskmere sign-in page. Use the browser tools to sign in. "
            f"The username is {fixture.TASKMERE_ACCOUNT[0]} and the password is {fixture.TASKMERE_ACCOUNT[1]}. "
            "The task is done when you see the signed-in home page."
        ),
        kind="taskmere.signin",
        pick="any",
        expected={"username": fixture.TASKMERE_ACCOUNT[0], "ok": True},
    ),
    Task(
        id="job_two_step",
        title="Two-step job application",
        start="/careers/jobs/DA-2047",
        prompt=(
            "A Chrome browser is open on a job posting at Quillmont Careers. Use the browser tools to apply for this "
            "job and submit the application. It takes two steps. Use only the information below and leave every "
            "other field as it is. Follow the format hints shown in the form.\n\n"
            "- Legal name: Katherine Lin; she goes by Kate\n"
            "- Email: kate.lin@example.com\n"
            "- Phone: (312) 555-0147, a U.S. number\n"
            "- Lives in: Evanston\n"
            "- Portfolio website: https://katelin.example.com\n"
            "- School: Lakeshore State University\n"
            "- Degree: Master's\n"
            "- Graduated: May 2019\n"
            "- Work authorization: U.S. citizen, legally authorized to work in the United States, and will never "
            "need visa sponsorship\n"
            "- How she heard about the job: Employee referral\n"
            "- Gender (voluntary): Decline to self-identify\n"
            "- Privacy consent: she agrees"
        ),
        kind="careers.application",
        expected={
            "job": "DA-2047",
            "first_name": "Katherine",
            "last_name": "Lin",
            "preferred_name": "Kate",
            "email": "kate.lin@example.com",
            "phone_country": "+1",
            "phone": "(312) 555-0147",
            "city": "Evanston",
            "linkedin": "",
            "website": "https://katelin.example.com",
            "school": "Lakeshore State University",
            "degree": "Master's",
            "grad_date": "05/2019",
            "authorized": "yes",
            "sponsorship": "no",
            "hear_about": "Employee referral",
            "gender": "Decline to self-identify",
            "consent": "on",
        },
        normalize=CAREER_NORMALIZE,
    ),
    Task(
        id="job_upload",
        title="Job application with a resume upload",
        start="/careers/jobs/PM-3310",
        prompt=(
            "A Chrome browser is open on a job posting at Quillmont Careers. Use the browser tools to apply for this "
            "job and submit the application. It takes two steps, and the form needs a resume: attach the file "
            f"{{workspace}}/{RESUME_NAME}. Use only the information below and leave every other field as it is. "
            "Follow the format hints shown in the form.\n\n"
            "- Legal name: Nadia Haddad\n"
            "- Email: nadia.haddad@example.com\n"
            "- Phone: (617) 555-0182, a U.S. number\n"
            "- Lives in: Somerville\n"
            "- School: Calder Bay University\n"
            "- Degree: PhD\n"
            "- Graduated: December 2024\n"
            "- Work authorization: legally authorized to work in the United States now, and will need visa "
            "sponsorship in the future\n"
            "- How she heard about the job: University career fair\n"
            "- Gender (voluntary): Female\n"
            "- Privacy consent: she agrees"
        ),
        kind="careers.application",
        expected={
            "job": "PM-3310",
            "first_name": "Nadia",
            "last_name": "Haddad",
            "preferred_name": "",
            "email": "nadia.haddad@example.com",
            "phone_country": "+1",
            "phone": "(617) 555-0182",
            "city": "Somerville",
            "linkedin": "",
            "website": "",
            "school": "Calder Bay University",
            "degree": "PhD",
            "grad_date": "12/2024",
            "authorized": "yes",
            "sponsorship": "yes",
            "hear_about": "University career fair",
            "gender": "Female",
            "consent": "on",
        },
        normalize=CAREER_NORMALIZE,
        upload={"field": "resume", "name": RESUME_NAME},
        workspace_files={RESUME_NAME: resume_bytes()},
    ),
    Task(
        id="checkout",
        title="Checkout",
        start="/shop/cart",
        prompt=(
            "A Chrome browser is open on the cart of the Kestrelmoor Outfitters shop. Use the browser tools to check "
            "out and place the order, with the details below. Leave every other field as it is.\n\n"
            "- Email: priya.raman@example.com; she does not want news and offers by email\n"
            "- Name: Priya Raman\n"
            "- Address: 418 Alder Court, Apartment 3B, Portland, Oregon 97205\n"
            "- Phone: (503) 555-0139\n"
            "- Delivery: Express\n"
            f"- Gift card: apply the code {fixture.GIFT_CODE}\n"
            "- Payment: bank transfer\n"
            "- She agrees to the terms of sale"
        ),
        kind="shop.order",
        expected={
            "email": "priya.raman@example.com",
            "news": "",
            "first_name": "Priya",
            "last_name": "Raman",
            "company": "",
            "address": "418 Alder Court",
            "address2": "Apartment 3B",
            "city": "Portland",
            "state": "Oregon",
            "zip": "97205",
            "phone": "(503) 555-0139",
            "delivery": "express",
            "gift_code": fixture.GIFT_CODE,
            "payment": "bank_transfer",
            "terms": "on",
            "total": "124.00",
        },
        normalize=SHIPPING_NORMALIZE,
    ),
    Task(
        id="search",
        title="Search, filter, open a result, report a field",
        start="/supply/search",
        prompt=(
            "A Chrome browser is open on the Fennick Supply catalog. Use the browser tools to search for hex bolt, "
            "filter the results to Stainless steel A4 and a length of 40 mm, open the M8 result, and read its "
            "manufacturer part number.\n\n"
            "End your reply with one line: FINAL ANSWER: <the manufacturer part number>"
        ),
        answer=TARGET_PART.mpn,
    ),
    Task(
        id="checkin",
        title="Daily check-in after a sign-in",
        start="/forum/",
        prompt=(
            "A Chrome browser is open on the Harborline Community forum. Use the browser tools to sign in and do "
            f"today's daily check-in. The username is {fixture.FORUM_ACCOUNT[0]} and the password is "
            f"{fixture.FORUM_ACCOUNT[1]}. Today is going Focused, and the note for today is: Thesis draft, chapter 2"
        ),
        kind="forum.checkin",
        expected={"user": fixture.FORUM_ACCOUNT[0], "mood": "Focused", "note": "Thesis draft, chapter 2"},
        normalize={"note": "text"},
    ),
    Task(
        id="settings",
        title="A settings change",
        start="/taskmere/open",
        prompt=(
            "A Chrome browser is open on the notification settings of Taskmere, and you are signed in. Use the "
            "browser tools to make these changes and save them. Leave everything else as it is.\n\n"
            "- Turn off Product updates\n"
            "- Turn on Mobile push\n"
            "- Set the digest frequency to Daily\n"
            "- Set the quiet hours from 22:30 to 07:00\n"
            "- Set the time zone to Berlin"
        ),
        kind="taskmere.settings",
        pick="last",
        expected={
            **fixture.DEFAULT_SETTINGS,
            "product_updates": "off",
            "mobile_push": "on",
            "digest_frequency": "Daily",
            "quiet_from": "22:30",
            "quiet_to": "07:00",
            "time_zone": "(UTC+01:00) Berlin",
        },
    ),
    Task(
        id="picture_grid",
        title="A picture grid",
        start="/market/listing/photos",
        prompt=(
            "A Chrome browser is open on the photo step of a new listing at Brindle Market. The listing is for mugs. "
            "Use the browser tools to select every tile that shows a mug, and no other tile, then press Use selected "
            "photos. The tiles are pictures with no text, so look at them in a screenshot of the page."
        ),
        kind="market.photos",
        pick="last",
        expected={"selected": TILES_WITH_A_MUG},
    ),
)
BY_ID = {task.id: task for task in TASKS}


def prepare_workspace(task: Task, workspace: Path) -> None:
    """The run's folder, with the task's files."""
    workspace.mkdir(parents=True, exist_ok=True)
    for name, data in task.workspace_files.items():
        (workspace / name).write_bytes(data)


def normalized(kind: str | None, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if kind == "digits":
        return re.sub(r"\D", "", value)
    if kind in ("text", "email"):
        text = " ".join(value.split())
        return text.lower() if kind == "email" else text
    return value


def deciding(task: Task, submissions: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, int]:
    """The submission that decides ``task``, and how many of its kind the site accepted (``duplicate`` rows aside)."""
    rows = [s for s in submissions if s.get("kind") == task.kind and s.get("status") == "accepted"]
    if task.id.startswith("job_"):  # the site has one application form; each job's rows are its own
        rows = [s for s in rows if (s.get("fields") or {}).get("job") == task.expected["job"]]
    if not rows:
        return None, 0
    if task.pick == "any":
        good = [s for s in rows if not field_differences(task, s)]
        return (good[0] if good else rows[-1]), len(rows)
    return (rows[0] if task.pick == "first" else rows[-1]), len(rows)


def field_differences(task: Task, row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each expected field the row gets wrong, with what was expected and what arrived."""
    fields = row.get("fields") or {}
    wrong: dict[str, dict[str, Any]] = {}
    for name, want in task.expected.items():
        kind = task.normalize.get(name, "exact")
        got = fields.get(name, "")
        if normalized(kind, got) != normalized(kind, want):
            wrong[name] = {"expected": want, "got": got}
    return wrong


def upload_problem(task: Task, row: Mapping[str, Any]) -> str | None:
    """Why the deciding row's file is not the task's file, by name and sha256; None when it is."""
    if task.upload is None:
        return None
    files = [f for f in row.get("files") or [] if f.get("field") == task.upload["field"]]
    if not files:
        return "no file arrived"
    want = hashlib.sha256(task.workspace_files[task.upload["name"]]).hexdigest()
    if files[-1].get("name") != task.upload["name"]:
        return f"the file is {files[-1].get('name')!r}"
    return None if files[-1].get("sha256") == want else "the file's sha256 differs"


def check(task: Task, submissions: Sequence[Mapping[str, Any]], reply: str = "") -> dict[str, Any]:
    """Did the run pass? ``passed``, and the evidence for it."""
    if task.answer is not None:
        found = web_answer.extract(reply)  # it takes off markup and quotes
        got = found.answer.rstrip(". ")
        return {
            "passed": got == task.answer,
            "reason": None if got == task.answer else f"the answer is {got!r}",
            "answer": got or None,
            "answer_status": found.status,
            "n_accepted": 0,
            "differences": {},
        }
    row, count = deciding(task, submissions)
    if row is None:
        refused = sum(1 for s in submissions if s.get("kind") == task.kind and s.get("status") == "refused")
        reason = f"the site refused all {refused} submissions" if refused else "nothing was submitted"
        return {"passed": False, "reason": reason, "n_accepted": 0, "differences": {}}
    differences = field_differences(task, row)
    file_problem = upload_problem(task, row)
    passed = not differences and file_problem is None
    reason = None
    if not passed:
        reason = file_problem or f"wrong fields: {', '.join(differences)}"
    return {
        "passed": passed,
        "reason": reason,
        "n_accepted": count,
        "differences": differences,
        "upload_problem": file_problem,
    }
