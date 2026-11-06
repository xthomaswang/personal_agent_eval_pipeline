"""The fixture site of the forms suite: six small fictional sites on one local http server, one server per run.

    uv run python agent_ability/forms/fixture.py   # serve it on a free port, for a look by hand

``Site`` binds 127.0.0.1 on a port the system picks (never one of ``RESERVED_PORTS``) and keeps its state in memory, so
every run starts from the same site. The pages are plain server-side forms from ``pages/``, grown from the static pages
of ui_vision. What a run did is recorded server-side, in ``site.submissions`` and in ``<out>/submissions.jsonl``:

- one row per submission, with its ``kind`` (``KINDS``), the fields as the server received them, the uploaded files'
  names and sha256 sums, and ``status``: ``accepted``, ``refused`` (it failed validation; nothing happened) or
  ``duplicate`` (a second application, order or check-in, which the site turns down as a real one would);
- every request, in ``<out>/requests.jsonl``.

Passwords are compared and dropped: a row says whether a sign-in worked, never what was typed.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import secrets
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
HOST = "127.0.0.1"
# The model servers and other services of this Mac. Port 0 gives an ephemeral port, far from these; checked anyway.
RESERVED_PORTS = (range(8400, 8800), range(8955, 8956), range(11434, 11435), range(11500, 11501))
MAX_BODY = 8 * 1024 * 1024
MAX_RESUME = 5 * 1024 * 1024

KINDS = (
    "taskmere.signin", "taskmere.settings", "careers.application", "shop.order", "forum.signin", "forum.checkin",
    "market.photos",
)  # fmt: skip
ONE_ONLY = ("careers.application", "shop.order", "forum.checkin")  # what cannot be done twice

# Fake accounts, one per site; the task prompts give them to the agent.
TASKMERE_ACCOUNT = ("rowan.ellis", "Lantern-4821")
FORUM_ACCOUNT = ("mara.quinn", "Tidewater-7315")
GIFT_CODE = "KM-GIFT-25"
GIFT_VALUE = 25.0
TILE_COUNT = 10

SITES: dict[str, dict[str, str]] = {
    "taskmere": {
        "brand": "Taskmere",
        "home": "/taskmere/",
        "theme": "taskmere",
        "nav": '<a href="#">My tasks</a><a href="#">Projects</a><a href="#">Inbox</a>',
        "footer": "Taskmere, 12 Example Row, Portland, OR",
    },
    "careers": {
        "brand": "Quillmont Careers",
        "home": "/careers/jobs/DA-2047",
        "theme": "careers",
        "nav": (
            '<a href="#">Teams</a><a href="#">Benefits</a><a href="#">Life at Quillmont</a>'
            '<a href="#">All open roles</a>'
        ),
        "footer": "Powered by Talentrail. Quillmont, 400 Example Plaza, Chicago, IL",
    },
    "shop": {
        "brand": "Kestrelmoor Outfitters",
        "home": "/shop/cart",
        "theme": "shop",
        "nav": '<a href="#">Packs</a><a href="#">Clothing</a><a href="/shop/cart">Cart</a>',
        "footer": '<a href="#">Refund policy</a> <a href="#">Shipping policy</a> <a href="#">Terms of service</a>',
    },
    "supply": {
        "brand": "Fennick Supply",
        "home": "/supply/search",
        "theme": "supply",
        "nav": '<a href="/supply/search">Catalog</a><a href="#">Quotes</a><a href="#">Help</a>',
        "footer": "Fennick Supply, trade fasteners since 1962",
    },
    "forum": {
        "brand": "Harborline Community",
        "home": "/forum/",
        "theme": "forum",
        "nav": '<a href="/forum/">All threads</a><a href="#">Members</a><a href="#">Rules</a>',
        "footer": "Harborline Community is run by its members.",
    },
    "market": {
        "brand": "Brindle Market",
        "home": "/market/listing/photos",
        "theme": "market",
        "nav": '<a href="#">Buy</a><a href="#">Sell</a><a href="#">Messages</a>',
        "footer": "Brindle Market, a place to sell what you no longer need.",
    },
}


@dataclass(frozen=True)
class Job:
    id: str
    title: str
    team: str
    location: str
    pay: str
    needs_resume: bool = False


JOBS = {
    "DA-2047": Job("DA-2047", "Senior Data Analyst", "Analytics", "Chicago, IL, Hybrid", "$118,000 to $142,000 a year"),
    "PM-3310": Job(
        "PM-3310", "Product Manager, Growth", "Product", "Boston, MA, Hybrid", "$131,000 to $158,000 a year", True
    ),
}  # fmt: skip
STEP1_REQUIRED = {"first_name": "Legal first name", "last_name": "Legal last name", "email": "Email", "phone": "Phone"}
STEP1_FIELDS = (*STEP1_REQUIRED, "preferred_name", "phone_country", "city", "linkedin", "website")
STEP2_REQUIRED = {
    "authorized": "Work authorization",
    "sponsorship": "Visa sponsorship",
    "consent": "Consent to data processing",
}
STEP2_FIELDS = (*STEP2_REQUIRED, "school", "degree", "grad_date", "hear_about", "gender")
RESUME_TYPES = (".pdf", ".doc", ".docx")

CART = (("Ridgeline Daypack 22L, Slate", 1, 89.0), ("Merino Hiking Socks, 2-pack", 2, 24.0))
DELIVERY = {"standard": ("Standard", 0.0), "express": ("Express", 12.0), "overnight": ("Overnight", 29.0)}
SHIPPING_REQUIRED = {
    "email": "Email",
    "first_name": "First name",
    "last_name": "Last name",
    "address": "Address",
    "city": "City",
    "state": "State",
    "zip": "ZIP code",
    "phone": "Phone",
}
SHIPPING_FIELDS = (*SHIPPING_REQUIRED, "company", "address2", "news", "delivery")

# The switches of the settings page, with their labels, and every field's value before any change.
SWITCHES = {
    "product_updates": "Product updates",
    "weekly_digest": "Weekly digest",
    "comments": "Comments",
    "mentions": "Mentions",
    "desktop": "Desktop notifications",
    "mobile_push": "Mobile push",
    "sms": "SMS alerts",
}
DEFAULT_SETTINGS = {
    "product_updates": "on",
    "weekly_digest": "off",
    "comments": "on",
    "mentions": "on",
    "desktop": "on",
    "mobile_push": "off",
    "sms": "off",
    "digest_frequency": "Weekly",
    "quiet_from": "21:00",
    "quiet_to": "08:00",
    "time_zone": "(UTC-05:00) Eastern Time (US & Canada)",
    "language": "English (US)",
}
_TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


@dataclass(frozen=True)
class Part:
    sku: str
    name: str
    thread: str
    length: int
    material: str
    pack: int
    price: str
    mpn: str  # shown on the item's page only


CATALOG = (
    Part("FS-10231", "Hex bolt M6 x 30", "M6", 30, "Zinc-plated steel", 100, "$11.40", "KD-30417-B"),
    Part("FS-10232", "Hex bolt M6 x 40", "M6", 40, "Stainless steel A2", 100, "$21.90", "RT-58120-M"),
    Part("FS-10233", "Hex bolt M6 x 40", "M6", 40, "Stainless steel A4", 100, "$29.75", "QL-71846-D"),
    Part("FS-10240", "Hex bolt M8 x 30", "M8", 30, "Stainless steel A4", 50, "$24.10", "VN-20953-H"),
    Part("FS-10241", "Hex bolt M8 x 40", "M8", 40, "Zinc-plated steel", 50, "$15.20", "BX-64471-T"),
    Part("FS-10242", "Hex bolt M8 x 40", "M8", 40, "Stainless steel A2", 50, "$26.35", "HC-19382-P"),
    Part("FS-10243", "Hex bolt M8 x 40", "M8", 40, "Stainless steel A4", 50, "$33.80", "ZW-48213-Q"),
    Part("FS-10244", "Hex bolt M8 x 50", "M8", 50, "Stainless steel A4", 50, "$38.15", "MJ-77054-C"),
    Part("FS-10251", "Hex bolt M10 x 40", "M10", 40, "Stainless steel A4", 25, "$31.60", "TG-35628-K"),
    Part("FS-10252", "Hex bolt M10 x 50", "M10", 50, "Zinc-plated steel", 25, "$18.45", "PF-90716-S"),
    Part("FS-20110", "Socket cap screw M8 x 40", "M8", 40, "Stainless steel A4", 50, "$36.20", "DN-26491-A"),
    Part("FS-20111", "Carriage bolt M8 x 40", "M8", 40, "Zinc-plated steel", 50, "$14.70", "LS-83305-E"),
)

THREADS = (
    ("Board rules: read this before you post", "Moderators", 3),
    ("Check-in points are changing on the first of the month", "Moderators", 41),
    ("Which winter jacket for under $200?", "coldhands", 27),
    ("Moving sale: monitor, office chair, rice cooker", "packing_again", 8),
    ("What to bring to a first visa interview", "northstar", 19),
    ("Sublet near the east gate from December", "harbor_rat", 5),
    ("Anyone driving north to see the leaves?", "mapleseeker", 12),
)


def page_text(name: str) -> str:
    return (PAGES / name).read_text(encoding="utf-8")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def money(value: float) -> str:
    return f"${value:,.2f}"


def fill(template: str, values: Mapping[str, Any]) -> str:
    """``template`` with its ``{{name}}`` marks replaced by ``values``, which are already HTML; a missing one raises."""
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values[m.group(1)]), template)


def restore_script(values: Mapping[str, str]) -> str:
    """The script that puts a form's values back (``restore`` of site.js)."""
    text = json.dumps(dict(values)).replace("</", "<\\/")
    return f'<script>addEventListener("DOMContentLoaded", () => restore({text}));</script>'


@dataclass(frozen=True)
class Upload:
    field: str
    name: str
    sha256: str
    size: int


def parse_multipart(boundary: bytes, body: bytes) -> tuple[dict[str, str], list[Upload]]:
    """The fields and files of a multipart body. By hand, so the sha256 of a file is of the exact bytes sent."""
    form: dict[str, str] = {}
    uploads: list[Upload] = []
    for chunk in body.split(b"--" + boundary):
        chunk = chunk.removeprefix(b"\r\n")
        if not chunk or chunk.startswith(b"--"):
            continue
        head, _, data = chunk.partition(b"\r\n\r\n")
        data = data.removesuffix(b"\r\n")
        name = re.search(rb'\bname="([^"]*)"', head)
        filename = re.search(rb'\bfilename="([^"]*)"', head)
        if name is None:
            continue
        key = name.group(1).decode("utf-8", "replace")
        if filename is None:
            form[key] = data.decode("utf-8", "replace")
        elif filename.group(1):  # an empty name is a file field left empty
            uploads.append(
                Upload(key, filename.group(1).decode("utf-8", "replace"), hashlib.sha256(data).hexdigest(), len(data))
            )
    return form, uploads


def parse_body(content_type: str, body: bytes) -> tuple[dict[str, str], list[Upload]]:
    if content_type.startswith("multipart/form-data"):
        found = re.search(r'boundary="?([^";]+)', content_type)
        return parse_multipart(found.group(1).encode(), body) if found else ({}, [])
    return dict(parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True)), []


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, str]
    form: dict[str, str]
    uploads: list[Upload]
    session: dict[str, Any]


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "text/html; charset=utf-8"
    location: str | None = None


def redirect(location: str) -> Response:
    return Response(303, location=location)


ROUTES: tuple[tuple[str, str, str], ...] = (
    ("GET", r"/static/([\w.-]+(?:/[\w.-]+)?)", "static"),
    ("GET", r"/taskmere/", "taskmere_root"),
    ("GET", r"/taskmere/signin", "taskmere_signin"),
    ("POST", r"/taskmere/signin", "taskmere_signin_post"),
    ("GET", r"/taskmere/home", "taskmere_home"),
    ("GET", r"/taskmere/open", "taskmere_open"),
    ("GET", r"/taskmere/settings/notifications", "taskmere_settings"),
    ("POST", r"/taskmere/settings/notifications", "taskmere_settings_post"),
    ("GET", r"/careers/jobs/([\w-]+)", "careers_job"),
    ("GET", r"/careers/apply/([\w-]+)/1", "careers_step1"),
    ("POST", r"/careers/apply/([\w-]+)/1", "careers_step1_post"),
    ("GET", r"/careers/apply/([\w-]+)/2", "careers_step2"),
    ("POST", r"/careers/apply/([\w-]+)/2", "careers_step2_post"),
    ("GET", r"/careers/apply/([\w-]+)/done", "careers_done"),
    ("GET", r"/shop/cart", "shop_cart"),
    ("GET", r"/shop/checkout", "shop_shipping"),
    ("POST", r"/shop/checkout", "shop_shipping_post"),
    ("GET", r"/shop/checkout/payment", "shop_payment"),
    ("POST", r"/shop/checkout/payment", "shop_payment_post"),
    ("GET", r"/shop/order", "shop_order"),
    ("GET", r"/supply/search", "supply_search"),
    ("GET", r"/supply/item/([\w-]+)", "supply_item"),
    ("GET", r"/forum/", "forum_home"),
    ("GET", r"/forum/signin", "forum_signin"),
    ("POST", r"/forum/signin", "forum_signin_post"),
    ("POST", r"/forum/checkin", "forum_checkin_post"),
    ("GET", r"/market/listing/photos", "market_photos"),
    ("POST", r"/market/listing/photos", "market_photos_post"),
    ("GET", r"/market/listing/done", "market_done"),
)
_ROUTES = tuple((method, re.compile(pattern), name) for method, pattern, name in ROUTES)
STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Site:
    """The site and its state, on 127.0.0.1 and a free port until ``stop``."""

    def __init__(self, out: Path | None = None) -> None:
        self.out = Path(out) if out else None
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        self.submissions: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.sessions: dict[str, dict[str, Any]] = {}
        self.settings = dict(DEFAULT_SETTINGS)
        self.lock = threading.Lock()
        self.httpd = self._bind()
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def _bind(self) -> ThreadingHTTPServer:
        site = self

        class Handler(_Handler):
            owner = site

        while True:
            httpd = ThreadingHTTPServer((HOST, 0), Handler)  # port 0: the system picks one that is free
            if not any(int(httpd.server_address[1]) in reserved for reserved in RESERVED_PORTS):
                httpd.daemon_threads = True
                return httpd
            httpd.server_close()

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def url(self, path: str) -> str:
        return self.base_url + path

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    # --- what the site keeps ---

    def _append(self, name: str, row: dict[str, Any]) -> None:
        if self.out:
            with (self.out / name).open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record(
        self, kind: str, fields: Mapping[str, Any], *, status: str = "accepted", files: Sequence[Upload] = ()
    ) -> dict[str, Any]:
        """Keep one submission. A second accepted row of a ONE_ONLY kind becomes ``duplicate``."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        with self.lock:
            if status == "accepted" and kind in ONE_ONLY and self.accepted(kind):
                status = "duplicate"
            row = {
                "t": round(time.time(), 3),
                "kind": kind,
                "status": status,
                "fields": dict(fields),
                "files": [{"field": f.field, "name": f.name, "sha256": f.sha256, "size": f.size} for f in files],
            }
            self.submissions.append(row)
            self._append("submissions.jsonl", row)
        return row

    def accepted(self, kind: str) -> list[dict[str, Any]]:
        return [s for s in self.submissions if s["kind"] == kind and s["status"] == "accepted"]

    # --- rendering ---

    def page(
        self,
        site: str,
        title: str,
        template: str,
        values: Mapping[str, Any] | None = None,
        *,
        alert: str = "",
        notice: str = "",
        form_values: Mapping[str, str] | None = None,
        status: int = 200,
    ) -> Response:
        banner = f'<div class="alert" role="alert">{esc(alert)}</div>' if alert else ""
        banner += f'<div class="notice" role="status">{esc(notice)}</div>' if notice else ""
        body = fill(page_text(template), values or {})
        text = fill(
            page_text("_layout.html"),
            {
                **SITES[site],
                "title": esc(title),
                "alert": banner,
                "body": body,
                "restore": restore_script(form_values) if form_values else "",
            },
        )
        return Response(status, text.encode("utf-8"))

    # --- routing ---

    def handle(self, request: Request) -> Response:
        for method, pattern, name in _ROUTES:
            found = pattern.fullmatch(request.path)
            if method == request.method and found:
                return getattr(self, name)(request, *found.groups())
        return Response(404, b"<!doctype html><title>Not found</title><h1>Page not found</h1>")

    def static(self, _request: Request, name: str) -> Response:
        path = (PAGES / name).resolve()
        kind = STATIC_TYPES.get(path.suffix)
        if kind is None or PAGES.resolve() not in path.parents or not path.is_file():
            return Response(404, b"not found", "text/plain")
        return Response(200, path.read_bytes(), kind)

    # --- Taskmere: a sign-in, and the notification settings ---

    def _signin_page(self, site: str, action: str, alert: str = "") -> Response:
        brand = SITES[site]["brand"]
        values = {
            "heading": f"Sign in to {esc(brand)}",
            "subheading": "Use your account to continue.",
            "action": action,
        }
        return self.page(site, f"Sign in - {brand}", "signin.html", values, alert=alert)

    def _sign_in(self, kind: str, account: tuple[str, str], request: Request) -> bool:
        username = request.form.get("username", "").strip()
        ok = (username, request.form.get("password", "")) == account
        self.record(kind, {"username": username, "ok": ok}, status="accepted" if ok else "refused")
        if ok:
            request.session[kind.split(".")[0]] = username
        return ok

    def taskmere_root(self, request: Request) -> Response:
        return redirect("/taskmere/home" if request.session.get("taskmere") else "/taskmere/signin")

    def taskmere_signin(self, _request: Request) -> Response:
        return self._signin_page("taskmere", "/taskmere/signin")

    def taskmere_signin_post(self, request: Request) -> Response:
        if self._sign_in("taskmere.signin", TASKMERE_ACCOUNT, request):
            return redirect("/taskmere/home")
        return self._signin_page("taskmere", "/taskmere/signin", "That username and password do not match.")

    def taskmere_home(self, request: Request) -> Response:
        user = request.session.get("taskmere")
        if not user:
            return redirect("/taskmere/signin")
        return self.page("taskmere", "Home - Taskmere", "taskmere_home.html", {"user": esc(user)})

    def taskmere_open(self, request: Request) -> Response:
        """The settings task starts signed in, as a person's own browser would be."""
        request.session["taskmere"] = TASKMERE_ACCOUNT[0]
        return redirect("/taskmere/settings/notifications")

    def _switch(self, name: str) -> str:
        on = self.settings[name] == "on"
        return (
            f'<div class="switch-row"><span id="t_{name}">{esc(SWITCHES[name])}</span>'
            f'<input type="hidden" name="{name}" value="{"on" if on else "off"}">'
            f'<button type="button" class="switch" role="switch" aria-checked="{str(on).lower()}" '
            f'aria-labelledby="t_{name}" data-field="{name}"></button></div>'
        )

    def _settings_page(self, *, alert: str = "", notice: str = "", values: Mapping[str, str] | None = None) -> Response:
        rest = {k: v for k, v in (values or self.settings).items() if k not in SWITCHES}
        switches = {f"switch_{name}": self._switch(name) for name in SWITCHES}
        return self.page(
            "taskmere", "Notifications - Settings - Taskmere", "taskmere_settings.html", switches,
            alert=alert, notice=notice, form_values=rest,
        )  # fmt: skip

    def taskmere_settings(self, request: Request) -> Response:
        if not request.session.get("taskmere"):
            return redirect("/taskmere/signin")
        return self._settings_page()

    def taskmere_settings_post(self, request: Request) -> Response:
        if not request.session.get("taskmere"):
            return redirect("/taskmere/signin")
        fields = {name: request.form.get(name, self.settings[name]).strip() for name in DEFAULT_SETTINGS}
        bad = [name for name in ("quiet_from", "quiet_to") if not _TIME.fullmatch(fields[name])]
        if bad:
            self.record("taskmere.settings", fields, status="refused")
            return self._settings_page(alert="Quiet hours must be written as HH:MM.", values=fields)
        self.settings = fields
        self.record("taskmere.settings", fields)
        return self._settings_page(notice="Your notification settings were saved.")

    # --- Quillmont Careers: a two-step application, with a resume on some jobs ---

    def _job_values(self, job: Job) -> dict[str, str]:
        return {
            "job_id": esc(job.id),
            "job_title": esc(job.title),
            "team": esc(job.team),
            "location": esc(job.location),
            "pay": esc(job.pay),
        }

    def careers_job(self, _request: Request, job_id: str) -> Response:
        job = JOBS.get(job_id)
        if job is None:
            return self.handle_missing()
        return self.page("careers", f"{job.title} - Quillmont Careers", "careers_job.html", self._job_values(job))

    def handle_missing(self) -> Response:
        return Response(404, b"<!doctype html><title>Not found</title><h1>Page not found</h1>")

    def _application(self, request: Request, job: Job) -> dict[str, Any]:
        return request.session.setdefault("applications", {}).setdefault(job.id, {"fields": {}, "resume": None})

    def _step1_page(self, job: Job, draft: Mapping[str, Any], alert: str = "") -> Response:
        resume = ""
        if job.needs_resume:
            attached = f"Attached: {esc(draft['resume'].name)}" if draft["resume"] else "No file chosen"
            resume = fill(page_text("careers_resume.html"), {"attached": attached})
        values = {
            **self._job_values(job),
            "enctype": 'enctype="multipart/form-data"' if job.needs_resume else "",
            "resume": resume,
        }
        return self.page(
            "careers", f"Apply: {job.title} - Quillmont Careers", "careers_step1.html", values,
            alert=alert, form_values=draft["fields"],
        )  # fmt: skip

    def careers_step1(self, request: Request, job_id: str) -> Response:
        job = JOBS.get(job_id)
        if job is None:
            return self.handle_missing()
        return self._step1_page(job, self._application(request, job))

    def careers_step1_post(self, request: Request, job_id: str) -> Response:
        job = JOBS.get(job_id)
        if job is None:
            return self.handle_missing()
        draft = self._application(request, job)
        draft["fields"].update({name: request.form.get(name, "").strip() for name in STEP1_FIELDS})
        missing = [label for name, label in STEP1_REQUIRED.items() if not draft["fields"][name]]
        resume = next((u for u in request.uploads if u.field == "resume"), None)
        problem = ""
        if job.needs_resume:
            if resume and not (resume.name.lower().endswith(RESUME_TYPES) and resume.size <= MAX_RESUME):
                problem = "The resume must be a PDF, DOC or DOCX file of at most 5 MB."
                resume = None
            draft["resume"] = resume or draft["resume"]
            if not draft["resume"]:
                missing.append("Resume/CV")
        if missing or problem:
            return self._step1_page(job, draft, problem or f"Please fill in: {', '.join(missing)}.")
        return redirect(f"/careers/apply/{job.id}/2")

    def _step2_page(self, job: Job, draft: Mapping[str, Any], alert: str = "", notice: str = "") -> Response:
        fields = draft["fields"]
        name = f"{fields.get('first_name', '')} {fields.get('last_name', '')}".strip()
        values = {**self._job_values(job), "applicant": esc(name)}
        return self.page(
            "careers", f"Apply: {job.title} - Quillmont Careers", "careers_step2.html", values,
            alert=alert, notice=notice, form_values={k: v for k, v in fields.items() if k in STEP2_FIELDS},
        )  # fmt: skip

    def careers_step2(self, request: Request, job_id: str) -> Response:
        job = JOBS.get(job_id)
        if job is None:
            return self.handle_missing()
        draft = self._application(request, job)
        if not draft["fields"].get("first_name"):
            return redirect(f"/careers/apply/{job.id}/1")
        return self._step2_page(job, draft)

    def careers_step2_post(self, request: Request, job_id: str) -> Response:
        job = JOBS.get(job_id)
        if job is None:
            return self.handle_missing()
        draft = self._application(request, job)
        fields = draft["fields"]
        if not fields.get("first_name"):
            return redirect(f"/careers/apply/{job.id}/1")
        fields.update({name: request.form.get(name, "").strip() for name in STEP2_FIELDS})
        if request.form.get("action") == "draft":
            return self._step2_page(job, draft, notice="Your draft was saved.")
        missing = [label for name, label in STEP2_REQUIRED.items() if not fields[name]]
        files = [draft["resume"]] if draft["resume"] else []
        row_fields = {"job": job.id, **fields}
        if missing:
            self.record("careers.application", row_fields, status="refused", files=files)
            return self._step2_page(job, draft, f"Please answer: {', '.join(missing)}.")
        self.record("careers.application", row_fields, files=files)
        return redirect(f"/careers/apply/{job.id}/done")

    def careers_done(self, request: Request, job_id: str) -> Response:
        job = JOBS.get(job_id)
        if job is None:
            return self.handle_missing()
        fields = self._application(request, job)["fields"]
        values = {
            **self._job_values(job),
            "applicant": esc(f"{fields.get('first_name', '')} {fields.get('last_name', '')}".strip()),
            "reference": f"QM-{job.id}-0417",
        }
        return self.page("careers", "Application received - Quillmont Careers", "careers_done.html", values)

    # --- Kestrelmoor Outfitters: a cart, shipping, payment ---

    @staticmethod
    def subtotal() -> float:
        return sum(quantity * price for _, quantity, price in CART)

    def _order(self, request: Request) -> dict[str, Any]:
        return request.session.setdefault("order", {"fields": {}, "gift_code": ""})

    def _total(self, order: Mapping[str, Any]) -> float:
        delivery = DELIVERY[order["fields"].get("delivery") or "standard"][1]
        return self.subtotal() + delivery - (GIFT_VALUE if order["gift_code"] else 0.0)

    def shop_cart(self, _request: Request) -> Response:
        rows = "".join(
            f"<tr><td>{esc(name)}</td><td>{quantity}</td><td>{money(quantity * price)}</td></tr>"
            for name, quantity, price in CART
        )
        values = {"lines": rows, "subtotal": money(self.subtotal())}
        return self.page("shop", "Your cart - Kestrelmoor Outfitters", "shop_cart.html", values)

    def _shipping_page(self, order: Mapping[str, Any], alert: str = "") -> Response:
        return self.page(
            "shop", "Shipping - Kestrelmoor Outfitters - Checkout", "shop_shipping.html",
            alert=alert, form_values=order["fields"],
        )  # fmt: skip

    def shop_shipping(self, request: Request) -> Response:
        return self._shipping_page(self._order(request))

    def shop_shipping_post(self, request: Request) -> Response:
        order = self._order(request)
        order["fields"].update({name: request.form.get(name, "").strip() for name in SHIPPING_FIELDS})
        missing = [label for name, label in SHIPPING_REQUIRED.items() if not order["fields"][name]]
        if order["fields"]["delivery"] not in DELIVERY:
            missing.append("Delivery method")
        if missing:
            return self._shipping_page(order, f"Please fill in: {', '.join(missing)}.")
        return redirect("/shop/checkout/payment")

    def _payment_page(
        self, order: Mapping[str, Any], *, alert: str = "", notice: str = "", values: Mapping[str, str] | None = None
    ) -> Response:
        fields = order["fields"]
        name, cost = DELIVERY[fields["delivery"]]
        discount = ""
        if order["gift_code"]:
            discount = f"<tr><th>Gift card {esc(order['gift_code'])}</th><td>-{money(GIFT_VALUE)}</td></tr>"
        page_values = {
            "ship_to": esc(f"{fields['first_name']} {fields['last_name']}, {fields['city']}"),
            "delivery": esc(name),
            "delivery_cost": money(cost) if cost else "Free",
            "subtotal": money(self.subtotal()),
            "discount": discount,
            "total": money(self._total(order)),
        }
        return self.page(
            "shop", "Payment - Kestrelmoor Outfitters - Checkout", "shop_payment.html", page_values,
            alert=alert, notice=notice, form_values=values,
        )  # fmt: skip

    def shop_payment(self, request: Request) -> Response:
        order = self._order(request)
        if not order["fields"].get("email"):
            return redirect("/shop/checkout")
        return self._payment_page(order)

    def shop_payment_post(self, request: Request) -> Response:
        order = self._order(request)
        if not order["fields"].get("email"):
            return redirect("/shop/checkout")
        kept = {name: request.form.get(name, "") for name in ("payment", "terms")}
        code = request.form.get("gift_code", "").strip()
        if request.form.get("action") == "apply" or (code and code.upper() != order["gift_code"]):
            if code.upper() != GIFT_CODE:
                return self._payment_page(order, alert="That code is not valid.", values=kept)
            order["gift_code"] = GIFT_CODE
            if request.form.get("action") == "apply":
                return self._payment_page(order, notice="Your gift card was applied.", values=kept)
        fields = {
            **order["fields"],
            "gift_code": order["gift_code"],
            "payment": kept["payment"],
            "terms": kept["terms"],
            "total": f"{self._total(order):.2f}",
        }
        missing = [
            label for name, label in (("payment", "Payment method"), ("terms", "Terms of sale")) if not kept[name]
        ]
        if missing:
            self.record("shop.order", fields, status="refused")
            return self._payment_page(order, alert=f"Please complete: {', '.join(missing)}.", values=kept)
        self.record("shop.order", fields)
        return redirect("/shop/order")

    def shop_order(self, request: Request) -> Response:
        order = self._order(request)
        values = {
            "reference": "KM-104417",
            "total": money(self._total(order)) if order["fields"].get("delivery") else "$0.00",
            "email": esc(order["fields"].get("email", "")),
        }
        return self.page("shop", "Order confirmed - Kestrelmoor Outfitters", "shop_order.html", values)

    # --- Fennick Supply: search, filter, open a result ---

    def supply_search(self, request: Request) -> Response:
        query = request.query.get("q", "").strip()
        material = request.query.get("material", "")
        length = request.query.get("length", "")
        words = query.lower().split()
        parts = [
            p
            for p in CATALOG
            if all(w in p.name.lower() for w in words)
            and (not material or p.material == material)
            and (not length or str(p.length) == length)
        ]
        rows = "".join(
            f'<tr><td><a href="/supply/item/{esc(p.sku)}">{esc(p.name)}</a></td><td>{esc(p.thread)}</td>'
            f"<td>{p.length} mm</td><td>{esc(p.material)}</td><td>{esc(p.price)}</td></tr>"
            for p in parts
        )
        values = {"count": f"{len(parts)} products", "rows": rows}
        return self.page(
            "supply", "Catalog - Fennick Supply", "supply_search.html", values,
            form_values={"q": query, "material": material, "length": length},
        )  # fmt: skip

    def supply_item(self, _request: Request, sku: str) -> Response:
        part = next((p for p in CATALOG if p.sku == sku), None)
        if part is None:
            return self.handle_missing()
        values = {
            "name": esc(part.name), "sku": esc(part.sku), "thread": esc(part.thread), "length": part.length,
            "material": esc(part.material), "pack": part.pack, "price": esc(part.price), "mpn": esc(part.mpn),
        }  # fmt: skip
        return self.page("supply", f"{part.name} - Fennick Supply", "supply_item.html", values)

    # --- Harborline Community: a sign-in, then the daily check-in ---

    def forum_home(self, request: Request, alert: str = "", notice: str = "") -> Response:
        user = request.session.get("forum")
        streak = 3
        if not user:
            card = page_text("forum_checkin_signed_out.html")
        elif self.accepted("forum.checkin"):
            card = fill(page_text("forum_checkin_done.html"), {"streak": streak + 1})
        else:
            card = fill(page_text("forum_checkin_form.html"), {"user": esc(user), "streak": streak})
        threads = "".join(
            f'<tr><td><a href="#">{esc(title)}</a></td><td>{esc(author)}</td><td>{replies}</td></tr>'
            for title, author, replies in THREADS
        )
        values = {"checkin": card, "threads": threads}
        return self.page(
            "forum", "All threads - Harborline Community", "forum_home.html", values, alert=alert, notice=notice
        )

    def forum_signin(self, _request: Request) -> Response:
        return self._signin_page("forum", "/forum/signin")

    def forum_signin_post(self, request: Request) -> Response:
        if self._sign_in("forum.signin", FORUM_ACCOUNT, request):
            return redirect("/forum/")
        return self._signin_page("forum", "/forum/signin", "That username and password do not match.")

    def forum_checkin_post(self, request: Request) -> Response:
        user = request.session.get("forum")
        if not user:
            return redirect("/forum/signin")
        fields = {
            "user": user,
            "mood": request.form.get("mood", "").strip(),
            "note": request.form.get("note", "").strip(),
        }
        if not fields["mood"]:
            self.record("forum.checkin", fields, status="refused")
            return self.forum_home(request, alert="Please choose how today is going.")
        row = self.record("forum.checkin", fields)
        if row["status"] == "duplicate":
            return self.forum_home(request, alert="You have already checked in today.")
        return self.forum_home(request, notice="Checked in. See you tomorrow.")

    # --- Brindle Market: a plain picker for a listing's photos ---

    def market_photos(self, _request: Request, alert: str = "") -> Response:
        tiles = "".join(
            f'<button type="button" class="tile" aria-pressed="false" aria-label="Photo {n}" data-tile="{n}">'
            f'<img src="/static/tiles/tile-{n:02d}.svg" alt=""></button>'
            for n in range(1, TILE_COUNT + 1)
        )
        return self.page(
            "market", "Choose photos - Brindle Market", "market_photos.html", {"tiles": tiles}, alert=alert
        )

    def market_photos_post(self, request: Request) -> Response:
        chosen = sorted({int(t) for t in request.form.get("selected", "").split(",") if t.strip().isdigit()})
        if not chosen:
            self.record("market.photos", {"selected": []}, status="refused")
            return self.market_photos(request, alert="Choose at least one photo.")
        self.record("market.photos", {"selected": chosen})
        return redirect("/market/listing/done")

    def market_done(self, _request: Request) -> Response:
        rows = self.accepted("market.photos")
        count = len(rows[-1]["fields"]["selected"]) if rows else 0
        return self.page("market", "Photos added - Brindle Market", "market_done.html", {"count": count})


class _Handler(BaseHTTPRequestHandler):
    owner: Site

    def log_message(self, *_args: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - the name http.server looks for
        self._serve()

    def do_POST(self) -> None:  # noqa: N802
        self._serve()

    def _session(self) -> tuple[str, dict[str, Any], bool]:
        cookie = next(
            (c.strip()[4:] for c in self.headers.get("Cookie", "").split(";") if c.strip().startswith("sid=")), ""
        )
        with self.owner.lock:
            if cookie in self.owner.sessions:
                return cookie, self.owner.sessions[cookie], False
            sid = secrets.token_hex(12)
            self.owner.sessions[sid] = {}
            return sid, self.owner.sessions[sid], True

    def _serve(self) -> None:
        parts = urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else b""
        form, uploads = parse_body(self.headers.get("Content-Type", ""), body) if body else ({}, [])
        sid, session, is_new = self._session()
        request = Request(
            self.command, parts.path, dict(parse_qsl(parts.query, keep_blank_values=True)), form, uploads, session
        )
        response = self.owner.handle(request)
        self.owner._append(
            "requests.jsonl",
            {"t": round(time.time(), 3), "method": self.command, "path": self.path, "status": response.status},
        )
        self.owner.requests.append({"method": self.command, "path": parts.path, "status": response.status})
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("Cache-Control", "no-store")
        if response.location:
            self.send_header("Location", response.location)
        if is_new:
            self.send_header("Set-Cookie", f"sid={sid}; Path=/; HttpOnly; SameSite=Lax")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response.body)


def main() -> int:
    site = Site()
    print(f"{site.base_url} (Ctrl-C ends it)")
    try:
        site.thread.join()
    except KeyboardInterrupt:
        site.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
