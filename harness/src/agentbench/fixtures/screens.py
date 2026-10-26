"""Synthetic fake-UI screenshots (Pillow) plus an accessibility-tree description of each.

Every screen is 1280x1024 and is drawn from a seeded `random.Random`, with Pillow's
bundled font (`ImageFont.load_default(size)`), so the PNG bytes are stable for a given
Pillow version. The scenes are generic desktop apps (editor, terminal, browser, file
manager, settings, notes, calendar) for a fictional project. No real screen content.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .textgen import TextGen

W, H = 1280, 1024
CODE_ALPHABET = "ACDEFHJKMNPQRTUVWXY3479"  # no look-alikes (0/O, 1/I/L, 5/S, 8/B, 2/Z, 6/G)


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size)


@dataclass
class Element:
    role: str
    label: str
    box: tuple[int, int, int, int]
    state: str = ""
    depth: int = 0

    def line(self) -> str:
        st = f" {self.state}" if self.state else ""
        return f"{'  ' * self.depth}[{self.role}] \"{self.label}\" {self.box}{st}"


@dataclass
class Theme:
    dark: bool
    bg: tuple[int, int, int]
    panel: tuple[int, int, int]
    text: tuple[int, int, int]
    muted: tuple[int, int, int]
    accent: tuple[int, int, int]
    line: tuple[int, int, int]


ACCENTS = [(10, 132, 255), (52, 199, 89), (255, 149, 0), (175, 82, 222), (255, 59, 48), (90, 200, 250),
           (88, 86, 214), (255, 45, 85), (0, 150, 136), (121, 85, 72)]


def make_theme(r: random.Random) -> Theme:
    accent = ACCENTS[r.randrange(len(ACCENTS))]
    if r.random() < 0.4:
        return Theme(True, (30, 31, 34), (43, 45, 49), (230, 230, 232), (150, 152, 158), accent, (60, 62, 68))
    return Theme(False, (250, 250, 251), (238, 239, 242), (28, 28, 30), (110, 112, 118), accent, (214, 216, 220))


@dataclass
class Screen:
    image: Image.Image
    elements: list[Element] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class Painter:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)
        self.g = TextGen(seed + 7)
        self.theme = make_theme(self.r)
        hue = [self.r.randint(40, 120), self.r.randint(60, 140), self.r.randint(90, 170)]
        self.r.shuffle(hue)
        self.img = Image.new("RGB", (W, H), tuple(hue))
        self.d = ImageDraw.Draw(self.img)
        self.els: list[Element] = []

    # ---------------------------------------------------------------- basics
    def text(self, xy: tuple[int, int], s: str, size: int = 14, fill=None) -> int:
        f = font(size)
        self.d.text(xy, s, fill=fill or self.theme.text, font=f)
        return int(self.d.textlength(s, font=f))

    def el(self, role: str, label: str, box: tuple[int, int, int, int], state: str = "", depth: int = 1) -> None:
        self.els.append(Element(role, label, tuple(int(v) for v in box), state, depth))

    def button(self, box, label, primary=False, depth=2, state="") -> None:
        t = self.theme
        self.d.rounded_rectangle(box, radius=6, fill=t.accent if primary else t.panel, outline=t.line)
        f = font(14)
        tw = self.d.textlength(label, font=f)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.d.text((cx - tw / 2, cy - 8), label, fill=(255, 255, 255) if primary else t.text, font=f)
        self.el("button", label, box, state, depth)

    def menu_bar(self, app: str) -> None:
        t = self.theme
        self.d.rectangle((0, 0, W, 26), fill=(236, 236, 238) if not t.dark else (40, 40, 44))
        x = 14
        self.d.ellipse((x, 8, x + 11, 19), fill=(90, 90, 96))
        x += 26
        for i, item in enumerate([app, "File", "Edit", "View", "Window", "Help"]):
            w = self.text((x, 5), item, 14 if i else 15, t.text)
            self.el("menuitem", item, (x - 4, 0, x + w + 4, 26), depth=1)
            x += w + 22
        clock = f"{self.r.randint(7, 18):02d}:{self.r.randint(0, 59):02d}"
        self.text((W - 70, 5), clock, 14, t.text)
        self.els.insert(0, Element("menubar", app, (0, 0, W, 26), "", 0))
        self.el("static", f"clock {clock}", (W - 72, 0, W - 20, 26), depth=1)

    def dock(self, apps: list[str], active: str) -> None:
        n = len(apps)
        width = n * 58 + 20
        x0 = (W - width) // 2
        self.d.rounded_rectangle((x0, H - 72, x0 + width, H - 8), radius=16, fill=(210, 212, 218), outline=(180, 182, 188))
        self.el("toolbar", "Dock", (x0, H - 72, x0 + width, H - 8), depth=0)
        for i, a in enumerate(apps):
            bx = x0 + 14 + i * 58
            col = ACCENTS[i % len(ACCENTS)]
            self.d.rounded_rectangle((bx, H - 64, bx + 46, H - 18), radius=11, fill=col)
            self.text((bx + 15, H - 52), a[0], 20, (255, 255, 255))
            if a == active:
                self.d.ellipse((bx + 20, H - 14, bx + 26, H - 9), fill=(60, 60, 60))
            self.el("button", a, (bx, H - 64, bx + 46, H - 18), "running" if a == active else "", 1)

    def window(self, box, title: str, focused=True) -> tuple[int, int, int, int]:
        t = self.theme
        x0, y0, x1, y1 = box
        self.d.rounded_rectangle((x0 + 4, y0 + 6, x1 + 4, y1 + 8), radius=10, fill=(0, 0, 0))
        self.d.rounded_rectangle(box, radius=10, fill=t.bg, outline=t.line)
        self.d.rounded_rectangle((x0, y0, x1, y0 + 34), radius=10, fill=t.panel)
        self.d.rectangle((x0, y0 + 24, x1, y0 + 34), fill=t.panel)
        for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            self.d.ellipse((x0 + 14 + i * 20, y0 + 11, x0 + 26 + i * 20, y0 + 23), fill=c)
        f = font(14)
        tw = self.d.textlength(title, font=f)
        self.d.text(((x0 + x1) / 2 - tw / 2, y0 + 9), title, fill=t.text, font=f)
        self.el("window", title, box, "focused" if focused else "", 0)
        return (x0 + 1, y0 + 35, x1 - 1, y1 - 1)

    def cursor(self) -> None:
        x, y = self.r.randint(100, W - 100), self.r.randint(80, H - 120)
        self.d.polygon([(x, y), (x, y + 20), (x + 5, y + 15), (x + 9, y + 24), (x + 12, y + 23), (x + 8, y + 14),
                        (x + 15, y + 14)], fill=(0, 0, 0), outline=(255, 255, 255))

    # ------------------------------------------------------------- contents
    def editor(self, c, file_name: str) -> None:
        t = self.theme
        x0, y0, x1, y1 = c
        self.d.rectangle((x0, y0, x0 + 220, y1), fill=t.panel)
        self.el("list", "Explorer", (x0, y0, x0 + 220, y1), depth=1)
        files = ["api.py", "config/", "jobs/", "parser.py", "storage.py", "errors.py", "__init__.py",
                 "defaults.yaml", "conftest.py", "test_parser.py", "test_api.py", "CHANGELOG.md", "README.md"]
        self.r.shuffle(files)
        for i, f in enumerate(files[: (y1 - y0 - 20) // 26]):
            sel = f == file_name
            y = y0 + 12 + i * 26
            if sel:
                self.d.rectangle((x0, y - 3, x0 + 220, y + 21), fill=t.accent)
            self.text((x0 + 18, y), f, 14, (255, 255, 255) if sel else t.text)
            self.el("list_item", f, (x0, y - 3, x0 + 220, y + 21), "selected" if sel else "", 2)
        self.d.rectangle((x0 + 220, y0, x1, y0 + 32), fill=t.panel)
        tabs = [file_name] + [self.g.ident() + ".py" for _ in range(2)]
        tx = x0 + 230
        for i, tab in enumerate(tabs):
            w = int(self.d.textlength(tab, font=font(13))) + 28
            self.d.rectangle((tx, y0 + 4, tx + w, y0 + 32), fill=t.bg if i == 0 else t.panel, outline=t.line)
            self.text((tx + 14, y0 + 11), tab, 13)
            self.el("tab", tab, (tx, y0 + 4, tx + w, y0 + 32), "selected" if i == 0 else "", 2)
            tx += w + 4
        y = y0 + 46
        n = 1
        code_lines = []
        while y < y1 - 24:
            block = self.g.code().splitlines()[1:-1]
            for ln in block:
                if y >= y1 - 24:
                    break
                self.text((x0 + 232, y), f"{n:>3}", 13, t.muted)
                col = t.accent if ln.lstrip().startswith(("def ", "func ", "pub fn", "SELECT", "export", "[")) else t.text
                self.text((x0 + 272, y), ln[:90].replace("\t", "    "), 13, col)
                code_lines.append(ln.strip())
                n += 1
                y += 20
            y += 20
            n += 1
        self.el("textbox", f"Editor {file_name}", (x0 + 220, y0 + 32, x1, y1), "focused", 1)
        for i, ln in enumerate(code_lines[:12]):
            self.el("static", ln[:70], (x0 + 272, y0 + 46 + 20 * i, x1 - 10, y0 + 64 + 20 * i), depth=2)

    def terminal(self, c, lines: list[str]) -> None:
        x0, y0, x1, y1 = c
        self.d.rectangle(c, fill=(22, 22, 26))
        y = y0 + 10
        for ln in lines:
            if y > y1 - 22:
                break
            col = (110, 220, 120) if ln.startswith("$") else (210, 210, 210)
            self.text((x0 + 12, y), ln[:120], 14, col)
            y += 21
        self.el("textbox", "Terminal output", c, "focused", 1)
        for i, ln in enumerate(lines[:14]):
            self.el("static", ln[:80], (x0 + 12, y0 + 10 + 21 * i, x1 - 12, y0 + 29 + 21 * i), depth=2)

    def browser(self, c, url: str, heading: str) -> None:
        t = self.theme
        x0, y0, x1, y1 = c
        self.d.rectangle((x0, y0, x1, y0 + 38), fill=t.panel)
        tx = x0 + 10
        for i in range(self.r.randint(2, 4)):
            label = self.g.project() + " docs" if i else heading[:22]
            self.d.rounded_rectangle((tx, y0 + 6, tx + 170, y0 + 34), radius=7, fill=t.bg if i == 0 else t.panel,
                                     outline=t.line)
            self.text((tx + 10, y0 + 12), label[:20], 13)
            self.el("tab", label, (tx, y0 + 6, tx + 170, y0 + 34), "selected" if i == 0 else "", 2)
            tx += 176
        self.d.rounded_rectangle((x0 + 60, y0 + 44, x1 - 60, y0 + 74), radius=8, fill=t.bg, outline=t.line)
        self.text((x0 + 76, y0 + 51), url, 14)
        self.el("textbox", "Address", (x0 + 60, y0 + 44, x1 - 60, y0 + 74), f"value={url}", 1)
        y = y0 + 100
        self.text((x0 + 60, y), heading, 26)
        self.el("heading", heading, (x0 + 60, y, x1 - 60, y + 32), depth=1)
        y += 50
        while y < y1 - 80:
            para = self.g.sentence()
            for chunk in [para[i:i + 105] for i in range(0, len(para), 105)]:
                if y >= y1 - 80:
                    break
                self.text((x0 + 60, y), chunk, 15)
                y += 22
            self.el("static", para[:90], (x0 + 60, y - 22, x1 - 60, y), depth=1)
            y += 14
        self.button((x0 + 60, y1 - 60, x0 + 200, y1 - 24), "Reload", True, depth=1)

    def files(self, c, folder: str, names: list[str], highlight: str | None) -> dict[str, Any]:
        t = self.theme
        x0, y0, x1, y1 = c
        self.d.rectangle((x0, y0, x0 + 190, y1), fill=t.panel)
        for i, s in enumerate(["Workspace", "Downloads", "Trash", "Projects", "Recent"]):
            self.text((x0 + 20, y0 + 16 + i * 28), s, 14)
            self.el("list_item", s, (x0, y0 + 10 + i * 28, x0 + 190, y0 + 36 + i * 28), depth=2)
        self.text((x0 + 210, y0 + 12), folder, 15, t.muted)
        self.el("static", f"Path {folder}", (x0 + 210, y0 + 8, x1 - 10, y0 + 32), depth=1)
        hy = y0 + 44
        for col, cx in (("Name", 210), ("Size", 640), ("Modified", 760)):
            self.text((x0 + cx, hy), col, 13, t.muted)
        self.d.line((x0 + 200, hy + 22, x1 - 10, hy + 22), fill=t.line)
        found = {}
        for i, name in enumerate(names):
            y = hy + 32 + i * 28
            if y > y1 - 30:
                break
            box = (x0 + 200, y - 4, x1 - 10, y + 22)
            sel = name == highlight
            if sel:
                self.d.rectangle(box, fill=t.accent)
            fg = (255, 255, 255) if sel else t.text
            size = f"{self.r.randint(1, 900)} KB"
            date = f"2026-{self.r.randint(1, 10):02d}-{self.r.randint(1, 28):02d}"
            self.text((x0 + 210, y), name, 14, fg)
            self.text((x0 + 640, y), size, 14, fg)
            self.text((x0 + 760, y), date, 14, fg)
            self.el("list_item", f"{name}, {size}, {date}", box, "selected" if sel else "", 2)
            if sel:
                found = {"name": name, "box": box}
        return found

    def settings(self, c, section: str) -> None:
        t = self.theme
        x0, y0, x1, y1 = c
        self.d.rectangle((x0, y0, x0 + 210, y1), fill=t.panel)
        for i, s in enumerate(["Editor", "Files", "Terminal", "Extensions", "Display", "Keyboard", "Appearance"]):
            y = y0 + 14 + i * 30
            if s == section:
                self.d.rounded_rectangle((x0 + 8, y - 5, x0 + 202, y + 21), radius=6, fill=t.accent)
            self.text((x0 + 22, y), s, 14, (255, 255, 255) if s == section else t.text)
            self.el("tab", s, (x0 + 8, y - 5, x0 + 202, y + 21), "selected" if s == section else "", 2)
        y = y0 + 20
        self.text((x0 + 240, y), section, 22)
        y += 46
        opts = ["Format on save", "Word wrap", "Show minimap", "Trim trailing whitespace", "Auto save",
                "Render whitespace", "Bracket pair colors", "Sticky scroll", "Insert final newline"]
        self.r.shuffle(opts)
        for o in opts:
            if y > y1 - 60:
                break
            on = self.r.random() < 0.5
            self.d.rounded_rectangle((x0 + 240, y, x0 + 258, y + 18), radius=4, fill=t.accent if on else t.bg,
                                     outline=t.line)
            if on:
                self.d.line((x0 + 244, y + 9, x0 + 248, y + 14, x0 + 255, y + 4), fill=(255, 255, 255), width=2)
            self.text((x0 + 270, y), o, 15)
            self.text((x0 + 270, y + 20), self.g.sentence()[:80], 12, t.muted)
            self.el("checkbox", o, (x0 + 240, y, x0 + 258, y + 18), "checked" if on else "unchecked", 2)
            y += 52
        self.d.rounded_rectangle((x0 + 240, y, x0 + 420, y + 28), radius=6, fill=t.bg, outline=t.line)
        size = self.r.choice([2, 4, 8])
        self.text((x0 + 250, y + 6), f"Tab size: {size}", 14)
        self.el("combobox", "Tab size", (x0 + 240, y, x0 + 420, y + 28), f"value={size}", 2)

    def notes(self, c) -> None:
        t = self.theme
        x0, y0, x1, y1 = c
        self.d.rectangle((x0, y0, x0 + 260, y1), fill=t.panel)
        for i in range(8):
            y = y0 + 12 + i * 56
            title = self.g.project() + " " + self.g.pick(["notes", "checklist", "ideas", "summary", "draft"])
            self.text((x0 + 16, y), title, 14)
            self.text((x0 + 16, y + 20), self.g.sentence()[:34], 12, t.muted)
            self.el("list_item", title, (x0, y - 4, x0 + 260, y + 44), "selected" if i == 0 else "", 2)
        y = y0 + 20
        self.text((x0 + 290, y), "Release notes draft", 22)
        self.el("heading", "Release notes draft", (x0 + 290, y, x1 - 20, y + 28), depth=1)
        y += 44
        for _ in range(9):
            s = "- " + self.g.sentence()[:95]
            self.text((x0 + 290, y), s, 14)
            self.el("static", s[:80], (x0 + 290, y, x1 - 20, y + 20), depth=2)
            y += 30

    def calendar(self, c) -> None:
        t = self.theme
        x0, y0, x1, y1 = c
        days = ["Mon 2", "Tue 3", "Wed 4", "Thu 5", "Fri 6"]
        colw = (x1 - x0 - 70) // 5
        for i, dname in enumerate(days):
            self.text((x0 + 70 + i * colw + 10, y0 + 10), dname, 14, t.muted)
        for h in range(9):
            y = y0 + 40 + h * 60
            self.text((x0 + 10, y), f"{8 + h:02d}:00", 12, t.muted)
            self.d.line((x0 + 64, y, x1, y), fill=t.line)
        events = ["Design review", "Team sync", "Release planning", "Focus time", "Build triage",
                  "Docs pass", "Retro", "Demo prep"]
        for _ in range(self.r.randint(5, 8)):
            day, start = self.r.randrange(5), self.r.randrange(8)
            name = self.g.pick(events)
            box = (x0 + 72 + day * colw, y0 + 42 + start * 60, x0 + 66 + (day + 1) * colw, y0 + 96 + start * 60)
            self.d.rounded_rectangle(box, radius=6, fill=self.theme.accent)
            self.text((box[0] + 8, box[1] + 6), name, 13, (255, 255, 255))
            self.el("button", f"{name}, {days[day]} {8 + start:02d}:00", box, depth=2)

    def dialog(self, title: str, message: str, buttons: list[str], width=520, height=200) -> dict[str, Any]:
        t = self.theme
        x0 = (W - width) // 2 + self.r.randint(-60, 60)
        y0 = (H - height) // 2 + self.r.randint(-80, 40)
        box = (x0, y0, x0 + width, y0 + height)
        self.d.rounded_rectangle((x0 + 6, y0 + 8, x0 + width + 6, y0 + height + 10), radius=12, fill=(0, 0, 0))
        self.d.rounded_rectangle(box, radius=12, fill=t.bg, outline=t.line, width=2)
        self.text((x0 + 24, y0 + 22), title, 18)
        self.el("dialog", title, box, "modal", 0)
        self.text((x0 + 24, y0 + 60), message, 14, t.muted)
        self.el("static", message, (x0 + 24, y0 + 56, x0 + width - 24, y0 + 80), depth=1)
        bx = x0 + width - 24
        out = {}
        for i, b in enumerate(reversed(buttons)):
            w = max(96, int(self.d.textlength(b, font=font(14))) + 36)
            bb = (bx - w, y0 + height - 56, bx, y0 + height - 22)
            self.button(bb, b, primary=(i == 0), depth=1)
            out[b] = bb
            bx -= w + 12
        return {"box": box, "buttons": out}

    def finish(self, meta: dict[str, Any] | None = None) -> Screen:
        self.cursor()
        return Screen(self.img, self.els, meta or {})


APPS = ["Code Editor", "Terminal", "Browser", "Files", "Settings", "Notes", "Calendar", "Image Viewer"]
KINDS = ["editor", "terminal", "browser", "files", "settings", "notes", "calendar"]


def _window_box(r: random.Random, wide: bool = True) -> tuple[int, int, int, int]:
    x0 = r.randint(20, 120)
    y0 = r.randint(40, 90)
    x1 = W - r.randint(20, 120) if wide else x0 + r.randint(760, 1000)
    y1 = H - r.randint(90, 140)
    return (x0, y0, x1, y1)


def render_screen(seed: int, kind: str | None = None, title: str | None = None) -> Screen:
    """A generic desktop scene. `kind` picks the focused app."""
    p = Painter(seed)
    r = p.r
    kind = kind or KINDS[r.randrange(len(KINDS))]
    app = {"editor": "Code Editor", "terminal": "Terminal", "browser": "Browser", "files": "Files",
           "settings": "Settings", "notes": "Notes", "calendar": "Calendar"}[kind]
    p.menu_bar(app)
    if r.random() < 0.5:  # a background window
        bb = _window_box(r, wide=False)
        bb = (bb[0] + 140, bb[1] + 30, min(W - 10, bb[2] + 180), bb[3] - 40)
        p.window(bb, f"{p.g.project()} - {r.choice(APPS)}", focused=False)
    box = _window_box(r)
    ttl = title or {"editor": "demo-app - parser.py", "terminal": "Terminal - /workspace/demo-app",
                    "browser": "Browser - localhost", "files": "Files - demo-app",
                    "settings": "Settings", "notes": "Notes", "calendar": "Calendar - November 2026"}[kind]
    c = p.window(box, ttl)
    meta: dict[str, Any] = {"kind": kind, "title": ttl}
    if kind == "editor":
        p.editor(c, ttl.split(" - ")[-1] if " - " in ttl else "parser.py")
    elif kind == "terminal":
        lines = []
        while len(lines) < 44:
            lines += ["$ " + p.g.pick(["uv run pytest -q", "git status --short", "ls -la src/demo_app",
                                       "uv run demo-app serve --port 8000", "scripts/release.sh 1.4.0",
                                       "tail -n 20 var/log/service.log", "uv run ruff check src tests"])]
            lines += p.g.log().splitlines()[1:-1]
        p.terminal(c, lines)
    elif kind == "browser":
        p.browser(c, f"http://localhost:{r.choice([8000, 8080, 3000])}/{p.g.ident('kebab')}", p.g.project() + " dashboard")
    elif kind == "files":
        names = [p.g.ident("kebab") + r.choice([".py", ".md", ".yaml", ".csv", ".json", ".png", ".tar.gz"])
                 for _ in range(24)]
        p.files(c, "/workspace/demo-app/" + r.choice(["data", "docs", "dist", "src", "notes"]), names, None)
    elif kind == "settings":
        p.settings(c, r.choice(["Editor", "Files", "Terminal", "Appearance"]))
    elif kind == "notes":
        p.notes(c)
    else:
        p.calendar(c)
    if r.random() < 0.25:
        p.dialog(r.choice(["Update available", "Background task finished", "Indexing complete"]),
                 p.g.sentence()[:60], ["Later", "OK"], width=460, height=170)
    p.dock(APPS, app)
    return p.finish(meta)


def render_verify_code(seed: int) -> Screen:
    """G2: a dialog with a large, clearly rendered random verification code (XXX-XXX)."""
    p = Painter(seed)
    r = p.r
    code = "".join(r.choice(CODE_ALPHABET) for _ in range(3)) + "-" + "".join(r.choice(CODE_ALPHABET) for _ in range(3))
    p.menu_bar("Settings")
    c = p.window(_window_box(r), "Settings")
    p.settings(c, "Display")
    t = p.theme
    width, height = 560, 300
    x0, y0 = (W - width) // 2, (H - height) // 2 - 40
    p.d.rounded_rectangle((x0 + 6, y0 + 8, x0 + width + 6, y0 + height + 10), radius=14, fill=(0, 0, 0))
    p.d.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=14, fill=(255, 255, 255), outline=(200, 200, 205), width=2)
    p.text((x0 + 28, y0 + 24), "Confirm device setup", 20, (28, 28, 30))
    p.text((x0 + 28, y0 + 60), "Enter this verification code in the other window:", 15, (90, 90, 96))
    p.d.rounded_rectangle((x0 + 60, y0 + 100, x0 + width - 60, y0 + 196), radius=10, fill=(242, 244, 248),
                          outline=(180, 184, 192), width=2)
    f = font(64)
    tw = p.d.textlength(code, font=f)
    p.d.text((x0 + width / 2 - tw / 2, y0 + 112), code, fill=(10, 10, 12), font=f)
    p.els.append(Element("dialog", "Confirm device setup", (x0, y0, x0 + width, y0 + height), "modal", 0))
    p.button((x0 + width - 140, y0 + height - 60, x0 + width - 28, y0 + height - 24), "Done", True, depth=1)
    p.dock(APPS, "Settings")
    return p.finish({"kind": "g2_verify", "code": code})


def render_save_dialog(seed: int) -> Screen:
    """M7: an unsaved-changes dialog over the editor; the Save button bounds are recorded."""
    p = Painter(seed)
    r = p.r
    p.menu_bar("Code Editor")
    c = p.window(_window_box(r), "demo-app - api.py")
    p.editor(c, "api.py")
    d = p.dialog("Save changes to api.py?", "Your changes will be lost if you don't save them.",
                 ["Don't Save", "Cancel", "Save"], width=540, height=190)
    p.dock(APPS, "Code Editor")
    return p.finish({"kind": "save_dialog", "save_box": list(d["buttons"]["Save"]),
                     "buttons": {k: list(v) for k, v in d["buttons"].items()}})


def render_files_highlight(seed: int, highlight: str) -> Screen:
    """M7: a file manager with one highlighted file whose name the model must read."""
    p = Painter(seed)
    r = p.r
    p.menu_bar("Files")
    c = p.window(_window_box(r), "Files - data")
    names = [p.g.ident("kebab") + r.choice([".csv", ".json", ".md", ".yaml"]) for _ in range(14)]
    names.insert(r.randint(3, 9), highlight)
    found = p.files(c, "/workspace/demo-app/data", names, highlight)
    p.dock(APPS, "Files")
    return p.finish({"kind": "files_highlight", "highlight": highlight, "box": list(found.get("box", ()))})


def a11y_lines(screen: Screen) -> list[str]:
    return [e.line() for e in screen.elements]
