#!/usr/bin/env python3
"""Replace third-party app names in result files with `<app>` before they are committed.

Stdlib only; run with /usr/bin/python3 -I. Usage: redact.py [--check] FILE...

The monitor TSVs and the pre-flight blocks in raw headers name foreign processes, which
can include the user's own apps. Apple processes and the `claude` agent sessions are kept:
they explain disturbances and are part of the setup (METHODOLOGY.md, "Quiet machine").
Every other app bundle is replaced, in the three places a process name appears:

  busy lists     `Name(1234) 30%`
  top lists      `top: Name 2.3, ...`
  paths          `/Applications/Name.app/...`

Names come from APPS below; paths are judged by the bundle's CFBundleIdentifier. quietlib.py
applies the same rule at sample time. --check only reports, and exits 1 if anything would
change.
"""
from __future__ import annotations

import plistlib
import re
import sys
from functools import lru_cache
from pathlib import Path

# Third-party process names seen in this repo's results on 2026-10-18 (reviewed by hand).
APPS = ["Discord Helper (Renderer)", "Google Drive", "WeChat", "Claude"]
_NAME = "|".join(re.escape(n) for n in APPS)
BUSY = re.compile(rf"(?<![\w.-])(?:{_NAME})(?=\(\d+\) [\d.]+%)")
TOP = re.compile(rf"(?<=[:,] )(?:{_NAME})(?= \d+(?:\.\d+)?(?:,|\\n|\n|$))")
PATH = re.compile(r"(?:~|/Users/[^/\s]+)?/Applications/[^\\\n\"\t;]*?\.app(?:/[^\\\n\"\t;,]*)?")


@lru_cache(maxsize=None)
def apple_bundle(app: str) -> bool:
    """True when the bundle at `app` (path ending in .app) has a com.apple. identifier."""
    try:
        with open(Path(app).expanduser() / "Contents" / "Info.plist", "rb") as f:
            return str(plistlib.load(f).get("CFBundleIdentifier", "")).startswith("com.apple.")
    except (OSError, plistlib.InvalidFileException, ValueError):
        return False


def redact(text: str) -> str:
    text = BUSY.sub("<app>", text)
    text = TOP.sub("<app>", text)

    def path(m: re.Match) -> str:
        p = m.group(0)
        outer = p.split(".app", 1)[0] + ".app"
        return p if apple_bundle(outer) else "<app>"

    return PATH.sub(path, text)


def main(argv: list[str]) -> int:
    check = "--check" in argv
    changed = 0
    for name in (a for a in argv if a != "--check"):
        f = Path(name)
        old = f.read_text(encoding="utf-8")
        new = redact(old)
        if new == old:
            continue
        changed += 1
        print(f"redact: {f}: {sum(1 for a, b in zip(old.splitlines(), new.splitlines()) if a != b)} line(s)")
        if not check:
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_text(new, encoding="utf-8")
            tmp.replace(f)
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
