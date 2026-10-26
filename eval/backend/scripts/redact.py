#!/usr/bin/env python3
"""Keep private details out of the public repo (DESIGN.md section 8).

Stdlib only, Python 3.9+; run with /usr/bin/python3 -I -B. Usage: redact.py [--check] FILE...

Without --check, replaces third-party app names with `<app>` in place. The monitor TSVs and the
pre-flight blocks name foreign processes, which can include the user's own apps. Apple
processes and the `claude` agent sessions are kept: they explain disturbances and are part of
the setup. Every other app bundle is replaced, in the three places a process name appears:

  busy lists     `Name(1234) 30%`
  top lists      `top: Name 2.3, ...`
  paths          app bundles under /Applications, by bundle path

Names come from APPS below; paths are judged by the bundle's CFBundleIdentifier. quietlib.py
applies the same rule at sample time. Symlinks and files that are not UTF-8 are skipped and
reported (exit 1); nothing is followed or rewritten through a link.

With --check, changes nothing and exits 1 after printing `path:line: problem` for:
  - a third-party app name that the rewrite would replace;
  - a home path (the users directory followed by a name; write ~ instead);
  - the current host name (socket.gethostname(), its first label, scutil's LocalHostName);
  - a private IPv4 address (10/8, 172.16/12, 192.168/16; 127.0.0.1 is fine);
  - a UUID, or a hex id of 32+ digits, as the value of a machine, user, session or device id key
    (machineID, userID, user_id, session_id, sessionId, device_id, ...); other UUIDs pass;
  - a credential: user info in a URL (a name or secret before the host's @), a GitHub token (the
    gh?_ and github_pat_ prefixes), a Hugging Face token, an sk- API key, or a token option with
    its value (defence in depth: quietlib already reports other sessions' tools by kind only);
  - a file that is not UTF-8 (reported, not a crash), a symlink, or not a regular file;
  - a file under docs/benchmark/<family>/<quant>/backend/<run-id>/capability/ (paths relative to the
    repository root, as the pre-commit hook passes them) that is not on the export allowlist:
    capability/results.json, and capability/<harness>/<run>/ calls.jsonl, summary.json, check.txt,
    with <harness> one of claude_code, hermes, localmelo (modeleval.capability.export.is_allowlisted).
    The backend's own results/<model>/<run-id>/capability/ (the layout before 2026-10-26) is held to
    the same allowlist.
The pre-commit hook (scripts/pre-commit) runs --check on every staged file.
"""
from __future__ import annotations

import os
import plistlib
import re
import socket
import subprocess
import sys
from functools import lru_cache
from pathlib import Path, PurePosixPath

# Third-party process names seen in the old campaign's results on 2026-10-18 (reviewed by hand).
APPS = ["Discord Helper (Renderer)", "Google Drive", "WeChat", "Claude"]
_NAME = "|".join(re.escape(n) for n in APPS)
BUSY = re.compile(rf"(?<![\w.-])(?:{_NAME})(?=\(\d+\) [\d.]+%)")
TOP = re.compile(rf"(?<=[:,] )(?:{_NAME})(?= \d+(?:\.\d+)?(?:,|\\n|\n|$))")
# The users directory is spelled in two parts so that this file passes its own check.
_USERS = "/Us" "ers/"
PATH = re.compile(rf"(?:~|{_USERS}[^/\s]+)?/Applications/[^\\\n\"\t;]*?\.app(?:/[^\\\n\"\t;,]*)?")
# A home path: the users directory followed by a name character (a bare mention of the
# directory, as in DESIGN.md, is fine).
HOME_PATH = re.compile(rf"{_USERS}[A-Za-z0-9_.][^\s/]*")
IPV4 = re.compile(r"(?<![\d.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\.?\d)")
_ID_KEY = r"(?:machine|user|session|device)[_-]?id"
_ID_VALUE = r"(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32,})(?![0-9a-f])"
KEYED_ID = re.compile(rf"(?i){_ID_KEY}(?:\\?[\"'])?\s*[:=]\s*(?:\\?[\"'])?\{{?{_ID_VALUE}")
# Credentials. Each prefix is spelled in two parts so that this file passes its own check.
CREDENTIAL = re.compile(
    r"://[^/@\s\"']+@"                                            # URL user info
    r"|(?<![A-Za-z0-9])(?:" + "gh" + r"[pousr]_|" + "github" + r"_pat_)[A-Za-z0-9_]{20,}"
    r"|(?<![A-Za-z0-9])" + "hf" + r"_[A-Za-z0-9]{20,}"
    r"|(?<![A-Za-z0-9-])" + "sk" + r"-[A-Za-z0-9][A-Za-z0-9_-]{15,}"
    r"|--" + "token" + r"[ =](?!\s)\S+")
# The capability export allowlist, mirrored from modeleval.capability.export.is_allowlisted: this file
# runs under the system Python in the pre-commit hook and cannot import the package
# (tests/integration/test_pipeline.py checks that the two agree).
CAPABILITY_HARNESSES = ("claude_code", "hermes", "localmelo")
CAPABILITY_FILES = ("calls.jsonl", "summary.json", "check.txt")
_RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


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


def hostnames() -> list[str]:
    """This machine's names as they could appear in a file, longest first. Names shorter than 3
    characters and localhost are left out (they would match ordinary text)."""
    names = set()
    full = socket.gethostname()
    candidates = [full, full.split(".")[0]]
    try:
        candidates.append(subprocess.run(["/usr/sbin/scutil", "--get", "LocalHostName"], capture_output=True,
                                         text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    for n in candidates:
        if len(n) >= 3 and n.lower() not in ("localhost", "localhost.localdomain"):
            names.add(n)
    return sorted(names, key=len, reverse=True)


def host_pattern(hosts: list[str]) -> re.Pattern | None:
    if not hosts:
        return None
    alt = "|".join(re.escape(h) for h in sorted(hosts, key=len, reverse=True))
    return re.compile(rf"(?i)(?<![A-Za-z0-9-])(?:{alt})(?![A-Za-z0-9-])")


def private_ipv4(line: str) -> list[str]:
    out = []
    for m in IPV4.finditer(line):
        a, b, c, d = (int(x) for x in m.groups())
        if max(a, b, c, d) > 255:
            continue
        if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
            out.append(m.group(0))
    return out


def scan_text(text: str, hosts: re.Pattern | None) -> list[tuple[int, str]]:
    """(line number, problem) for the content rules."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if HOME_PATH.search(line):
            out.append((i, "home path (write ~ or a relative path)"))
        if hosts is not None and hosts.search(line):
            out.append((i, "host name"))
        for ip in private_ipv4(line):
            out.append((i, f"private IPv4 address {ip}"))
        if KEYED_ID.search(line):
            out.append((i, "machine/user/session/device id"))
        if CREDENTIAL.search(line):
            out.append((i, "credential (URL user info, token or API key)"))
    old, new = text.splitlines(), redact(text).splitlines()
    out += [(i, "third-party app name (run redact.py without --check)")
            for i, (a, b) in enumerate(zip(old, new), 1) if a != b]
    return sorted(out)


def _capability_tail(parts: tuple[str, ...]) -> tuple[str, ...] | None:
    """The part of a path below a run's capability/ directory: docs/benchmark/.../backend/<run-id>/
    capability/<tail> (the results contract, whatever levels sit between benchmark/ and backend/), or
    results/<model>/<run-id>/capability/<tail>."""
    for i, p in enumerate(parts):
        if p == "docs" and parts[i + 1:i + 2] == ("benchmark",):
            for j in range(i + 2, len(parts) - 2):
                if parts[j] == "backend" and parts[j + 2] == "capability":
                    return parts[j + 3:]
        if p == "results" and len(parts) > i + 3 and parts[i + 3] == "capability":
            return parts[i + 4:]
    return None


def capability_problem(path: str) -> str | None:
    """Why a path under a run's capability/ directory is not on the export allowlist."""
    tail = _capability_tail(PurePosixPath(path.replace(os.sep, "/")).parts)
    if tail is None or tail == ("results.json",) or (
            len(tail) == 3 and tail[0] in CAPABILITY_HARNESSES and _RUN_NAME.match(tail[1])
            and tail[1] not in (".", "..") and tail[2] in CAPABILITY_FILES):
        return None
    return ("not on the capability export allowlist (results.json; <harness>/<run>/"
            + ", ".join(CAPABILITY_FILES) + "; harness " + ", ".join(CAPABILITY_HARNESSES) + ")")


def check_file(name: str, hosts: re.Pattern | None) -> list[str]:
    """Problems of one file as "name:line: problem" or "name: problem" lines."""
    f = Path(name)
    out = []
    cap = capability_problem(name)
    if cap:
        out.append(f"{name}: {cap}")
    if f.is_symlink():
        return out + [f"{name}: symlink (commit the file itself, or nothing)"]
    if not f.exists():
        return out + [f"{name}: missing"]
    if not f.is_file():
        return out + [f"{name}: not a regular file"]
    data = f.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return out + [f"{name}: not UTF-8 (byte {e.start})"]
    return out + [f"{name}:{line}: {what}" for line, what in scan_text(text, hosts)]


def rewrite(name: str) -> int:
    """Replace third-party app names in place; 1 if the file could not be processed."""
    f = Path(name)
    if f.is_symlink() or not f.is_file():
        print(f"redact: {name}: skipped (symlink, missing or not a regular file)")
        return 1
    try:
        old = f.read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"redact: {name}: skipped (not UTF-8, byte {e.start})")
        return 1
    new = redact(old)
    if new != old:
        print(f"redact: {name}: {sum(1 for a, b in zip(old.splitlines(), new.splitlines()) if a != b)} line(s)")
        tmp = f.with_name(f.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as g:
            g.write(new)
        tmp.replace(f)
    return 0


def main(argv: list[str]) -> int:
    check = "--check" in argv
    files = [a for a in argv if a != "--check"]
    if any(a.startswith("-") for a in files):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    if not check:
        return max((rewrite(n) for n in files), default=0)
    hosts = host_pattern(hostnames())
    problems = [p for n in files for p in check_file(n, hosts)]
    for p in problems:
        print(f"redact: {p}")
    sys.stdout.flush()
    if problems:
        print(f"redact: {len(problems)} problem(s); fix them or keep these files out of the commit", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
