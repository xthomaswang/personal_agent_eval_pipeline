"""Canned assistant tool calls and tool results for M1 and the 20-step M4 trajectory.

The canned assistant turns are appended to the history verbatim (the model's own output
is discarded), so every framework sees byte-identical prompts. Each M4 step is one
assistant tool call (about 150 tokens including a short preamble), one tool result
(text) and one new screenshot.
"""

from __future__ import annotations

import json
from typing import Any


def call(cid: str, name: str, args: dict[str, Any], content: str = "") -> dict[str, Any]:
    """An OpenAI-format assistant message with one tool call (arguments as a JSON string)."""
    return {"role": "assistant", "content": content, "tool_calls": [{
        "id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


# ------------------------------------------------------------------------- M1
M1_ASSISTANT = call("call_m1_click", "computer_click", {"x": 287, "y": 361, "button": "left", "clicks": 1},
                    "I'll tick the \"Format on save\" checkbox in the Editor section.")
M1_CAPTION = "Screenshot after the click."

# ------------------------------------------------------------------------- M4
# (content, tool name, arguments, tool-result head, screenshot kind, window title)
M4_STEPS: list[tuple[str, str, dict[str, Any], str, str, str]] = [
    ("I'll start by opening the project in the Code Editor so that I can follow the changes on screen while I work. "
     "After that I'll run the tests, fix the failure, update the configuration and the changelog, then build and "
     "verify the archive, checking each step before moving on."
     " The release steps in docs/release.md match this plan, so I'll follow them in the same order and note anything unexpected.",
     "app_launch", {"app_name": "Code Editor", "args": ["/workspace/demo-app"], "wait_for_window": True},
     "Launched Code Editor (pid 48190).\nNew window w-7c21 \"demo-app - Code Editor\" at (40, 60), size 1200x880, focused.\n"
     "Explorer root: /workspace/demo-app (12 folders, 41 files). Restored 2 editor tabs from the last session.\n"
     "Windows:\n", "editor", "demo-app - README.md"),
    ("The editor is open with the project tree on the left and README.md in the first tab. Before changing anything, "
     "I'll run the full service test suite to see which tests fail on main, so that I know the baseline and don't "
     "mistake an old failure for one I introduced."
     " The project context says the full suite takes about forty seconds, so a ten-minute timeout leaves plenty of room.",
     "run_tests", {"runner": "pytest", "path": "tests", "fail_fast": False, "timeout_s": 600},
     "pytest exited with status 1 after 38.4 s: 212 passed, 1 failed, 3 skipped.\n\nFAILED tests/test_parser.py::test_empty_input\n"
     "  tests/test_parser.py:41: in test_empty_input\n    rows = parse_rows(\"\\n\\nname,tags,created_at\\n\")\n"
     "  src/demo_app/parser.py:52: in parse_rows\n    if not rows[0]:\nE   IndexError: list index out of range\n\nOutput (tail):\n",
     "terminal", "Terminal - /workspace/demo-app"),
    ("The suite reports 212 passed and one failure: tests/test_parser.py::test_empty_input raises IndexError in "
     "parser.py at line 52, where the code reads rows[0]. That matches the known failure noted in the repository "
     "state. Let me read the test first to understand the behaviour it expects."
     " Reading the test before the implementation tells me what the intended behaviour is, not just what the code does today.",
     "fs_read_file", {"path": "tests/test_parser.py", "offset": 1, "limit": 80},
     "tests/test_parser.py (lines 1-80 of 96):\n", "editor", "demo-app - test_parser.py"),
    ("The test feeds the parser two blank lines followed by the header row and expects the header to be found and an "
     "empty result to be returned, so blank lines before the header must be ignored. Now I'll read the parser around "
     "the failing line to see how the rows are collected."
     " The fix should be small: valid files must be parsed exactly as before, and only leading blank lines are handled differently.",
     "fs_read_file", {"path": "src/demo_app/parser.py", "offset": 30, "limit": 60},
     "src/demo_app/parser.py (lines 30-89 of 140):\n", "editor", "demo-app - parser.py"),
    ("The parser builds the row list without filtering, so a leading blank line becomes rows[0] and the header check "
     "fails. I'll filter empty rows first and keep the original error for input without a header.",
     "fs_edit_file", {"path": "src/demo_app/parser.py", "edits": [{
         "old_text": "    rows = [row for row in reader]\n    if not rows[0]:\n        raise ImportFormatError(\"missing header row\")",
         "new_text": "    rows = [row for row in reader if any(cell.strip() for cell in row)]\n    if not rows:\n"
                     "        raise ImportFormatError(\"missing header row\")"}]},
     "Applied 1 edit to src/demo_app/parser.py (1 replacement, +2 -2 lines).\n\nDiff:\n", "editor", "demo-app - parser.py"),
    ("The edit applied cleanly and keeps the original error message for input that has no header row at all. The "
     "parser now filters empty rows before the header check. I'll rerun only the failing test to confirm the fix "
     "before running anything broader."
     " The whole suite still has to pass again before the release build, but a narrow run is the quickest confirmation.",
     "run_tests", {"runner": "pytest", "path": "tests/test_parser.py", "filter": "test_empty_input", "fail_fast": True},
     "pytest exited with status 0 after 1.9 s: 1 passed, 23 deselected.\n\nOutput (tail):\n", "terminal",
     "Terminal - /workspace/demo-app"),
    ("The parser test passes now, and the change is limited to the row filtering. Next comes the job API port. The "
     "project context says local overrides belong in config/local.ini, which is not committed, so I'll check whether "
     "that file already exists before writing it, to avoid overwriting existing settings."
     " The memory notes say the loader reads this file only at startup, which matters later when the release build is started from the unpacked archive. Only then will I write the setting.",
     "fs_read_file", {"path": "config/local.ini"},
     "Error: file not found: config/local.ini\nThe directory config/ exists and contains:\n", "files", "Files - config"),
    ("The file does not exist yet, so nothing can be lost by creating it. I'll write it with the job port set to 8080 "
     "and keep the storage path at its documented default."
     " The loader expects plain INI sections, so I'll keep the format simple.",
     "fs_write_file", {"path": "config/local.ini",
                       "content": "[jobs]\nport = 8080\nretries = 3\n\n[storage]\npath = var/demo.sqlite3\n",
                       "overwrite": True, "create_dirs": True},
     "Wrote 66 bytes to config/local.ini (created). The file is ignored by git (.gitignore line 14).\nLoader check:\n",
     "editor", "demo-app - local.ini"),
    ("The job port is configured, and the loader check reads the new value. Now I'll add the changelog entry under "
     "the Unreleased heading, describing both the importer fix and the new override."
     " Each entry follows the conventional prefix style used in the commit history.",
     "fs_edit_file", {"path": "CHANGELOG.md", "edits": [{
         "old_text": "## Unreleased\n",
         "new_text": "## Unreleased\n\n- fix: skip empty CSV rows before checking the header row in the importer\n"
                     "- docs: describe the job API port override in config/local.ini\n"}]},
     "Applied 1 edit to CHANGELOG.md (1 replacement, +3 -0 lines).\n\nDiff:\n", "editor", "demo-app - CHANGELOG.md"),
    ("The changelog entry is in place. The release script needs a clean working tree and refuses to run otherwise, "
     "so I'll check the repository state before committing. I expect two modified files, the parser and the "
     "changelog; config/local.ini should not appear because git ignores it."
     " Untracked files are fine, but unexpected modifications would have to be discussed with the user first. If anything else shows up, I'll stop and ask.",
     "git_status", {"repo_path": "/workspace/demo-app", "include_untracked": True},
     "On branch main\nYour branch is up to date with 'origin/main'.\n\nChanges not staged for commit:\n"
     "  modified:   CHANGELOG.md\n  modified:   src/demo_app/parser.py\n\nUntracked files:\n  notes/scratch.md\n\nDetails:\n",
     "terminal", "Terminal - /workspace/demo-app"),
    ("Only the two expected files are modified, plus the untracked scratch note, which I'll leave alone. I'll commit "
     "the parser fix and the changelog entry together."
     " The message explains the cause of the failure, not just the change.",
     "git_commit", {"message": "fix: skip empty rows in the CSV importer\n\nThe parser indexed the first row before removing "
                               "empty lines, so input that started with blank lines raised IndexError. Filter empty rows "
                               "first and add a changelog entry for the 1.4.0 release.",
                    "files": ["src/demo_app/parser.py", "CHANGELOG.md"]},
     "[main 3b7e0d1] fix: skip empty rows in the CSV importer\n 2 files changed, 5 insertions(+), 2 deletions(-)\n\nHooks:\n",
     "terminal", "Terminal - /workspace/demo-app"),
    ("The commit went through and the tree is clean apart from the untracked note. Now I'll build the release "
     "archive with the project's release script."
     " According to docs/release.md it builds the front end, packs the service and writes the archive to dist/.",
     "shell_run", {"command": "scripts/release.sh 1.4.0", "cwd": "/workspace/demo-app", "timeout_s": 900,
                   "description": "Build the 1.4.0 release archive; the script also prints the archive's SHA-256."},
     "exit code 0 (52.7 s)\nstdout (tail):\n", "terminal", "Terminal - /workspace/demo-app"),
    ("The script finished and printed the archive's SHA-256. To verify the archive itself rather than the working "
     "tree, I'll unpack it into a temporary directory."
     " Testing the unpacked copy catches files that are missing from the archive but present in the repository.",
     "shell_run", {"command": "mkdir -p /tmp/demo-release && tar -xzf dist/demo-app-1.4.0.tar.gz -C /tmp/demo-release "
                              "&& ls /tmp/demo-release/demo-app-1.4.0",
                   "description": "Unpack the release archive into a temporary directory to test it in isolation."},
     "exit code 0 (1.3 s)\nstdout:\n", "terminal", "Terminal - /tmp/demo-release"),
    ("The archive unpacked into a single top-level folder with the service and the static files. I'll start the "
     "unpacked service in the background and keep its log under .agent/logs.",
     "shell_run", {"command": "cd /tmp/demo-release/demo-app-1.4.0 && (nohup uv run demo-app serve --port 8000 > "
                              "/workspace/demo-app/.agent/logs/release-serve.log 2>&1 &)",
                   "timeout_s": 30, "description": "Start the release build of the service on port 8000 in the background."},
     "exit code 0 (0.4 s)\nBackground process started: pid 48213.\nLog (first lines):\n", "terminal",
     "Terminal - /tmp/demo-release"),
    ("The service process started and its log shows the server listening on port 8000, so it should be ready in a "
     "moment. I'll open the health endpoint in a new browser tab, so that the check is visible on screen as the "
     "release process asks."
     " The version reported there must match 1.4.0. If the page is blank, I'll wait a moment and reload it once.",
     "browser_navigate", {"url": "http://localhost:8000/health", "new_tab": True, "wait_until": "load"},
     "Loaded http://localhost:8000/health in tab 4 (HTTP 200, 38 ms).\nTitle: (none, JSON document)\nVisible text:\n",
     "browser", "Browser - localhost:8000/health"),
    ("The page loaded with HTTP 200 and shows a small JSON document. The response text is small on screen, so I'll "
     "zoom into the page body to read the version string exactly instead of guessing it."
     " Misreading a single digit would make the whole verification meaningless, so this extra step is worth it.",
     "image_zoom", {"region": {"x": 80, "y": 180, "width": 520, "height": 120}, "scale": 2},
     "Zoomed region (80, 180, 600, 300) at 2x. Recognized text:\n", "browser", "Browser - localhost:8000/health"),
    ("The zoomed view reads status ok and version 1.4.0. To double-check without relying on the rendered page or on "
     "text recognition, I'll query the endpoint directly and look at the raw response."
     " An explicit Accept header makes sure the service answers with JSON, and a short timeout is enough for a local check.",
     "http_request", {"method": "GET", "url": "http://localhost:8000/health", "headers": {"Accept": "application/json"},
                      "timeout_s": 5},
     "HTTP/1.1 200 OK (12 ms)\ncontent-type: application/json\ncontent-length: 37\n\n{\"status\": \"ok\", \"version\": \"1.4.0\"}\n\nServer log (tail):\n",
     "browser", "Browser - localhost:8000/health"),
    ("The endpoint returns HTTP 200 with status ok and version 1.4.0, so the release build starts and reports the "
     "right version. That completes the verification. I'll stop the temporary service now so that no stray process "
     "is left running at the end of the task."
     " TERM lets the server shut down cleanly; I'll only use KILL if it does not exit within a few seconds.",
     "process_kill", {"pid": 48213, "signal": "TERM"},
     "Sent TERM to pid 48213. The process exited with status 0 after 0.6 s.\nRemaining processes started in this session:\n",
     "terminal", "Terminal - /tmp/demo-release"),
    ("The temporary service exited cleanly and no other processes from this session are running. The last step is "
     "the release note with the version, the date, the most important changes and any known issues. I'll open the "
     "Notes app to write it."
     " The note gets the release tag, as docs/release.md describes, so the maintainers can find it. Pushing tags and uploading the archive are left to a maintainer.",
     "app_launch", {"app_name": "Notes", "wait_for_window": True},
     "Launched Notes (pid 48230).\nNew window w-8d02 \"Notes\" at (120, 80), size 1040x820, focused.\nRecent notes:\n",
     "notes", "Notes"),
    ("",
     "notes_create", {"title": "demo-app 1.4.0 release",
                      "body": "## demo-app 1.4.0 (2026-10-17)\n\n- Fixed: the CSV importer now skips empty rows before "
                              "checking the header.\n- Docs: the job API port can be overridden in config/local.ini.\n"
                              "- Verified: the release archive starts and /health reports 1.4.0.\n\nKnown issues: none.",
                      "tags": ["release", "demo-app"], "pinned": False},
     "Created note n-1f3e \"demo-app 1.4.0 release\" with tags release, demo-app.\nNotes list now:\n", "notes", "Notes"),
]

M4_CAPTION = "Screenshot after step {k}."
