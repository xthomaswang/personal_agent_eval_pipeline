"""M7 tool-call correctness cases.

Each case runs on top of the agent prefix (system prompt + all tools). `expect` is
checked by `agentbench.m7check`; `reference` is one correct answer (used by tests and by
the smart mock server). Image cases reference generated screenshots by id and are N/A
in --text-only mode.
"""

from __future__ import annotations

from typing import Any

from .agent_steps import call

NOTICE = ("Scheduled maintenance: on Saturday, 2026-11-07, between 06:00 and 08:00 UTC, the demo-app staging "
          "environment will be unavailable while the database is migrated to the new storage layout. Jobs that are "
          "queued during the window will start after the migration completes, in their original order. The public "
          "API keeps its current version, and no client changes are required. If a job fails after the migration, "
          "rerun it once; if it fails again, attach the job id and the last fifty lines of the job log to the "
          "release issue so that the maintainers can investigate. The next maintenance window is planned for "
          "early December and will be announced at least one week in advance.")

SETTINGS_PY = ('"""Runtime settings for the development server."""\n\nDEBUG = True\nPORT = 8000\nHOST = "127.0.0.1"\n'
               'LOG_LEVEL = "info"\nMAX_UPLOAD_MB = 16\n')

HELLO_PY = "print(\"Hello, \\\"world\\\"\")\nprint('done')"
COMMIT_MSG = 'fix: handle "null" values in the parser\n\nCloses #42'
TYPED = "Café déjà vu — naïve résumé ✓"


def _u(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _tool(cid: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": cid, "content": text}


PATH = "path"

CASES: list[dict[str, Any]] = [
    {"id": "m7-01-single", "category": "single call",
     "messages": [_u("Show me the contents of src/demo_app/config/defaults.yaml.")],
     "expect": {"calls": [{"name": "fs_read_file", "args": {"path": {"eq": "src/demo_app/config/defaults.yaml", "norm": PATH}}}]},
     "reference": [{"name": "fs_read_file", "arguments": {"path": "src/demo_app/config/defaults.yaml"}}]},

    {"id": "m7-02-multi-args", "category": "multiple args",
     "messages": [_u("Search the src directory for the regular expression TODO\\(.*\\) in Python files only, ignoring case, "
                     "and show 2 lines of context around each match.")],
     "expect": {"calls": [{"name": "fs_grep", "args": {
         "pattern": {"contains": "TODO"}, "path": {"eq": "src", "norm": PATH},
         "glob": {"in": ["*.py", "**/*.py", "src/**/*.py"]}, "case_insensitive": {"eq": True},
         "context_lines": {"eq": 2}}}]},
     "reference": [{"name": "fs_grep", "arguments": {"pattern": "TODO\\(.*\\)", "path": "src", "glob": "*.py",
                                                    "case_insensitive": True, "context_lines": 2}}]},

    {"id": "m7-03-nested-object", "category": "nested objects",
     "messages": [_u("Send a POST request to http://localhost:8080/api/jobs with the header X-Request-Source set to bench "
                     "and this JSON body: {\"name\": \"nightly-build\", \"retries\": 3, \"notify\": {\"on_failure\": true, "
                     "\"channel\": \"builds\"}}")],
     "expect": {"calls": [{"name": "http_request", "args": {
         "method": {"eq": "POST"}, "url": {"eq": "http://localhost:8080/api/jobs"},
         "headers": {"subset": {"X-Request-Source": "bench"}},
         "json_body": {"eq": {"name": "nightly-build", "retries": 3, "notify": {"on_failure": True, "channel": "builds"}}}}}]},
     "reference": [{"name": "http_request", "arguments": {
         "method": "POST", "url": "http://localhost:8080/api/jobs", "headers": {"X-Request-Source": "bench"},
         "json_body": {"name": "nightly-build", "retries": 3, "notify": {"on_failure": True, "channel": "builds"}}}}]},

    {"id": "m7-04-array-of-objects", "category": "nested objects (array of objects, multi-turn)",
     "messages": [_u("Open src/demo_app/settings.py."),
                  call("call_m7_04", "fs_read_file", {"path": "src/demo_app/settings.py"}),
                  _tool("call_m7_04", "src/demo_app/settings.py (7 lines):\n" + SETTINGS_PY),
                  _u("In that file, replace DEBUG = True with DEBUG = False and PORT = 8000 with PORT = 8080. Use a "
                     "single edit call with two edits.")],
     "expect": {"calls": [{"name": "fs_edit_file", "args": {
         "path": {"eq": "src/demo_app/settings.py", "norm": PATH},
         "edits": {"match_each": [{"old_text": {"contains": "DEBUG = True"}, "new_text": {"contains": "DEBUG = False"}},
                                  {"old_text": {"contains": "PORT = 8000"}, "new_text": {"contains": "PORT = 8080"}}]}}}]},
     "reference": [{"name": "fs_edit_file", "arguments": {"path": "src/demo_app/settings.py", "edits": [
         {"old_text": "DEBUG = True", "new_text": "DEBUG = False"}, {"old_text": "PORT = 8000", "new_text": "PORT = 8080"}]}}]},

    {"id": "m7-05-array-strings", "category": "arrays",
     "messages": [_u("Delete the build leftovers build/a.tmp, build/b.tmp and build/c.tmp now; no need to ask.")],
     "expect": {"calls": [{"name": "fs_delete", "args": {
         "paths": {"set_eq": ["build/a.tmp", "build/b.tmp", "build/c.tmp"], "norm": PATH}}}]},
     "reference": [{"name": "fs_delete", "arguments": {"paths": ["build/a.tmp", "build/b.tmp", "build/c.tmp"]}}]},

    {"id": "m7-06-array-integers", "category": "arrays",
     "messages": [_u("Close browser tabs 2, 5 and 7.")],
     "expect": {"calls": [{"name": "browser_close_tabs", "args": {"tab_ids": {"set_eq": [2, 5, 7]}}}]},
     "reference": [{"name": "browser_close_tabs", "arguments": {"tab_ids": [2, 5, 7]}}]},

    {"id": "m7-07-enum", "category": "enums",
     "messages": [_u("Snap the focused window to the left half of the screen.")],
     "expect": {"calls": [{"name": "window_arrange", "args": {"layout": {"eq": "tile_left"}}}]},
     "reference": [{"name": "window_arrange", "arguments": {"layout": "tile_left"}}]},

    {"id": "m7-08-key-combo", "category": "enums / arrays",
     "messages": [_u("Press Command+Shift+T to reopen the last closed browser tab.")],
     "expect": {"calls": [{"name": "computer_key_press", "args": {"keys": {"set_eq": ["cmd", "shift", "t"], "norm": "key"}}}]},
     "reference": [{"name": "computer_key_press", "arguments": {"keys": ["cmd", "shift", "t"]}}]},

    {"id": "m7-09-numbers", "category": "numbers",
     "messages": [_u("Scroll down 5 notches at the point (640, 512).")],
     "expect": {"calls": [{"name": "computer_scroll", "args": {
         "x": {"eq": 640}, "y": {"eq": 512}, "direction": {"eq": "down"}, "amount": {"eq": 5}}}]},
     "reference": [{"name": "computer_scroll", "arguments": {"x": 640, "y": 512, "direction": "down", "amount": 5}}]},

    {"id": "m7-10-float-bool", "category": "numbers and booleans",
     "messages": [_u("Set the system volume to 35% and make sure the output is not muted.")],
     "expect": {"calls": [{"name": "system_set_volume", "args": {"level": {"range": [0.345, 0.355]},
                                                                 "muted": {"eq": False}}}]},
     "reference": [{"name": "system_set_volume", "arguments": {"level": 0.35, "muted": False}}]},

    {"id": "m7-11-booleans", "category": "numbers and booleans",
     "messages": [_u("Write the line retry_limit = 5 to config/local.ini. Replace the file if it already exists, and "
                     "create the config directory if it is missing.")],
     "expect": {"calls": [{"name": "fs_write_file", "args": {
         "path": {"eq": "config/local.ini", "norm": PATH}, "content": {"contains": "retry_limit = 5"},
         "overwrite": {"eq": True}, "create_dirs": {"eq": True}}}]},
     "reference": [{"name": "fs_write_file", "arguments": {"path": "config/local.ini", "content": "retry_limit = 5\n",
                                                          "overwrite": True, "create_dirs": True}}]},

    {"id": "m7-12-chinese-args", "category": "Unicode / Chinese args",
     "messages": [_u("Create a note titled 会议纪要 with the body 下周一上午十点讨论发布计划。 and the tag 工作.")],
     "expect": {"calls": [{"name": "notes_create", "args": {
         "title": {"eq": "会议纪要", "norm": "strip"}, "body": {"eq": "下周一上午十点讨论发布计划。", "norm": "strip"},
         "tags": {"set_eq": ["工作"]}}}]},
     "reference": [{"name": "notes_create", "arguments": {"title": "会议纪要", "body": "下周一上午十点讨论发布计划。",
                                                         "tags": ["工作"]}}]},

    {"id": "m7-13-chinese-instruction", "category": "Unicode / Chinese args",
     "messages": [_u("请在终端里运行命令 ls -la /workspace/demo-app/logs ，只运行这一条命令。")],
     "expect": {"calls": [{"name": "shell_run", "args": {"command": {"eq": "ls -la /workspace/demo-app/logs", "norm": "ws"}}}]},
     "reference": [{"name": "shell_run", "arguments": {"command": "ls -la /workspace/demo-app/logs",
                                                      "description": "List the log directory."}}]},

    {"id": "m7-14-unicode-text", "category": "Unicode / Chinese args",
     "messages": [_u("Type the following text into the focused field, exactly as written:\n\n```\n" + TYPED + "\n```")],
     "expect": {"calls": [{"name": "computer_type_text", "args": {"text": {"eq": TYPED, "norm": "strip"}}}]},
     "reference": [{"name": "computer_type_text", "arguments": {"text": TYPED}}]},

    {"id": "m7-15-quotes-newlines", "category": "quotes and newlines in strings",
     "messages": [_u("Create the file scripts/hello.py with exactly this content:\n\n```python\n" + HELLO_PY + "\n```")],
     "expect": {"calls": [{"name": "fs_write_file", "args": {
         "path": {"eq": "scripts/hello.py", "norm": PATH}, "content": {"eq": HELLO_PY, "norm": "strip"}}}]},
     "reference": [{"name": "fs_write_file", "arguments": {"path": "scripts/hello.py", "content": HELLO_PY + "\n"}}]},

    {"id": "m7-16-commit-message", "category": "quotes and newlines in strings (multi-turn)",
     "messages": [_u("What is staged right now?"),
                  call("call_m7_16", "git_status", {}),
                  _tool("call_m7_16", "On branch main\nChanges to be committed:\n  modified:   src/demo_app/parser.py\n"
                                      "  modified:   tests/test_parser.py\n"),
                  _u("Commit exactly what is staged, with this exact message (keep the blank line):\n\n```\n" + COMMIT_MSG + "\n```")],
     "expect": {"calls": [{"name": "git_commit", "args": {"message": {"eq": COMMIT_MSG, "norm": "strip"}}}]},
     "reference": [{"name": "git_commit", "arguments": {"message": COMMIT_MSG}}]},

    {"id": "m7-17-long-string", "category": "long string args",
     "messages": [_u("Write the following text to docs/notice.md exactly as given, as a single paragraph:\n\n```\n" + NOTICE + "\n```")],
     "expect": {"calls": [{"name": "fs_write_file", "args": {
         "path": {"eq": "docs/notice.md", "norm": PATH}, "content": {"eq": NOTICE, "norm": "strip"}}}]},
     "reference": [{"name": "fs_write_file", "arguments": {"path": "docs/notice.md", "content": NOTICE + "\n"}}]},

    {"id": "m7-18-parallel", "category": "two parallel calls",
     "messages": [_u("Read README.md and CHANGELOG.md. The two reads are independent, so issue both calls at once.")],
     "expect": {"calls": [{"name": "fs_read_file", "args": {"path": {"eq": "README.md", "norm": PATH}}},
                          {"name": "fs_read_file", "args": {"path": {"eq": "CHANGELOG.md", "norm": PATH}}}]},
     "reference": [{"name": "fs_read_file", "arguments": {"path": "README.md"}},
                   {"name": "fs_read_file", "arguments": {"path": "CHANGELOG.md"}}]},

    {"id": "m7-19-similar-tools-glob", "category": "choosing between similar tools",
     "messages": [_u("Find all files whose names end in .yaml anywhere under config/.")],
     "expect": {"calls": [{"name": "fs_search_files", "args": {"pattern": {"contains": ".yaml"}}}]},
     "reference": [{"name": "fs_search_files", "arguments": {"pattern": "**/*.yaml", "root": "config"}}]},

    {"id": "m7-20-similar-tools-http", "category": "choosing between similar tools",
     "messages": [_u("Get the raw JSON from http://localhost:8000/health. I only need the response body, not a browser page.")],
     "expect": {"calls": [{"name": "http_request", "args": {"method": {"eq": "GET"},
                                                            "url": {"eq": "http://localhost:8000/health"}}}]},
     "reference": [{"name": "http_request", "arguments": {"method": "GET", "url": "http://localhost:8000/health"}}]},

    {"id": "m7-21-no-call", "category": "no call needed",
     "messages": [_u("In one sentence, what does the HTTP status code 404 mean?")],
     "expect": {"no_call": True},
     "reference": []},

    {"id": "m7-22-image-read", "category": "call after an image", "image": "m7-files-highlight",
     "messages": [{"role": "user", "content": [
         {"type": "image_ref", "image": "m7-files-highlight"},
         {"type": "text", "text": "This is the Files window. Read the contents of the file that is highlighted in blue."}]}],
     "expect": {"calls": [{"name": "fs_read_file", "args": {"path": {"endswith": "{highlight}"}}}]},
     "reference": [{"name": "fs_read_file", "arguments": {"path": "data/{highlight}"}}]},

    {"id": "m7-23-image-click", "category": "call after an image", "image": "m7-save-dialog",
     "messages": [{"role": "user", "content": [
         {"type": "image_ref", "image": "m7-save-dialog"},
         {"type": "text", "text": "Click the Save button in this dialog."}]}],
     "expect": {"alternatives": [
         [{"name": "computer_click", "point_in_box": {"x": "x", "y": "y", "box": "{save_box}"}}],
         [{"name": "ui_find_element", "args": {"query": {"icontains": "save"}}}]]},
     "reference": [{"name": "computer_click", "arguments": {"x": "{save_cx}", "y": "{save_cy}"}}]},

    {"id": "m7-24-calendar", "category": "nested objects",
     "messages": [_u("Schedule \"Release planning\" on 2026-11-03 from 10:00 to 11:00 in room B2. It repeats weekly for "
                     "4 occurrences, with reminders 10 and 30 minutes before.")],
     "expect": {"calls": [{"name": "calendar_create_event", "args": {
         "title": {"eq": "Release planning", "norm": "strip"}, "start": {"startswith": "2026-11-03T10:00"},
         "end": {"startswith": "2026-11-03T11:00"}, "location": {"icontains": "B2"},
         "reminders_min": {"set_eq": [10, 30]},
         "recurrence": {"subset": {"frequency": "weekly", "count": 4}}}}]},
     "reference": [{"name": "calendar_create_event", "arguments": {
         "title": "Release planning", "start": "2026-11-03T10:00", "end": "2026-11-03T11:00", "location": "Room B2",
         "reminders_min": [10, 30], "recurrence": {"frequency": "weekly", "interval": 1, "count": 4}}}]},
]

QUICK_CASES = ["m7-01-single", "m7-03-nested-object", "m7-12-chinese-args", "m7-18-parallel", "m7-21-no-call",
               "m7-22-image-read"]
