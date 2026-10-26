"""Generic agent tool schemas (OpenAI `tools` format; the Anthropic format is derived).

Computer use, windows, filesystem, shell, git, tests, browser, HTTP, search, notes,
calendar, tasks, settings. Everything is generic and synthetic: no banking, no messaging,
no personal data. Paths live under the fictional project root /workspace/demo-app.
"""

from __future__ import annotations

from typing import Any


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


def _s(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "string", "description": desc, **kw}


def _i(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "integer", "description": desc, **kw}


def _n(desc: str, **kw: Any) -> dict[str, Any]:
    return {"type": "number", "description": desc, **kw}


def _b(desc: str) -> dict[str, Any]:
    return {"type": "boolean", "description": desc}


def _point(desc: str) -> dict[str, Any]:
    return {"type": "object", "description": desc, "properties": {
        "x": _i("Horizontal pixel coordinate, 0 at the left edge."),
        "y": _i("Vertical pixel coordinate, 0 at the top edge.")}, "required": ["x", "y"], "additionalProperties": False}


def _rect(desc: str) -> dict[str, Any]:
    return {"type": "object", "description": desc, "properties": {
        "x": _i("Left edge in pixels."), "y": _i("Top edge in pixels."),
        "width": _i("Width in pixels.", minimum=1), "height": _i("Height in pixels.", minimum=1)},
        "required": ["x", "y", "width", "height"], "additionalProperties": False}


TOOLS: list[dict[str, Any]] = [
    # ------------------------------------------------------------ computer use
    _fn("computer_screenshot",
        "Capture the current screen and return it as an image, together with the list of visible windows. "
        "Coordinates in every computer_* tool refer to pixels of the most recent screenshot, which is 1280x1024 "
        "unless a region is given. Take a new screenshot after any action that changes the screen before you rely "
        "on positions again; layouts move when dialogs open, menus expand or windows are resized. Do not take "
        "screenshots in a tight loop: wait for the UI to settle with computer_wait when an operation is in progress.",
        {"region": _rect("Optional sub-rectangle to capture. Omit to capture the whole display."),
         "display": _i("Display index for multi-monitor setups. The primary display is 0.", minimum=0)},
        []),
    _fn("computer_click",
        "Move the pointer to a position and click. Use the coordinates of the element's center as seen in the "
        "latest screenshot. Prefer ui_click_element when you already have an element id from ui_find_element, "
        "because element ids survive small layout changes while raw coordinates do not. Double-clicks and "
        "triple-clicks are expressed with the clicks parameter. Hold modifier keys with modifiers, for example "
        "cmd-click to select several files in a file manager.",
        {"x": _i("Horizontal pixel coordinate of the click target."),
         "y": _i("Vertical pixel coordinate of the click target."),
         "button": _s("Mouse button to press.", enum=["left", "right", "middle"]),
         "clicks": _i("Number of consecutive clicks: 1 (single), 2 (double) or 3 (triple).", minimum=1, maximum=3),
         "modifiers": {"type": "array", "description": "Modifier keys held during the click.",
                       "items": {"type": "string", "enum": ["shift", "ctrl", "alt", "cmd"]}}},
        ["x", "y"]),
    _fn("computer_drag",
        "Press the mouse button at a start point, move to an end point and release. Use it to move windows by "
        "their title bar, resize panes with splitters, select text ranges, or drag files between folders. Long "
        "drags across the screen should use a duration of at least 300 ms so that the target application "
        "registers intermediate pointer events.",
        {"start": _point("Where the button is pressed."), "end": _point("Where the button is released."),
         "button": _s("Mouse button held during the drag.", enum=["left", "right", "middle"]),
         "duration_ms": _i("How long the movement takes, in milliseconds.", minimum=0, maximum=10000)},
        ["start", "end"]),
    _fn("computer_type_text",
        "Type text into the element that currently has keyboard focus. Click or focus the target field first and "
        "verify the focus in a screenshot when it matters. The text is typed literally, including Unicode "
        "characters; it is not interpreted as key names, so use computer_key_press for shortcuts such as "
        "cmd+s. Set press_enter to submit a single-line field after typing.",
        {"text": _s("The exact text to type."),
         "delay_ms": _i("Delay between keystrokes in milliseconds, for applications that drop fast input.",
                        minimum=0, maximum=500),
         "press_enter": _b("Press Enter after the text has been typed.")},
        ["text"]),
    _fn("computer_key_press",
        "Press a key or a key combination. All keys in the list are pressed together in order and released in "
        "reverse order, so [\"cmd\", \"shift\", \"t\"] performs Command+Shift+T. Key names are lowercase: cmd, "
        "ctrl, alt, shift, enter, tab, esc, backspace, delete, space, up, down, left, right, home, end, pageup, "
        "pagedown, f1 to f12, or a single character. Use repeat to press the same combination several times.",
        {"keys": {"type": "array", "description": "Keys pressed together, modifiers first.",
                  "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
         "repeat": _i("How many times to press the combination.", minimum=1, maximum=50)},
        ["keys"]),
    _fn("computer_scroll",
        "Scroll the content under a screen position. The amount is measured in wheel notches; one notch moves "
        "about three lines of text in most applications. Scroll inside the pane you want to move, not over a "
        "toolbar, because many applications route wheel events to the element under the pointer.",
        {"x": _i("Horizontal pixel coordinate of the scroll position."),
         "y": _i("Vertical pixel coordinate of the scroll position."),
         "direction": _s("Scroll direction.", enum=["up", "down", "left", "right"]),
         "amount": _i("Number of wheel notches.", minimum=1, maximum=50)},
        ["x", "y", "direction", "amount"]),
    _fn("computer_wait",
        "Pause before the next action. Use it after starting a build, opening a large file or loading a page, "
        "instead of taking repeated screenshots. With until=idle the wait ends early when the screen stops "
        "changing; with until=window_change it ends when a window opens or closes.",
        {"seconds": _n("Maximum time to wait, in seconds.", minimum=0.1, maximum=120),
         "until": _s("Early-exit condition.", enum=["none", "idle", "window_change"])},
        ["seconds"]),
    # ----------------------------------------------------------------- windows
    _fn("window_list",
        "List open windows with their ids, titles, owning application, position, size and whether they are "
        "focused or minimized. Window ids are stable for the lifetime of a window and are used by window_focus, "
        "window_resize, window_arrange and ui_find_element.",
        {"app_name": _s("Only list windows of this application, for example \"Terminal\"."),
         "include_minimized": _b("Include minimized windows in the result.")},
        []),
    _fn("window_focus",
        "Bring a window to the front and give it keyboard focus. If the window is minimized it is restored first. "
        "Focusing a window does not move it to another display.",
        {"window_id": _s("Id of the window, as returned by window_list.")},
        ["window_id"]),
    _fn("window_resize",
        "Resize a window and optionally move it. Sizes are in pixels and are clamped to the visible area of the "
        "display. Some dialogs have a fixed size and ignore this call; the result reports the size that was "
        "actually applied.",
        {"window_id": _s("Id of the window, as returned by window_list."),
         "width": _i("New width in pixels.", minimum=200, maximum=1280),
         "height": _i("New height in pixels.", minimum=150, maximum=1024),
         "position": _point("Optional new position of the window's top-left corner.")},
        ["window_id", "width", "height"]),
    _fn("window_arrange",
        "Snap a window to a predefined layout: the left or right half, the top or bottom half, maximized, or "
        "centered at its current size. restore returns the window to the size it had before the last arrange "
        "call. Without window_id the focused window is arranged.",
        {"layout": _s("Target layout.", enum=["tile_left", "tile_right", "tile_top", "tile_bottom", "maximize",
                                               "center", "restore"]),
         "window_id": _s("Id of the window to arrange. Defaults to the focused window.")},
        ["layout"]),
    _fn("app_launch",
        "Start an application, or focus it if it is already running. Available applications include Code Editor, "
        "Terminal, Browser, Files, Settings, Notes, Calendar and Image Viewer. Arguments are passed to the "
        "application the same way as on a command line, for example a folder to open in Code Editor.",
        {"app_name": _s("Name of the application to start."),
         "args": {"type": "array", "description": "Command-line arguments for the application.",
                  "items": {"type": "string"}},
         "wait_for_window": _b("Return only after the first window of the application is visible.")},
        ["app_name"]),
    _fn("clipboard_write",
        "Replace the clipboard contents with plain text. Prefer typing short strings directly; use the clipboard "
        "for long or multi-line text that would be slow to type, then paste it with cmd+v.",
        {"text": _s("Text to place on the clipboard.")},
        ["text"]),
    _fn("ui_find_element",
        "Search the accessibility tree of the visible windows for elements whose label, value or help text "
        "matches a query. Returns element ids, roles, labels and bounding boxes, best matches first. This is "
        "more reliable than reading positions from a screenshot for small controls such as checkboxes and tabs.",
        {"query": _s("Text to look for in labels, values and help text. Matching is case-insensitive."),
         "role": _s("Restrict the search to one element role.",
                    enum=["button", "link", "textbox", "checkbox", "menuitem", "tab", "list_item", "any"]),
         "window_id": _s("Only search inside this window."),
         "max_results": _i("Maximum number of elements to return.", minimum=1, maximum=50)},
        ["query"]),
    _fn("ui_click_element",
        "Activate an element found with ui_find_element. press is a normal click, double_press opens items in "
        "lists, right_press opens the context menu, and focus moves keyboard focus without clicking.",
        {"element_id": _s("Element id returned by ui_find_element."),
         "action": _s("How to activate the element.", enum=["press", "double_press", "right_press", "focus"])},
        ["element_id"]),
    _fn("image_zoom",
        "Return an enlarged crop of the latest screenshot. Use it to read small text, inspect icons or check "
        "whether a checkbox is ticked before acting on it. The crop does not change the coordinate system used "
        "by other tools.",
        {"region": _rect("Area of the latest screenshot to enlarge."),
         "scale": _n("Magnification factor.", minimum=1, maximum=8)},
        ["region"]),
    # -------------------------------------------------------------- filesystem
    _fn("fs_read_file",
        "Read a text file from the workspace. Paths may be absolute or relative to /workspace/demo-app. Large "
        "files are returned in pages: use offset and limit (in lines) to read a specific part. Always read a "
        "file before editing it, and never guess its contents. Binary files are returned as base64 when "
        "encoding is base64.",
        {"path": _s("Path of the file to read."),
         "offset": _i("First line to return, starting at 1.", minimum=1),
         "limit": _i("Maximum number of lines to return.", minimum=1, maximum=5000),
         "encoding": _s("How to decode the file.", enum=["utf-8", "latin-1", "base64"])},
        ["path"]),
    _fn("fs_write_file",
        "Create or replace a file with the given content. Parent directories are created only when create_dirs "
        "is true. An existing file is replaced only when overwrite is true; otherwise the call fails, which "
        "protects against accidental data loss. Use fs_edit_file for small changes to existing files.",
        {"path": _s("Path of the file to write."),
         "content": _s("Complete new content of the file."),
         "overwrite": _b("Replace the file if it already exists."),
         "create_dirs": _b("Create missing parent directories.")},
        ["path", "content"]),
    _fn("fs_edit_file",
        "Apply exact text replacements to an existing file. Each edit replaces old_text with new_text; old_text "
        "must appear exactly once in the file unless replace_all is true, otherwise the whole call fails and the "
        "file is left unchanged. Edits are applied in order, so later edits see the result of earlier ones. "
        "Include enough surrounding context in old_text to make it unique.",
        {"path": _s("Path of the file to edit."),
         "edits": {"type": "array", "description": "Replacements to apply, in order.", "minItems": 1, "items": {
             "type": "object", "properties": {
                 "old_text": _s("Exact text to replace."),
                 "new_text": _s("Replacement text."),
                 "replace_all": _b("Replace every occurrence instead of exactly one.")},
             "required": ["old_text", "new_text"], "additionalProperties": False}}},
        ["path", "edits"]),
    _fn("fs_list_dir",
        "List the entries of a directory with their type, size and modification time. Use recursive with a "
        "small max_depth to get an overview of a project; the output is truncated after 2000 entries. Hidden "
        "files are skipped unless include_hidden is true.",
        {"path": _s("Directory to list."),
         "recursive": _b("Descend into subdirectories."),
         "include_hidden": _b("Include entries whose names start with a dot."),
         "max_depth": _i("Maximum recursion depth when recursive is true.", minimum=1, maximum=10)},
        ["path"]),
    _fn("fs_search_files",
        "Find files by name with a glob pattern such as \"**/*.yaml\" or \"src/**/test_*.py\". This matches file "
        "names and paths only; it does not look inside files (use fs_grep for that). Results are sorted by "
        "modification time, newest first.",
        {"pattern": _s("Glob pattern for file paths."),
         "root": _s("Directory to search from. Defaults to the workspace root."),
         "max_results": _i("Maximum number of paths to return.", minimum=1, maximum=1000),
         "include_hidden": _b("Also match files inside hidden directories.")},
        ["pattern"]),
    _fn("fs_grep",
        "Search file contents with a regular expression and return matching lines with file names and line "
        "numbers. Narrow the search with path and glob; for example glob \"*.py\" searches Python files only. "
        "Use fs_search_files instead when you are looking for files by name.",
        {"pattern": _s("Regular expression to search for."),
         "path": _s("File or directory to search in. Defaults to the workspace root."),
         "glob": _s("Only search files whose names match this glob."),
         "case_insensitive": _b("Ignore letter case when matching."),
         "context_lines": _i("Lines of context to show before and after each match.", minimum=0, maximum=20),
         "max_matches": _i("Stop after this many matches.", minimum=1, maximum=5000)},
        ["pattern"]),
    _fn("fs_move",
        "Move or rename a file or directory. The destination's parent directory must exist. An existing "
        "destination is replaced only when overwrite is true.",
        {"source": _s("Current path."), "destination": _s("New path."),
         "overwrite": _b("Replace the destination if it exists.")},
        ["source", "destination"]),
    _fn("fs_delete",
        "Delete one or more files or directories. Directories are deleted only when recursive is true. Deleted "
        "files cannot be recovered, so double-check the paths, and never delete anything outside the workspace.",
        {"paths": {"type": "array", "description": "Paths to delete.", "items": {"type": "string"}, "minItems": 1},
         "recursive": _b("Delete directories together with their contents.")},
        ["paths"]),
    # ------------------------------------------------------------- shell, dev
    _fn("shell_run",
        "Run a shell command in a non-interactive bash session and return its exit code, stdout and stderr. "
        "The working directory defaults to /workspace/demo-app. Commands that wait for input will time out, so "
        "pass flags such as --yes where needed. Prefer the dedicated fs_* tools for reading, searching and "
        "editing files, and git_* tools for version control. Give a short description of what the command does.",
        {"command": _s("The command line to execute."),
         "cwd": _s("Working directory for the command."),
         "timeout_s": _i("Kill the command after this many seconds.", minimum=1, maximum=1800),
         "env": {"type": "object", "description": "Extra environment variables.",
                 "additionalProperties": {"type": "string"}},
         "description": _s("One sentence describing what the command does and why.")},
        ["command"]),
    _fn("process_kill",
        "Send a signal to a process started in this session, for example a development server that no longer "
        "responds. TERM asks the process to exit; use KILL only if TERM had no effect after a few seconds.",
        {"pid": _i("Process id.", minimum=1),
         "signal": _s("Signal to send.", enum=["TERM", "KILL", "INT", "HUP"])},
        ["pid"]),
    _fn("git_status",
        "Show the current branch, the upstream tracking state and the staged, modified and untracked files of a "
        "repository. Run it before committing to make sure only the intended files are included.",
        {"repo_path": _s("Repository root. Defaults to the workspace root."),
         "include_untracked": _b("List untracked files as well.")},
        []),
    _fn("git_diff",
        "Show changes as a unified diff: unstaged changes by default, or staged changes when staged is true. "
        "Restrict the output to some files with files.",
        {"repo_path": _s("Repository root. Defaults to the workspace root."),
         "staged": _b("Show staged changes instead of unstaged ones."),
         "files": {"type": "array", "description": "Only show these paths.", "items": {"type": "string"}},
         "context_lines": _i("Lines of context around each change.", minimum=0, maximum=50)},
        []),
    _fn("git_commit",
        "Create a commit. When files is given, exactly those paths are staged before committing; otherwise the "
        "current index is committed as it is. The message is used verbatim, including line breaks: the first "
        "line is the summary, followed by a blank line and the body.",
        {"message": _s("Commit message."),
         "files": {"type": "array", "description": "Paths to stage before committing.", "items": {"type": "string"}},
         "amend": _b("Amend the previous commit instead of creating a new one."),
         "allow_empty": _b("Allow a commit without changes.")},
        ["message"]),
    _fn("run_tests",
        "Run the project's test suite or a subset of it and return a structured summary with failures, "
        "durations and the relevant part of the output. Use filter to run a single test file or test name while "
        "iterating, and the full suite before you report that a change is done.",
        {"runner": _s("Test runner to use.", enum=["pytest", "jest", "go", "cargo"]),
         "path": _s("Test file or directory."),
         "filter": _s("Runner-specific selection expression, for example a test name pattern."),
         "fail_fast": _b("Stop at the first failure."),
         "timeout_s": _i("Abort the run after this many seconds.", minimum=10, maximum=3600)},
        ["runner"]),
    _fn("python_exec",
        "Execute a short Python 3 snippet in a fresh interpreter and return stdout, stderr and the value of the "
        "last expression. Use it for quick calculations, data inspection or format conversions, not for "
        "long-running work. The interpreter has no network access.",
        {"code": _s("Python source code to execute."),
         "timeout_s": _i("Abort after this many seconds.", minimum=1, maximum=300)},
        ["code"]),
    # ---------------------------------------------------------- browser, web
    _fn("browser_navigate",
        "Open a URL in the browser, in the current tab or a new one, and wait until the page has loaded. Returns "
        "the final URL after redirects, the page title and the tab id. Use http_request instead when you only "
        "need the raw response of an API endpoint.",
        {"url": _s("Absolute URL to open."),
         "new_tab": _b("Open the URL in a new tab."),
         "wait_until": _s("Load state to wait for.", enum=["load", "domcontentloaded", "networkidle"])},
        ["url"]),
    _fn("browser_get_text",
        "Return the visible text of the current page, or of the elements matching a CSS selector, with links "
        "listed at the end. Long pages are truncated to max_chars characters.",
        {"selector": _s("CSS selector. Omit to read the whole page."),
         "max_chars": _i("Truncate the text after this many characters.", minimum=100, maximum=200000),
         "tab_id": _i("Tab to read. Defaults to the active tab.", minimum=0)},
        []),
    _fn("browser_click",
        "Click an element on the page, identified by a CSS selector or by its visible text. Waits for the "
        "element to become visible and enabled before clicking, up to five seconds.",
        {"selector": _s("CSS selector of the element."),
         "text": _s("Visible text of the element, used when no selector is given."),
         "tab_id": _i("Tab to act in. Defaults to the active tab.", minimum=0)},
        []),
    _fn("browser_fill_form",
        "Fill several form fields at once and optionally submit the form. Each field is located by a CSS "
        "selector; text inputs are cleared before the value is entered, checkboxes accept \"true\" or \"false\", "
        "and select elements accept the visible option text.",
        {"fields": {"type": "array", "description": "Fields to fill, in order.", "minItems": 1, "items": {
            "type": "object", "properties": {"selector": _s("CSS selector of the field."),
                                             "value": _s("Value to enter.")},
            "required": ["selector", "value"], "additionalProperties": False}},
         "submit": _b("Submit the form after filling it."),
         "tab_id": _i("Tab to act in. Defaults to the active tab.", minimum=0)},
        ["fields"]),
    _fn("browser_close_tabs",
        "Close browser tabs by id. Tab ids are shown in the tab strip and returned by browser_navigate. Closing "
        "the last tab leaves an empty browser window open.",
        {"tab_ids": {"type": "array", "description": "Ids of the tabs to close.", "items": {"type": "integer"},
                     "minItems": 1}},
        ["tab_ids"]),
    _fn("http_request",
        "Send an HTTP request and return the status code, headers and body. JSON bodies are serialized from "
        "json_body and the Content-Type header is set automatically. Only hosts on the local network and the "
        "documentation mirror are reachable. Use this instead of the browser for APIs, health checks and "
        "machine-readable endpoints.",
        {"method": _s("HTTP method.", enum=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]),
         "url": _s("Absolute URL."),
         "headers": {"type": "object", "description": "Extra request headers.",
                     "additionalProperties": {"type": "string"}},
         "query": {"type": "object", "description": "Query parameters appended to the URL.",
                   "additionalProperties": {"type": "string"}},
         "json_body": {"type": "object", "description": "JSON request body for POST, PUT and PATCH."},
         "timeout_s": _n("Request timeout in seconds.", minimum=0.5, maximum=300)},
        ["method", "url"]),
    _fn("web_search",
        "Search the offline documentation mirror and return titles, URLs and short snippets. Use specific "
        "queries with library names and version numbers; open promising results with browser_navigate.",
        {"query": _s("Search query."),
         "max_results": _i("Number of results to return.", minimum=1, maximum=20),
         "site": _s("Restrict results to one documentation site, for example \"docs.python.org\"."),
         "recency": _s("Only return pages updated within this period.", enum=["day", "week", "month", "year", "any"])},
        ["query"]),
    # -------------------------------------------------------------- productivity
    _fn("notes_create",
        "Create a note in the Notes app. The body supports Markdown. Tags help to find the note later; pinned "
        "notes are shown at the top of the list.",
        {"title": _s("Title of the note."), "body": _s("Content of the note, in Markdown."),
         "tags": {"type": "array", "description": "Tags to attach.", "items": {"type": "string"}},
         "pinned": _b("Pin the note to the top of the list.")},
        ["title", "body"]),
    _fn("calendar_create_event",
        "Create an event in the local Calendar app. Times are ISO 8601 date-times in the workstation's time "
        "zone, for example 2026-11-03T10:00. A recurring event repeats with the given frequency and interval "
        "until count occurrences have been created.",
        {"title": _s("Title of the event."),
         "start": _s("Start date-time, ISO 8601."),
         "end": _s("End date-time, ISO 8601."),
         "location": _s("Room or place, free text."),
         "reminders_min": {"type": "array", "description": "Reminders, in minutes before the start.",
                           "items": {"type": "integer", "minimum": 0}},
         "recurrence": {"type": "object", "description": "Repeat rule. Omit for a single event.", "properties": {
             "frequency": _s("How often the event repeats.", enum=["daily", "weekly", "monthly"]),
             "interval": _i("Repeat every N periods.", minimum=1),
             "count": _i("Total number of occurrences.", minimum=1)},
             "required": ["frequency"], "additionalProperties": False}},
        ["title", "start", "end"]),
    _fn("task_create",
        "Add a task to the project task list. Use it to track follow-up work that is out of scope for the "
        "current change, so it is not forgotten.",
        {"title": _s("Short title of the task."),
         "due_date": _s("Due date, ISO 8601 date such as 2026-11-15."),
         "priority": _s("Priority of the task.", enum=["low", "medium", "high", "urgent"]),
         "checklist": {"type": "array", "description": "Sub-steps of the task.", "items": {"type": "string"}},
         "project": _s("Project the task belongs to.")},
        ["title"]),
    _fn("settings_set",
        "Change a setting of an application, such as the editor's tab size or the terminal's font. Workspace "
        "settings apply to the current project only; user settings apply everywhere.",
        {"app": _s("Application whose setting is changed."),
         "key": _s("Setting key, for example \"editor.formatOnSave\"."),
         "value": {"description": "New value of the setting.", "anyOf": [{"type": "string"}, {"type": "number"},
                                                                         {"type": "boolean"}]},
         "scope": _s("Where the setting applies.", enum=["user", "workspace"])},
        ["app", "key", "value"]),
    _fn("system_set_volume",
        "Set the output volume of the workstation. level is a fraction between 0 (silent) and 1 (maximum); "
        "muting keeps the level so that unmuting restores it.",
        {"level": _n("Volume level between 0 and 1.", minimum=0, maximum=1),
         "muted": _b("Mute or unmute the output.")},
        ["level"]),
    _fn("archive_create",
        "Pack files and directories into an archive. The destination's extension should match the format. "
        "Higher compression levels are slower and rarely worth it for already-compressed data such as images.",
        {"paths": {"type": "array", "description": "Files and directories to include.", "items": {"type": "string"},
                   "minItems": 1},
         "destination": _s("Path of the archive to create."),
         "format": _s("Archive format.", enum=["zip", "tar.gz", "tar.zst"]),
         "compression_level": _i("Compression level from 0 (store) to 9 (maximum).", minimum=0, maximum=9)},
        ["paths", "destination"]),
]

TOOL_INDEX = {t["function"]["name"]: t for t in TOOLS}


def anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same tools in Anthropic Messages format (name / description / input_schema)."""
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in tools]
