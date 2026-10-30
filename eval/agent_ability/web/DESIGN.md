# Web search suite: design

Draft of 2026-10-30, for review.

## 0. Decisions of 2026-10-30 (they override the sections below where they differ)

- **Arms:** D (chrome-devtools-mcp text tools), **S-cdp** (chrome-devtools-mcp with `--experimentalVision`:
  `take_screenshot` plus `click_at`, `type_text`, `press_key`, scroll, all sent through CDP, so no real pointer or
  keyboard is used), and C (closed book). S-cdp replaces cua-driver as the S arm. **S-cua** (cua-driver, OS-level
  input, a headed Chrome) is a fourth arm that runs only when `--arms` names it; it needs a dedicated macOS Space
  (`--cua-space`).
- **Chrome for D and S-cdp runs headless** (`--headless=new`) with the user agent of the installed Chrome's normal
  build (`Chrome/<major>.0.0.0`), so nothing appears on the screen and runs may go while the Mac is in use. The
  window is 1280×800 and the headless viewport 1280×713. A hand test (Chrome 156) reached Google results with AI
  Overviews and no CAPTCHA. S-cua alone runs a visible window.
- **Search:** Google as served, AI Overviews kept. **Chrome:** the installed stable build; a version change stops a
  study.
- **Harnesses:** Claude Code, Hermes and **localmelo** from the start. localmelo runs D, S and C. Tool-result
  images reach its model with `-X agent.tool_images=on`.
- **Model:** `--model`, default `Qwen3.8-Flash-Next-oQ4e-mtp`, for all three harnesses. **Claude Code binary:**
  `--claude-bin`, default `eval/backend/vendor/claude-code/2.1.312`, the one the capability reference run pins.
- **chrome-devtools-mcp** 1.11.0 is installed in `web/mcp/` (exact pin, lockfile) and needs
  `--no-page-id-routing` (otherwise every page tool requires a `pageId`). It runs with `--categoryDebugging` left
  on, because `--categoryDebugging=false` removes `take_snapshot` and `take_screenshot`.
- **Snapshot size:** a Google results `take_snapshot` is about 47K characters, mostly `url="..."` attributes. The
  tap rewrites each `url="<full URL>"` to `url="<host>"` for every harness alike; clicking uses `uid`, not URLs.
- **Answer aliases:** rows 3538, 3352, 2849 and 69 keep their place and get hand-written aliases (no swaps).

## 1. Goal

Agents answer short factual questions by operating Google Chrome:

- **D (DOM):** chrome-devtools-mcp 1.11.0, which reads pages as accessibility-tree text.
- **S (screenshot):** chrome-devtools-mcp 1.11.0 with `--experimentalVision` (S-cdp), which works from
  screenshots and sends coordinates, keys and text through CDP.
- **C (closed book):** no tools.

With the answer one search away (SimpleQA Verified's authors call its questions easy with tools), a
failure measures how the agent drives the browser.

## 2. Where it lives and how it runs

It is a study runner in `eval/agent_ability/web/`, like `e3/`. It imports `smoke` and `bench.harness`
and edits neither, so the harness epoch is unchanged.

smoke's `run_one` refuses unknown task kinds, and `adapters.overrides` knows only the filesystem server.
So `web/run.py` has its own `execute` of the same shape, and `web/arms.py` its own builders, both built on
smoke's adapter helpers and workspace lock. Resume, `STOP`, the lock and battery checks, and
`environment.json` follow e3. `environment.json` also records the model, the versions of Chrome, Claude Code (its
`--version`), chrome-devtools-mcp (from `web/mcp/package.json`), localmelo and, with S-cua, cua-driver, and the
macOS and Xcode builds (`sw_vers`, `xcodebuild -version`).

```
cd eval
uv run python agent_ability/web/run.py --run-dir agent_ability/runs/web/<study> [--dry-run]
uv run python agent_ability/web/report.py agent_ability/runs/web/<study>
```

Options of `run.py`:

- `--harnesses`, `--arms` and `--repeats`: the cells and their count (default: all three harnesses, D, S and C,
  one repeat; `S-cua` only when named);
- `--questions`: pilot ids, `easy`, `medium` or `all` (default `all`);
- `--localmelo`: the clone; default `eval/agent_ability/e3/vendor/localmelo-a9fb42b`, whose commit is pinned and
  whose tree must be clean;
- `--base-url`: the model endpoint, `http://host:port/v1` (default: the bench manifest's);
- `--model`: the model id (default `Qwen3.8-Flash-Next-oQ4e-mtp`). It reaches Hermes's config and localmelo's
  `--chat-model` through `bench.harness.MODEL`, and Claude Code through its per-run copies (section 5);
- `--claude-bin`: the Claude Code binary (default `eval/backend/vendor/claude-code/2.1.312`);
- `--cua-space`: the macOS Space number of the S-cua Chrome window; required when S-cua is among the arms;
- `--expected-chrome-version`: refuse any other installed Chrome (default: record the installed one);
- `--captcha-pause-s`: the pause before a retry (default 1800);
- `--no-gpu-lock`: do not require the GPU lock, for a stub model.

Files in the study directory: `web.json` (the checkpoint), `schedule.json`, `environment.json`, `web.log`, and
per run `runs/<run_id>/` with `record.json`, `tap.jsonl`, `mcp.json` (localmelo), `chrome/` (`History`,
`tabs.json`, `chrome.json`) and `<harness>/` (the bench run directory).

**Promotion** to the capability leg means a new epoch and a new reference run. It needs:

- `web/` as a snapshot part, or the task kind moved into `smoke/`;
- live pins on:
  - Chrome for Testing (stable Chrome updates itself);
  - the chrome-devtools-mcp lock;
  - cua-driver, for S-cua;
  - the profile template's hash;
  - the dataset revision;
- its own task set, with a decision rule fixed in advance.

D, S and C all run headless. Only S-cua needs an unattended GUI session.

## 3. Arms, harnesses and schedule

| Arm | Tools | Chrome |
|---|---|---|
| D | 13 chrome-devtools-mcp tools | on the start page |
| S | 5 bench-defined tools over chrome-devtools-mcp (`--experimentalVision`) | on the start page |
| S-cua | the same 5 tools over cua-driver | on the start page, headed, on the Space on screen |
| C | none | not launched |

**Harnesses: Claude Code, Hermes and localmelo**, each on D, S and C (and S-cua when named). All three pass
tool-result images to the model: Claude Code by default, Hermes with `image_input_mode: native` and
`supports_vision: true`, localmelo with `-X agent.tool_images=on`.

**Budget.** Flash-Next (vision), thinking off, 30 proxy steps and 40 turns. The wall cap is 1800 s for
D, S and S-cua and 600 s for C.

**Order** is repeat-major. Each (repeat, question) runs its nine cells (harness × arm) back to back,
rotated as in e3 by (question index + repeat), and the C runs space out Google queries.

| Stage | Runs | Estimate |
|---|---|---|
| 0 plumbing | 2 questions × 3 arms × 3 harnesses = 18 | ~65 min |
| 1 | 24 × 3 arms × 3 harnesses = 216 | ~13 h |
| 2 | second repeat, 216 | ~13 h, if Stage 1 is informative |

Estimated time per run: C 0.5 min, D 4 min and S 5.5 min, plus 20 s for Chrome. These numbers scale
Flash-Next's 40-80 s Aider runs to D's 5-30K-token snapshots and S's extra steps. They are not measured
yet; Stage 0 will replace them.

## 4. Chrome lifecycle

**Launch.** The runner starts the installed Google Chrome with the same flags for both arms. The binary is
`$WEB_CHROME_BINARY` if set, else the executable of the app with Chrome's bundle id (found through Spotlight).
The tap's chrome-devtools-mcp child then attaches through `--browserUrl http://127.0.0.1:<port>`. If chrome-devtools-mcp launched Chrome
itself, only D would get Puppeteer's `--enable-automation`, which adds an infobar and
`navigator.webdriver`, a bot signal to Google. Remote debugging is allowed because the bench profile is
not Chrome's default one.

**Profile.** The template lives at `eval/agent_ability/runs/web/profile-template/`, gitignored through
`runs/`. `web/chrome.py init` makes it once, with the run flags:

- first-run UI dismissed by hand;
- no sign-in or sync;
- English;
- notification and location prompts blocked;
- Google as the search engine.

Each run uses an APFS clone (`cp -c`) in its run directory, so every run starts with the same cookies
and no history from earlier runs. Studies never open the template. A profile serves one Chrome at a
time, so runs are sequential under the workspace lock, and preflight refuses while any Chrome uses a
bench profile.

```
--headless=new --user-data-dir=<clone> --remote-debugging-port=<free port> --remote-debugging-address=127.0.0.1
--no-first-run --no-default-browser-check --use-mock-keychain --disable-background-networking
--disable-features=Translate --lang=en-US --window-size=1280,800 --user-agent=<normal build, Chrome/<major>.0.0.0>
--host-resolver-rules="MAP huggingface.co ~NOTFOUND, MAP hf.co ~NOTFOUND, MAP *.kaggle.com ~NOTFOUND, ..."
```

S-cua alone runs without `--headless=new` and with `--window-position=0,25`. The window is 1280×800 and the
headless viewport 1280×713. The mock keychain leaves the user's keychain alone. With background networking off,
Chrome sends no telemetry. Translate's bubble would hide page content from S. The resolver rules block
answer-key hosts: Hugging Face, Kaggle, OpenAI's blob store and `raw.githubusercontent.com`.

**Per run:**

1. Clone the profile and launch Chrome.
2. Over CDP's HTTP endpoints, leave one tab, on `https://www.google.com/?hl=en`.
3. Run the harness; t0 is its start.
4. Save the tab list, then close Chrome over CDP (`Browser.close`). A SIGTERM leaves about the last ten seconds
   of History uncommitted. If Chrome is still running after 10 s, SIGTERM; SIGKILL after 15 s more.
5. Keep `Default/History` and delete the clone.

The first run records Chrome's version, and `bench_chrome` refuses any other. A change in Chrome's version
stops the study.

**Search:** Google as served, with AI Overviews. A run with a `google.com/sorry` URL in its History or its
tabs is marked `env_blocked` and re-queued once, in a run directory of its own, after a 30-minute pause
(`--captcha-pause-s`). Two blocked runs in a row stop the study.

**S-cua** runs a headed Chrome (`bench_chrome(headless=False)`), and the tap gets its pid (`--chrome-pid`). It needs:

- CuaDriver.app's Accessibility and Screen Recording grants, and its daemon running (preflight runs
  `cua-driver status`);
- the Chrome window on the Space on screen, because a window on another Space gives no screenshot.

Input is OS-level but sent in the background: no pointer move and no focus change. The Mac must be awake and
unlocked, with the Space on screen the one given by `--cua-space`.

- Before each S-cua run the runner reads the Space on screen (`cua-driver call list_windows '{}'`, its
  `current_space_id`). If it is not `--cua-space`, it rechecks every 60 s for up to 30 min, then stops the study
  (exit 3) and writes no record.
- During a run the tap refuses every call while the browser's Space is not on screen ("not on screen"). Such a
  run is `env_blocked` (`blocked_by: space`): it is left out of the counts and run again at once, without the
  CAPTCHA pause. Two blocked runs in a row stop the study.

## 5. Tools, prompt and answer protocol

In every arm, the MCP server is `web/tap.py`, a stdlib stdio relay named `webtap` in all three harnesses
(Hermes has toolsets named `web` and `browser`, and `-t <name>` of an MCP server with such a name would select
the built-in tools too). It logs every call to `<run_dir>/tap.jsonl`. Only its mode and tool list vary: `--mode
dom` (D), `--mode screen` (S), `--mode cua` (S-cua) and `--mode none` (C); D and S add `--browser-url` of the run's
Chrome, S-cua `--chrome-pid` and `--cua-bin`.

The harness starts the tap with the runner's own interpreter (`sys.executable`). The tap's environment is set
in the MCP config, because localmelo and Hermes give an MCP server a reduced one: `PATH` is the directory of the Node
in `web/mcp/.nvmrc` (found under nvm at run time), then `/usr/bin:/bin`; and
`CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS=1` and `CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS=1`.
S-cua adds the real `HOME`: cua-driver finds its daemon through it, and `bench.harness` gives localmelo an empty
HOME of its own.

- **D:**
  - chrome-devtools-mcp is installed in `web/mcp/`; the tap starts `node_modules/.bin/chrome-devtools-mcp` as its
    child.
  - Flags: `--no-page-id-routing --usageStatistics=false --performanceCrux=false --javascriptEvaluation=false
    --fileNavigations=false --categoryEmulation=false --categoryPerformance=false --categoryNetwork=false
    --categoryMemory=false`.
  - `--categoryDebugging` stays at its default (on): `--categoryDebugging=false` removes `take_snapshot` and
    `take_screenshot`.
  - The tap passes `navigate_page`, `new_page`, `list_pages`, `select_page`, `close_page`, `wait_for`,
    `take_snapshot`, `click`, `hover`, `fill`, `type_text`, `press_key` and `handle_dialog`.
  - It drops `new_page.isolatedContext`, which bypasses History, and withholds `take_screenshot` and
    `click_at`, which would make D a screenshot arm.
- **S (S-cdp):**
  - The child is chrome-devtools-mcp with `--experimentalVision` and `--screenshotMaxWidth=1280
    --screenshotMaxHeight=800`.
  - Tools: `screenshot`, `click`, `type_text`, `press_keys` and `scroll`.
  - Coordinates are 0-1000: in ui_vision, Flash-Next clicked right 54/56 times that way and 11/56 in
    pixels. The tap maps them to the CSS pixels of the viewport.
  - Scroll is PageDown/PageUp, by whole screens. Back, forward and reload (Alt+ArrowLeft, Alt+ArrowRight, F5 and
    the Command forms) map to the child's `navigate_page`, because CDP key events never reach Chrome's own
    shortcuts.
  - Screenshots are JPEGs of the 1280×713 viewport, of at most 256 KB, returned as an image block only, with no
    `structuredContent` copy.
  - The tap refuses paste, quit, DevTools, incognito and new windows.
  - The child's text results, which hold URLs and page titles, are never shown.
- **S-cua:** the same five tools, served by cua-driver on the window of the run's headed Chrome. Screenshots are of
  the whole window, browser UI included.
- **C:** none.

Per harness:

- Claude Code: `--tools "" --allowedTools "mcp__webtap__*"`, `--mcp-config` with the tap, and
  `ENABLE_TOOL_SEARCH=false`. `bench.harness` would source `agent/configs/claude-code/claude-code.env`, which pins a
  Claude Code binary that an update can delete and the model of an earlier study. So each run gets a copy of that
  file with `CLAUDE_BIN` and the five model variables replaced (`bench.harness.CLAUDE_ENV` points at it for the
  run), and its copied home's `settings.json` gets `"model"` replaced. Preflight sources the copy as the launcher
  does and refuses if any other line changed, a safety variable (`ANTHROPIC_API_KEY` empty,
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `..._CLAUDE_MDS`, `..._AUTO_MEMORY`) is lost, or a value is off. The
  code is the minimum of `modeleval/capability/runner.py`, ported because that package cannot be imported here.
- Hermes: only `mcp_servers.webtap` (every other server disabled), `tools.tool_search.enabled: off` and
  `-t webtap`. S and S-cua also set `agent.image_input_mode: native` and `model.supports_vision: true`.
- localmelo: `--mcp <mcp.json> --mcp-only -o json` before the `--` that precedes the prompt, and for S and S-cua
  `-X agent.tool_images=on`. Its checkout is pinned to commit `a9fb42b`; preflight refuses another commit or a
  dirty tree.

The runner swaps the harness builders of `bench.harness` (and, for Claude Code, its `CLAUDE_ENV`) for one run and
restores them afterwards, as smoke's adapters do. For the length of the study it also sets `bench.harness.MODEL`,
the manifest's model and Claude Code binary, the model endpoint and the localmelo clone, and restores them. The
harness epoch does not change.

D, S and S-cua get byte-identical prompts, which quote GAIA's first two answer rules:

```
A Google Chrome window is open on Google Search. Use the browser tools to search the web and
answer the question below, checking the answer on the pages you find. Only read: do not sign in,
post, buy or change settings.

Question: {problem}

End your reply with one line: FINAL ANSWER: <answer>
Your final answer should be a number OR as few words as possible OR a comma separated list of
numbers and/or strings. If you are asked for a number, don't use comma to write your number
neither use units such as $ or percent sign unless specified otherwise. Write a date as day,
month name and year. If you cannot find the answer, write FINAL ANSWER: unknown
```

C's first paragraph is "Answer the question below from your own knowledge. You have no tools."

## 6. Task data

`web/fetch.py` downloads `simpleqa_verified.csv` from `google/simpleqa-verified` with `hf_hub_download`,
once, into `$HF_HOME`. It pins revision `0dc97e0d28d8233463e005cdc4475cc2a13ba2dc` and records the
file's sha256.

`web/data/simpleqa_verified_pilot.yaml` holds:

- the revision, the hash, the MIT notice and the citation;
- the dataset's fields for each row;
- the hand-set `kind`, `aliases` and `unordered` for each row.

Rows 3538, 3352, 2849 and 69 get drafted aliases or a same-tier swap. A test checks the rows against the
CSV when it is present. Aliases added later are dated, and every run is rescored; all arms share them.
FRAMES (6-8 rows) is an optional add-on after Stage 1.

## 7. Scoring

Deterministic; no LLM judge.

1. **Extract** the answer from the last `FINAL ANSWER:` line, case-insensitive, after markup is
   stripped. If there is no such line, the verdict is `no_answer`; an answer of `unknown`, or an empty
   one, is `abstain`.
2. **Pre-normalize** both sides:
   - NFKD fold;
   - trim;
   - drop a trailing `.` and a leading `the `.
3. **Compare** with the gold answer and each alias, by `kind`:
   - `number`: strip `$ % ,` and whitespace. What is left must be one number, plus an optional unit.
     Exact up to 50, ±1% up to 10,000, ±5% above.
   - `year`: exact (±1% would accept 1968 for 1987).
   - `date`: day, month name and year in any order, or ISO; each component the gold has must match.
   - `string`: GAIA's `normalize_str` (whitespace and punctuation removed, lowercase), then equality.
   - `list`: GAIA's rule. Split on `,` and `;`, require equal length, and compare in order (or as
     multisets when `unordered`).

`tests/test_web_score.py` covers:

- each step;
- `$1,234`;
- the tolerance edges at 50, 51, 10,000 and 20,000;
- hedges such as `12 or 13`;
- dates and lists;
- an oracle in which every gold answer and alias passes and decoys fail.

## 8. Metrics and report

Navigation counts top-level History visits between t0 and exit, ends of redirect chains only.

| Group | Fields |
|---|---|
| verdict | `pass`, `fail`, `abstain`, `no_answer`, `timeout`, `harness_error`, `env_blocked` and `interrupted`; the last two are excluded |
| cost | `n_llm` and `n_tool` (C counts tool-less `aux` requests too), `output_tokens`, `uncached_tokens`, `n_images`, `wall_s` |
| navigation | `n_nav`, `n_search`, `n_pages`, `gold_url_visited`, `zero_nav` (a pass or fail with `n_nav = 0`), `serp_only`; null for C |
| flags | `captcha`, `blocked_by` (`captcha` or `space` when the verdict is `env_blocked`), `space_lost` (S-cua: refused calls), `answer_key_suspect` (a URL or title matching `simple[ _-]?qa`), `foreign_tools` and `search_tool_leak` (a tool in the model's requests that the web server did not offer), `mcp_output_truncated` (null: not derivable yet) |

A run is `env_blocked` when a `google.com/sorry` URL is in its History or tabs, or, for S-cua, the tap refused
calls because the browser's Space was not on screen; `harness_error` when the harness
raised or no model call succeeded, `timeout` when the wall cap killed it, and otherwise the answer's grade.
`record.json` holds these fields, the visited URLs, and the tap's call counts by tool and status.

Per harness and arm, the report gives:

- verdicts by tier, with costs;
- zero-navigation answers and their accuracy;
- D and S accuracy, split by whether C ever got the question right;
- D vs S, paired by question: the arm with the higher pass rate over a question's counted repeats wins it, and
  ties are dropped. The exact two-sided sign test (α = 0.05) and the minimum detectable difference, the smallest
  net margin the test calls significant, are the backend report's (`modeleval/stats.py`).

## 9. Risks and open questions

Risks:

- Flash-Next's tool-result images are unproven; the G2b probe must pass before S runs.
- Long pages in D can hit Claude Code's MCP output cap.
- A harness may offer a search tool of its own next to the tap's. `foreign_tools` and `search_tool_leak` in each
  record show it; Stage 0 must check.
- S-cua depends on the Space staying on screen for a run; a lost Space costs the run.
- 24 questions support only descriptive conclusions.

Decided in section 0: AI Overviews kept, S-cdp for S, stable Chrome with a stop on an update, hand-written
aliases for rows 3538, 3352, 2849 and 69.

## 10. Build plan

Each step is one agent's work and runs only its own tests (`cd eval; uv run pytest
agent_ability/tests/<file>`). The full suite runs before the PR.

| Step | Files | Tests |
|---|---|---|
| 1 Data | YAML, `data.py`, `fetch.py`, drafted aliases | `test_web_data.py`: schema, tiers, gold hosts, CSV check |
| 2 Scoring | `score.py`, `answer.py` | `test_web_score.py`; `test_web_answer.py` on synthetic stdout |
| 3 Tap | `tap.py` (modes `dom`, `screen`, `none`) | `test_web_tap.py`, fake child server |
| 4 Chrome | `chrome.py`, `history.py` | synthetic History; fake `/json/*` server; opt-in real Chrome |
| 5 Arms | `arms.py` | `test_web_arms.py`: command lines, Hermes config, restoration |
| 6 Runner | `run.py` | `test_web_run.py`, fakes: verdicts, resume, CAPTCHA pause, identical D/S prompts |
| 7 Report | `report.py` | `test_web_report.py`, synthetic records |
| 8 Calibration | `calibrate.py`: D and S scripted on a local page, no model | then the G2b probe and Stage 0 |
| 9 Docs | `README.md`, `.gitignore` for `web/mcp/node_modules/` | none |
