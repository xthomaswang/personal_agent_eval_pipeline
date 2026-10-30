# Web search suite: design

Draft of 2026-10-30, for review. Nothing is built yet.

## 0. Decisions of 2026-10-30 (they override the sections below where they differ)

- **Arms:** D (chrome-devtools-mcp text tools), **S-cdp** (chrome-devtools-mcp with `--experimentalVision`:
  `take_screenshot` plus `click_at`, `type_text`, `press_key`, scroll, all sent through CDP, so no real pointer or
  keyboard is used), **S-cua** (cua-driver, OS-level input; deferred until a dedicated macOS Space exists for it),
  and C (closed book).
- **Chrome for D and S-cdp runs headless** (`--headless=new`) with the user agent of the installed Chrome's normal
  build (`Chrome/<major>.0.0.0`), so nothing appears on the screen and runs may go while the Mac is in use. A
  hand test (Chrome 156) reached Google results with AI Overviews and no CAPTCHA. S-cua alone needs a visible window.
- **Search:** Google as served, AI Overviews kept. **Chrome:** the installed stable build; a version change stops a
  study.
- **Harnesses:** Claude Code, Hermes and **localmelo** from the start. localmelo runs D and C; it joins the S arms
  once its tool-result image switch lands (today an image tool result reaches its model only as a text
  placeholder).
- **chrome-devtools-mcp** 1.11.0 is installed in `web/mcp/` (exact pin, lockfile) and needs
  `--no-page-id-routing` (otherwise every page tool requires a `pageId`).
- **Snapshot size:** a Google results `take_snapshot` is about 47K characters, mostly `url="..."` attributes. The
  tap rewrites each `url="<full URL>"` to `url="<host>"` for every harness alike; clicking uses `uid`, not URLs.
- **Answer aliases:** rows 3538, 3352, 2849 and 69 keep their place and get hand-written aliases (no swaps).

## 1. Goal

Agents answer short factual questions by operating Google Chrome:

- **D (DOM):** chrome-devtools-mcp 1.11.0, which reads pages as accessibility-tree text.
- **S (screenshot):** cua-driver 0.38.0, which works from screenshots and pixel input.
- **C (closed book):** no tools.

With the answer one search away (SimpleQA Verified's authors call its questions easy with tools), a
failure measures how the agent drives the browser.

## 2. Where it lives and how it runs

It is a study runner in `eval/agent_ability/web/`, like `e3/`. It imports `smoke` and `bench.harness`
and edits neither, so the harness epoch is unchanged.

smoke's `run_one` refuses unknown task kinds, and `adapters.overrides` knows only the filesystem server.
So `web/run.py` has its own `run_one` of the same shape, built on smoke's adapter helpers and locks.
Resume, `STOP`, the lock and battery checks, and `environment.json` follow e3. `environment.json` also
records the versions of Chrome, chrome-devtools-mcp and cua-driver.

```
cd eval
uv run python agent_ability/web/run.py --run-dir agent_ability/runs/web/<study> [--dry-run]
uv run python agent_ability/web/report.py agent_ability/runs/web/<study>
```

**Promotion** to the capability leg means a new epoch and a new reference run. It needs:

- `web/` as a snapshot part, or the task kind moved into `smoke/`;
- live pins on:
  - Chrome for Testing (stable Chrome updates itself);
  - the chrome-devtools-mcp lock;
  - cua-driver;
  - the profile template's hash;
  - the dataset revision;
- its own task set, with a decision rule fixed in advance.

D and C go first. S needs an unattended GUI session.

## 3. Arms, harnesses and schedule

| Arm | Tools | Chrome |
|---|---|---|
| D | 13 chrome-devtools-mcp tools | on the start page |
| S | 5 bench-defined tools over cua-driver | on the start page |
| C | none | not launched |

**Harnesses: Claude Code and Hermes.** smoke's adapters already wire MCP for both, and both pass
tool-result images to the model (Hermes with `image_input_mode: native` and `supports_vision: true`).
localmelo cannot run S: its MCP client is unsmoked, and images do not reach its model. It joins D and C
later, outside the D/S pairing.

**Budget.** Flash-Next (vision), thinking off, 30 proxy steps and 40 turns. The wall cap is 1800 s for
D and S and 600 s for C.

**Order** is repeat-major. Each (repeat, question) runs its six cells back to back, rotated as in e3,
and the C runs space out Google queries. Each harness gets one warm-up per session.

| Stage | Runs | Estimate |
|---|---|---|
| 0 plumbing | 2 questions × 3 × 2 = 12 | ~45 min |
| 1 | 24 × 3 arms × 2 harnesses = 144 | ~8.5 h |
| 2 | second repeat, 144 | ~8.5 h, if Stage 1 is informative |

Estimated time per run: C 0.5 min, D 4 min and S 5.5 min, plus 20 s for Chrome. These numbers scale
Flash-Next's 40-80 s Aider runs to D's 5-30K-token snapshots and S's extra steps. They are not measured
yet; Stage 0 will replace them.

## 4. Chrome lifecycle

**Launch.** The runner starts the installed Google Chrome with the same flags for both arms. D's server
then attaches through `--browserUrl http://127.0.0.1:<port>`. If chrome-devtools-mcp launched Chrome
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
--user-data-dir=<clone> --remote-debugging-port=<free port> --no-first-run --no-default-browser-check
--use-mock-keychain --disable-background-networking --disable-field-trial-config
--disable-features=Translate --lang=en-US --window-position=0,25 --window-size=1280,800
--host-resolver-rules="MAP huggingface.co ~NOTFOUND, MAP hf.co ~NOTFOUND, MAP *.kaggle.com ~NOTFOUND, ..."
```

The mock keychain leaves the user's keychain alone. With background networking and field trials off,
Chrome's features stay fixed and it sends no telemetry. Translate's bubble would hide page content from
S. The resolver rules block answer-key hosts: Hugging Face, Kaggle, OpenAI's blob store and
`raw.githubusercontent.com`.

**Per run:**

1. Clone the profile and launch Chrome.
2. Over CDP's HTTP endpoints, leave one tab, on `https://www.google.com/?hl=en`.
3. Run the harness; t0 is its start.
4. Save the tab list, then SIGTERM, which flushes History; SIGKILL after 15 s.
5. Keep `Default/History` and delete the clone.

A change in Chrome's version stops the study.

**Search:** Google as served, with AI Overviews. A run that hits Google's `/sorry/` CAPTCHA page is
marked `env_blocked` and re-queued once after a 30-minute pause. Two in a row stop the study.

**S needs from cua-driver:**

- CuaDriver.app's existing Accessibility and Screen Recording grants;
- `list_windows`, to find Chrome's window;
- `get_window_state` without the accessibility tree;
- the input tools with `delivery_mode: foreground`, because Chromium ignores background wheel events and
  omnibox keys;
- the agent cursor off.

Foreground input moves the real pointer, so S needs the Mac awake, unlocked and unattended.

## 5. Tools, prompt and answer protocol

In every arm, the MCP server is `web/tap.py`, a stdlib stdio relay named `web` in both harnesses. It
logs every call. Only its tool list varies:

- **D:**
  - chrome-devtools-mcp is installed in `web/mcp/` and launched by absolute path (Hermes rewrites a
    bare `npx`).
  - Flags: `--usageStatistics=false --performanceCrux=false --javascriptEvaluation=false
    --fileNavigations=false --categoryEmulation=false --categoryPerformance=false
    --categoryNetwork=false --categoryMemory=false`, with `CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS=1`.
  - Debugging stays on, for `take_snapshot`.
  - The tap passes `navigate_page`, `new_page`, `list_pages`, `select_page`, `close_page`, `wait_for`,
    `take_snapshot`, `click`, `hover`, `fill`, `type_text`, `press_key` and `handle_dialog`.
  - It drops `new_page.isolatedContext`, which bypasses History, and withholds `take_screenshot` and
    `click_at`, which would make D a screenshot arm.
- **S:**
  - Tools: `screenshot`, `click`, `type_text`, `press_keys` and `scroll`.
  - Coordinates are 0-1000: in ui_vision, Flash-Next clicked right 54/56 times that way and 11/56 in
    pixels.
  - Screenshots are 1280×800 JPEGs of at most 256 KB, returned as an image block only, with no
    `structuredContent` copy.
  - The tap pins the window and refuses paste, quit, DevTools and incognito.
  - cua-driver's own tools stay hidden, because they expose the accessibility tree.
- **C:** none.

Claude Code runs with `--tools "" --allowedTools "mcp__web__*"` and `ENABLE_TOOL_SEARCH=false`. Hermes
runs with only `mcp_servers.web`, `tool_search` off and `-t web`.

D and S get byte-identical prompts, which quote GAIA's first two answer rules:

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
| cost | `n_llm` and `n_tool` (C counts tool-less `aux` requests too), output and uncached tokens, images, wall time |
| navigation | `n_nav`, `n_search`, `n_pages`, `gold_url_visited`, `zero_nav` (an answer with `n_nav = 0`), `serp_only` |
| flags | `answer_key_suspect` (a URL or title matching `simple[ _-]?qa`); truncated MCP output |

Per harness and arm, the report gives:

- verdicts by tier, with costs;
- zero-navigation answers and their accuracy;
- D and S accuracy, split by whether C ever got the question right;
- D vs S, paired by question as in e3, with a sign test and its minimum detectable difference.

## 9. Risks and open questions

Risks:

- Flash-Next's tool-result images are unproven; the G2b probe must pass before S runs.
- Long pages in D can hit Claude Code's MCP output cap.
- 24 questions support only descriptive conclusions.

For the user:

1. Keep AI Overviews (recommended: the real web, the same for both arms), or use `udm=14`, which a typed
   query drops?
2. Can S run overnight, with the Mac unlocked and unattended?
3. For S: the five tap tools (recommended), or cua-driver's tools verbatim?
4. Stable Chrome, stopping on an update (recommended), or Chrome for Testing now?
5. For rows 3538, 3352, 2849 and 69: aliases or swaps?

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
