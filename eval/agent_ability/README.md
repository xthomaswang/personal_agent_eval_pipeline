# harness-smoke

A quick, harness-agnostic smoke test for agent harnesses on a local model. It runs the same tasks,
with byte-identical prompts, through Claude Code (headless), Hermes Agent (one-shot) and localmelo
(direct mode) via personal-agent-bench's `bench.harness`, then checks each result with the upstream
benchmark's own checker.

## Suite

- **MCPMark filesystem, easy** (`eval-sys/mcpmark` @ `cd45b7f`), 10 tasks:
  `file_context/{file_splitting,pattern_matching,uppercase}`,
  `file_property/{largest_rename,txt_merging}`, `folder_structure/structure_analysis`,
  `legal_document/file_reorganize`, `papers/papers_counting`,
  `student_database/{duplicate_name,recommender_name}`.
- **Aider polyglot, Python** (`Aider-AI/polyglot-benchmark` @ `7e0611e`), 5 exercises:
  `transpose`, `proverb`, `pig-latin`, `list-ops`, `variable-length-quantity`.
  `--extended` adds `robot-name`, `grade-school`, `affine-cipher`, `scale-generator`, `wordy`,
  `phone-number` and `react`. `--task 'aider/<name>'` selects any of the 34 Python exercises.
- **Warm-up**, unscored, first for every harness: create `warmup.txt` containing `ready`.

The default run is 3 harnesses x (15 tasks + warm-up) = 48 runs, 45 scored, in harness-major order.
The MCP-only arm (`--mcp-arm`) runs the 10 MCPMark tasks on Claude Code and Hermes: 22 runs, 20 scored.

## Data

Everything lives under the data directory (`HARNESS_SMOKE_DATA`):

| Path | Content |
|---|---|
| `aider-polyglot-benchmark/7e0611e77b54/` | shallow fetch of exactly that commit |
| `mcpmark/cd45b7f57923/` | shallow fetch of exactly that commit |
| `mcpmark/filesystem-envs/` | the six start zips; `manifest.json` records URL, size, sha256, ETag and fetch time |

Fetch or re-check with `.venv/bin/python -I scripts/fetch_data.py`; it is idempotent. Every run
re-verifies the commits and the zip hashes before it starts.

Downloaded content is untrusted data. Only two things from it ever execute:

- MCPMark's `verify.py`, as `python -I -B verify.py` from the run directory (outside the workspace);
- the Aider unit tests, run by pytest in a throwaway check directory.

Zips are extracted with path checks: no absolute paths, no `..`, no symlinks, nothing outside the
category folder, and `__MACOSX/` skipped.

## Running

The runner executes in personal-agent-bench's environment, which provides `bench.harness`:

```sh
cd <personal-agent-bench checkout>
uv run python ../harness-smoke/run.py --dry-run      # plan, prompt hashes, preflight
uv run python ../harness-smoke/run.py                # the full built-in-tools suite
uv run python ../harness-smoke/run.py --mcp-arm      # Claude Code + Hermes, filesystem MCP server only
uv run python ../harness-smoke/run.py --harness hermes --task 'aider/*'
uv run python ../harness-smoke/run.py --list-tasks
```

Each task gets `bench.harness.run(..., timeout_s=600, max_steps=30, max_turns=40, thinking="off")`.
The flags `--timeout`, `--max-steps`, `--max-turns` and `--thinking` change these values, and
`--no-warmup` skips the warm-up. `--random-seed` has the proxy set a fresh random `seed` on every
request, for a server whose default seed is fixed (Recursive's rec engine). `--one-reasoning-field` has it keep one of
`reasoning_content` and `reasoning` on replayed assistant messages, which rec requires and Hermes
does not do on its own (`../personal-agent-bench/docs/proxy.md`).

`--watch-solution` counts the steps a harness takes after its solution already passed (Aider tasks
only; MCPMark tasks run unwatched).
- While the harness runs, a thread polls the solution file every 0.5 s and keeps each distinct
  content. It only reads the workspace.
- After the run, each version is checked by the same checker in a fresh workspace.
- summary.json and the run's record get `watch`:
  - the first passing version (`first_pass_t`);
  - the model requests that started after it (`planning_after_first_pass`, with tools;
    `aux_after_first_pass`, without). A run that answers right after the passing write counts 1:
    the request that carries the write's result;
  - the number of versions, and each version's check.
- See `smoke/watch.py` for the time base.

Preflight refuses to start (exit 2) unless all of these hold:

- the data verifies;
- the smoke `.venv` imports pytest;
- the harness binaries exist;
- `experiments/.gpu-lock` is held by this session (`holder: agent-benchmark ...`);
- the model server answers `GET /v1/models` (with the bearer token from `BENCH_UPSTREAM_TOKEN_CMD`, if
  that is set; see `../personal-agent-bench/docs/proxy.md`);
- for `--mcp-arm`, Node and the MCP server are installed.

The runner does not take the GPU lock itself: take it with
`../personal-agent-bench/scripts/gpu-lock.sh acquire "<purpose>" <eta-minutes>` before starting oMLX,
and release it after oMLX stops.

The MCP server (`@modelcontextprotocol/server-filesystem@2025.12.18`, the version MCPMark pins)
is installed per project with nvm's Node, which `mcp/.nvmrc` pins:

```sh
nvm install "$(cat mcp/.nvmrc)"
npm install --prefix mcp --ignore-scripts --no-audit --no-fund
```

## Outputs

Each run writes `runs/<YYYYmmdd-HHMMSS>[-mcp]/`:

- `results.json` and `results.md`. The table is harness x task with:
  - the verdict;
  - `wall_s`;
  - the proxy's `n_llm`, `n_tool` and `n_llm_aux`;
  - the harness's native step count (Claude Code `turns`, Hermes `api_calls`/`tool_use`, localmelo `steps`);
  - notes (tests modified, extra files, check detail).

  The header records the settings, the data commits, this suite's commit, the macOS build
  (`sw_vers`) and `xcodebuild -version`. Both files are rewritten after every run, so a partial run
  still leaves a table.
- `agent-env.sh`, the shell hook described below.
- `<harness>/<task>/`, which holds:
  - everything `bench.harness` writes (manifest, prompt, calls, proxy log, stdout and stderr, summary);
  - `check.log` and, for Aider, `check/`;
  - `workspace/`, the agent's final workspace;
  - with `--watch-solution`, `solution_versions/`: every version of the solution file
    (`<n>-<file>`), listed in `versions.jsonl`.

Verdicts:

| Verdict | When |
|---|---|
| `timeout` | the harness hit the wall-clock limit, even if the check passes |
| `harness_error` | `bench.harness.run` raised, or no main LLM call succeeded and the check failed |
| `pass` / `fail` | otherwise, from the checker |

For Aider, the checker:

- copies the solution files into a fresh directory;
- restores the pristine tests;
- runs `python -I -B -m pytest -q -p no:cacheprovider` with a 180 s limit.

Changed tests are flagged (`tests_modified`), never used.

Workspaces are rebuilt for every harness x task at the fixed path `ws/<task>`, so prompts are
byte-identical across harnesses. After the check, each workspace moves into its run directory.
`ws/.lock` stops two runs from sharing them.

## What the suite changes around bench.harness

`smoke/adapters.py` wraps `bench.harness.BUILDERS` for the length of one run and restores it
afterwards; `bench/harness.py` itself is untouched.

- **PATH.** Every harness gets the shims and the smoke `.venv` first on PATH. macOS login shells
  (Claude Code's Bash tool, Hermes's terminal) run `path_helper`, which puts `/usr/bin` back in
  front. Each harness therefore sources `agent-env.sh` through its own hook:
  - Claude Code through `CLAUDE_ENV_FILE`;
  - Hermes through `terminal.shell_init_files` in its per-run config.

  localmelo uses `/bin/sh -c` and keeps the inherited PATH.
- **pip shims.** The shims make `pip` and `pip3` fail, so an agent cannot install into the system
  Python.
- **MCP arm, Claude Code.** It runs with `--tools ""`, `--allowedTools "mcp__fs__*"`, one stdio
  server `fs` and `ENABLE_TOOL_SEARCH=false`.
- **MCP arm, Hermes.** It gets an enabled `mcp_servers.fs` entry, with every other server
  disabled, and runs with `-t fs`. localmelo has no MCP client and is left out of this arm.

## Self-tests

These run offline, with no model, no oMLX and no GPU:

```sh
cd <personal-agent-bench checkout>
uv run pytest ../harness-smoke/tests -q
```

| Test file | What it covers |
|---|---|
| `test_aider_checks.py` | for all 34 Python exercises, the stub fails and `.meta` reference passes; tampered tests, planted `conftest.py`, missing or symlinked solutions |
| `test_mcpmark_checks.py` | every untouched workspace fails `verify.py` and a scripted oracle passes; unsafe zip entries and hash mismatches are refused |
| `test_adapters.py` | builder wrappers and restoration; the installed MCP server over stdio; PATH resolution through Hermes's real terminal code and through Claude Code's Bash command layout |
| `test_runner_e2e.py` | the runner end to end through `bench.harness` with a fake harness and a fake upstream: pass, fail, timeout, tamper and crash verdicts; identical prompts; step counts; dry-run plans |

## E3: prompt-cache A/B of localmelo flag sets

`e3/` compares five localmelo flag sets (arms) on prompt-cache reuse, with the smoke runner's `run_one` and
`bench.harness` underneath, on twelve Aider exercises (sgf-parsing, book-store, zebra-puzzle, pov,
scale-generator, connect, transpose, bowling, paasio, wordy, zipper, react), 2 repeats each:

| Arm | Extra localmelo flags |
|---|---|
| S0 | none |
| C1 | `--window-eviction chunk` |
| B1 | `--window-eviction boundary` |
| B2 | `--window-eviction boundary --fixed-prefix on` |
| S2 | `--fixed-prefix on` |

Every run also passes `--output-format json` (localmelo's result line, with `attempts` and `n_aux`) and gets
the capability leg's budget: 30 proxy steps, 40 turns, 3600 s, thinking off, request and response bodies
logged. That is 120 runs. The order is task-major: for each (task, repeat) all five arms run back to back,
and the arm order is rotated by (task index + repeat), the task index counted from 0 and the repeat from 1,
so no arm is always first.

**localmelo build.** The arms run a separate clone pinned at `ce4a3a8` in `e3/vendor/localmelo`
(gitignored), made the way `memory/README.md` describes for the native probe and D11: `git clone
--no-hardlinks <localmelo's git dir> e3/vendor/localmelo`, `git checkout --detach ce4a3a8`, `git remote remove
origin`, then `UV_PYTHON_DOWNLOADS=never uv sync --frozen --all-extras` in the clone. The driver points
`bench.harness` at it (`manifest.LOCALMELO_REPO`) and refuses to start if it is at another commit or has
uncommitted changes.

```
cd eval
uv run python agent_ability/e3/run.py --run-dir <dir> --dry-run   # the schedule and every check; no model
uv run python agent_ability/e3/run.py --run-dir <dir>             # run; the same command resumes
uv run python agent_ability/e3/report.py <dir>                    # <dir>/report.md and report.json
```

The operator takes the GPU lock (`holder: agent-benchmark`) and starts the model server first; the driver
never takes or releases the lock, and refuses to start without it. `<dir>/e3.json` is the checkpoint
(`schedule.json` and `environment.json` sit next to it): finished runs are never re-run, and a run that was
started and left no record is recorded as `interrupted` rather than replayed. Before each run the driver
stops cleanly (exit code 3) when `<dir>/STOP` exists, when the lock has gone, or on battery below 30%.
`environment.json` has the macOS and Xcode builds, the localmelo commit and the model server's record at the
start and at the end of the last session. `--tasks`, `--arms` and `--repeats` select a subset.

The report has, per arm, the pass counts (per task and repeat), cache efficiency (cached / prompt tokens over
each run's model calls except its first, pooled and as the median of the per-run values), prompt, uncached
and output tokens (the output tokens are the `completion_tokens` of the same calls), steps and wall time,
reflection health (reflections, unreadable ones and their rate, found in the logged bodies), `attempts` and
`n_aux`. When S0 and another arm are both in the study, it also has a "Paired passes against S0" section: for
each other arm, the tasks both passed at least once, and on them the mean output tokens, uncached tokens,
steps and wall time of the passing runs for S0 and for the arm, with the arm/S0 ratio of output and of
uncached tokens. Its decision rule picks the arm with the highest cache efficiency
that loses no task S0 passed in every repeat and whose unreadable-reflection rate is at most 2 points above
S0's, ties going to the simpler arm; it lists every (task, arm) pair whose pass count differs from S0's for a
rerun. A reflection request is one whose last user message starts with localmelo's `REFLECTION_PROMPT`, or a
tool-less one that carries that text in its system message; tool-less requests without it are counted
apart. A reflection is unreadable when its reply has no JSON object with a `recommended_action` key; the
retry that follows it at once is not counted as another reflection.

A study has only five to eight reflections per arm, too few to compare unreadable rates, so `e3/replay.py`
sends each reflection of the chosen arms again, several times, to the model server, exactly as the recorder
proxy forwarded it (sampling, tools and chat-template settings included), and classifies every reply:
readable, a tool call instead of text, invalid JSON that names `recommended_action`, or no JSON. It needs the
same lock, appends to its output file and resumes from it, and stops on `STOP` next to that file:

```
uv run python agent_ability/e3/replay.py --out <dir>/replay.jsonl --arms S0,V64,O20 --repeats 6 \
    --localmelo <the studies' clone> <study dir> [<study dir> ...]
```

Tests: `uv run pytest agent_ability/tests/test_e3_run.py agent_ability/tests/test_e3_report.py
agent_ability/tests/test_e3_replay.py` (from `eval/`; fake runs only).

## Expected runtime

These figures are estimates from 18 tok/s decode and 13-16K tokens of harness prompt overhead per
call. They are not measured.

| Arm | Typical | Ceiling |
|---|---|---|
| Built-in tools | 1.5-2 h, up to about 2.5 h if many tasks use their full step budget | 8 h (48 x 600 s) |
| MCP-only | 30-45 min | 3.7 h |

## Limitations

- Claude Code's `CLAUDE_ENV_FILE` handling was read from the 2.1.302 binary and its command layout
  is emulated in a test; it has not been exercised with the real binary and a model.
- `verify.py` runs under the smoke `.venv`'s Python 3.13 rather than MCPMark's own environment.
  The ten easy verifiers use only the standard library.
- Hermes keeps its one-shot default of denying commands it flags as dangerous.
