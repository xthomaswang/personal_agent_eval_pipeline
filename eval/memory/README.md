# memory

Does an agent keep a correct, current and compact memory? Two parts:

- **T4, memory consolidation** (`t4/`, `pab/`): a synthetic 10-day timeline of messages, emails and
  calendar entries is fed one day at a time, one agent session per day. Afterwards a fixed reader answers
  60 probe questions from the memory store alone, so the score is about what the agent wrote, not about
  how well it retrieves. Spec: `t4/task.yaml`.
- **localmelo's native T4 probe and D11** (`native_t4_and_d11.md`): T4 with localmelo's own long-term
  memory instead of pab's tools, and a cross-task A/B of `--admit-on-outcome`. Designed; they run after
  localmelo's R6-lite merges.
- **R3, offline retrieval** (planned here; built in personal-agent-bench): recall@k and latency of
  localmelo's retrievers on recorded agent runs. No agent loop.

## localmelo for the native probe and D11

They run their own localmelo clone, `eval/memory/vendor/localmelo` (gitignored, no remote), so a run of
several hours never sees another session move a shared checkout. Pinned at **b7aa590** (adds digest task names and verified-check filtering #128 on top of R6-lite and content digests).
Made with `git clone --no-hardlinks <localmelo's git dir> eval/memory/vendor/localmelo`,
`git checkout --detach <sha>`, `git remote remove origin`, then
`UV_PYTHON_DOWNLOADS=never uv sync --frozen --all-extras` in the clone. The runners point bench.harness at
it with `BENCH_LOCALMELO_REPO` and use its `melo-retrieve` for the exports. A memory directory belongs to
one build (history.db v2 and long_term.db v1 are not read by older builds), so moving the pin starts new
memory directories.

## T4

```
t4/scenario.py         the story: people, records, the gold memory items, the 60 probes
t4/gen.py              python -m memory.t4.gen --seed 1  ->  t4/fixtures/v1/ (--check verifies)
t4/fixtures/v1/        days/day01..10.jsonl, facts.yaml, probes.yaml, drop_list.yaml
pab/                   the bench's MCP server (stdio): feed_day, memory_*, submit
t4/reader.py           the fixed reader (one call per probe, thinking off, temperature 0)
t4/oracle.py           accuracy per category, fabricated_source, source_precision, duplicate_ratio, memory_tokens
t4/native.py           resumable localmelo native trial, exports and both fixed-reader arms
t4/routing.py          Hippo decisions against version introductions and admitted-store state
t4/values.py           answer normalization and key-value extraction
```

**The timeline.** 46 story records over 2026-04-06..15 and 40 distractors placed by the seed (86 in
all). The story has updates (a rescheduled offsite and dentist, a corrected rent, a delayed delivery),
relative dates ("this Saturday", "two weeks from today"), cross-source links (an email and the calendar
entry for the same event), facts that only combine across days ("Jamie's sister" on day 1, her flight on
days 4 and 7), and near misses (Harbor Hall and Harbor Room; another flight number; a second concert
date). Probes, 12 per category:

| Category | What it needs |
|---|---|
| extraction | one fact from one record |
| cross_session | facts from records of different days, linked by a person, place or event |
| temporal | a relative date resolved against the record's date |
| knowledge_update | the latest value after a change |
| abstention | an answer the timeline never gives (the reader must say null) |

**Running one trial** (the runner that drives a harness over the 10 sessions is not built yet): each
session starts the harness with pab as its MCP server, `python -m memory.pab --run-dir <run> --session <n>`,
and the prompt in `t4/task.yaml`. Then:

```sh
uv run python -m memory.t4.reader --run-dir <run>     # needs oMLX, BENCH_MODEL and the GPU lock
uv run python -m memory.t4.oracle --run-dir <run>     # writes <run>/t4/oracle.json
```

**Checks that hold offline** (`uv run pytest memory/tests`): the fixtures regenerate byte for byte; the
gold store cites only records it can see, with source_precision 1 and no duplicates, on every day; a
scripted agent that writes the gold items through pab ends with exactly the gold store; citing a record
before it was fed is flagged.

**Reader checks on fixtures v1** (2026-10-26; Qwen3.8-27B oQ5 + MTP on oMLX v0.7.1rc1-1-g57620a3f,
thinking off, temperature 0; macOS 26B5101f, Xcode 27A9275; M5 Max 128 GB):

- Empty-memory filter, 3 reps: no answerable probe was answered right from an empty store, so none is
  dropped (`t4/fixtures/v1/drop_list.yaml`). The reader answered null to every probe.
- Gold store (the ceiling), 3 reps: 48/48 answerable and 12/12 abstention (every abstention a literal
  null), the same answers in every rep. The gold store is 1,415 tokens; a reader call takes about 2.9 s
  at 1,562 prompt tokens.
- Earlier ceiling runs found three faults, fixed in v1 before this result: "the next reading group"
  (ab12) depended on today's date, which the reader is not told; "which room at Harbor Hall" (ab11) was
  answered with the floor the store does give; and the oracle credited any unparsable answer to an
  abstention probe. ab04, ab07, ab08 and ab10 were also reworded so the question fits its answer type.

**Scoring rules.** Accuracy is counted per probe and reported per category. The empty-memory filter
drops answerable probes that the reader gets right from an empty store; abstention probes are never
dropped, since an empty store answers them right by construction, and are reported apart. The headline
is the accuracy over the four answerable categories.

### Native localmelo driver

`memory.t4.native` implements section 1 of `native_t4_and_d11.md`. From `eval/`, with the model server,
`BENCH_MODEL`, tokenizer (`BENCH_MODEL_PATH` or `--tokenizer`) and GPU lock arranged by the caller:

```sh
BENCH_LOCALMELO_REPO="$PWD/memory/vendor/localmelo" nice -n 19 uv run python -m memory.t4.native --run-dir <trial>
```

Each day has its own process, workspace, MCP config and recorder under `sessions/dayNN/`; all ten
processes share `<trial>/mem` and a temporary harness HOME at `<trial>/home`. Only pab's `feed_day` and
`submit` are exposed. The driver keeps localmelo's R5/R6 flags at their defaults, adds
`--reasoning compress --keep-decision-text --output-format json`, and uses 30 proxy calls, 900 s and thinking off. MCP is
launched with an isolated Python bootstrap that explicitly names the trusted `eval/` root, because
this project is not installed as a Python package.

Re-running the same command resumes `t4/native.json`. Completed sessions, including budget exhaustion,
are not replayed. A started session whose process was interrupted is recorded as interrupted (or
recovered from its completed harness summary), then the driver proceeds to the next day; it never
replays a partially-mutated memory. A changed build, model, fixture fingerprint or tokenizer requires
a fresh trial. Export/reader failures leave the checkpoint resumable without repeating ingestion.

After ingestion, `t4/export.jsonl` and `t4/decisions.jsonl` come from the pinned clone's `melo-retrieve`.
Each has an adjacent `.diagnostic.json` with the command's exit code, stderr and missing-store flag;
a missing long-term database produces empty exports and still proceeds through both readers. Other
export errors stop evaluation and remain resumable.
`t4/routing.json` reports decision-row/item verdicts, unscorable provenance and never-admitted versions.
`t4/store-everything.jsonl` retains every decision's text, including ignored duplicates; missing texts
are an error, not silently dropped. The native arm writes `reader/` and `t4/oracle.json`; the same
reader/oracle runs on the rebuilt store under `comparators/store-everything/`, using the original pab
events for fabricated-source checks. Per-session proxy model calls, wall time and admitted-by-signal
counts are saved in the checkpoint. Without a tokenizer, the oracle explicitly reports null tokens.

CPU-only checks (no oMLX required):

```sh
nice -n 19 uv run python -I -m pytest memory/tests/test_native_t4.py memory/tests/test_t4.py
nice -n 19 uv run python -I -m pytest memory/tests/test_native_t4_smoke.py
```

The smoke uses the suite's localmelo clone if available (otherwise skips), a loopback scripted model,
and fresh scratch/HOME directories. It drives pab through localmelo, quotes facts from actual tool
outputs in structured notes (no gold lookup), answers reader probes with scripted nulls, and handles
llm-retriever catalogs; its scores are **not model-quality measurements**.

The 2026-10-28 scripted smoke at localmelo `15c3cde` completed 10 sessions and 20 pab tool calls in one
memory directory. It exported 11 active entries (10 episodic reasoning notes plus 1 user-request digest)
and 30 decision rows; both 60-probe reader/oracle arms completed. **Provenance limitation:** all source
id lists were empty, even on the note quoting a record read by the preceding `feed_day`. All 30 rows
were therefore unscorable for routing, with 38 never-admitted gold versions; both source precisions
were 0. The bench does not invent missing source ids. The notes currently derive provenance from the
same step's tool result, while a factual note following `feed_day` accompanies `submit`, whose result
contains no records. Environment: macOS 27.2 build `26B123`, Xcode 27.1 build `27A9275`; CPU stub only.

## D11 runner

`memory.d11.run` implements section 2 of `native_t4_and_d11.md`: 120 direct-mode runs over the 12 Aider
exercises of `aider12-v0` (5 passes × 12 exercises × arms `off`/`on`). Real-model runs require the
shared GPU lock and the user's window (~6 h of oQ5 across both arms). From `eval/`:

```sh
BENCH_LOCALMELO_REPO="$PWD/memory/vendor/localmelo" nice -n 19 uv run python -m memory.d11.run --run-dir <trial>
```

Each pass permutes the 12 exercises with a seed equal to the pass number (identical for both arms);
each exercise runs in both arms back to back, with the first arm alternating by pass (`off` on odd
passes, `on` on even passes), and gets a fresh exercise copy before every run. Each arm shares one
memory directory (`<trial>/mem-off`, `<trial>/mem-on`), runs with `--admit-on-outcome <arm> --reasoning compress --output-format json`,
and uses `aider12-v0` budgets (30 proxy calls, 600 s, thinking off) and a temporary harness HOME at `<trial>/home`.

Outputs under `<trial>/`:

- `d11.json`: resumable checkpoint with the config, `localmelo_sha`, the 120-run schedule, and per-run records.
  Finished runs are skipped on resume; a `started` run with no `record.json` is recorded as `interrupted`
  (salvaging `summary.json` and `history.db` if present), never replayed against an already-mutated memory,
  left out of the D2 pairs, and listed in the report.
- `runs/<run_id>/`: per-run proxy logs, `summary.json`, `check.json`, `record.json` and final `workspace/`.
- `mem-off/{export,decisions}.jsonl` and `mem-on/{export,decisions}.jsonl`: active long-store entries and
  admission decisions exported with the pinned clone's `melo-retrieve`.
- `report.json` (and `results.json`) and `report.md`: per-arm and per-pass pass rate, `n_llm`, `wall_s`,
  entries admitted by signal (`admitted_by`), long-store entries written per run, recall hits split into
  `same_exercise` vs `cross_exercise` via `llm_calls.recall_json` `origin_task_ids`, and the D2 decision
  rule on passes 2–5 with paired McNemar and bootstrap 95% CIs.

CPU-only checks (no oMLX or GPU lock required):

```sh
nice -n 19 uv run python -I -m pytest memory/tests/test_d11.py
```

## Memory across chats (`across/`)

Does localmelo recall what you told it in another chat, and what does the recall cost? Four arms on one chat model:
**M2** (plus an embedding model; BM25 and cosine fused), **L1** (the chat model alone picks from BM25 candidates),
**B0** (BM25 only) and **N0** (memory off). The arms are the `ARMS` table in `across/run.py`, the `-X` settings each
passes to `melo serve` (`memory.retriever=embedding|llm|bm25`, `memory.across_chats=false`). The study is pinned to
localmelo **5f1498f4** (`--localmelo-sha` changes it); the preflight checks the sha, a clean tree, the settings
(`melo config get`) and that the endpoint lists both models.

Files: `data/tasks_v1.yaml` (43 fictional facts in EN and ZH, 60 questions: exact 20, paraphrase 15, crosslang 10,
superseded 5, lookalike 5, nothing to recall 5), `data.py` (validator), `score.py` (grading, hit@k), `client.py`
(console API), `server.py` (one `melo serve` per cell, model-server probes), `proxy.py` (timing proxy), `run.py`
(driver), `report.py`.

A **cell** is one arm in one repeat: a fresh HOME (own memory files, token and config; the real `~/.cache/localmelo`
is never touched), `melo serve` on a free port with stdin closed and no browser, 43 teaching messages each in a new
chat (an old value always before its replacement), then 60 questions each in a new chat, deleted once its answer is
read so a question never feeds the next. Arms rotate across repeats (4 arms x 3 repeats = 12 cells of 103 turns).
The last three teaching messages are long fillers on purpose: localmelo pins the newest "about" and "preferences"
quotes (600 characters) into every chat's system prompt, and the fillers fill that window so no asked fact is pinned.
The gateway config sets temperature 0 and thinking off (`--extra-body`). The model also has a `search_memory` tool;
`turn.memory` reports only the automatic recall.

**Two things the driver does to the chats.** When the chat model also tries to save a fact to a file, the tool call
asks for approval and the turn stays `running` (this build adds a "kept" line to a stated fact's message and tells the
model not to save it in a file; the report's first table, teaching turns that still asked for approval, checks that);
the driver denies every approval request (`POST /api/approvals/{id}`,
`deny`) and stops a teaching turn after 60 s (`--teach-timeout`; questions get `--turn-timeout`). Both are counted per
turn, per cell (`cell.json` `counts`) and per arm in the report. **Timing proxy:** the gateway's chat and embedding URLs
point at a stdlib proxy in the driver that forwards to `--base-url` unchanged and writes one line per request to
`cells/<cell>/proxy.jsonl` (path, model, stream, tools, `t_start`, `t_headers`, `t_first_byte`, `t_end`, status,
bytes). The report turns it into chat time to first byte per arm and, for M2, relates each turn's `embed_ms` and the
embeddings requests that overlap the chat request to it. A reply the gateway does not stream is sent whole, so its
first byte is about its end; the report says how many requests stream. Warm-up requests go straight to the server,
and only to the models the arm uses (the embedding model only in M2).

```sh
cd eval
uv run python -m memory.across.run --run-dir memory/runs/across-<name> --localmelo <checkout> \
    --base-url http://127.0.0.1:8500/v1 --dry-run     # plan and preflight, nothing started
# without --dry-run, plus --arms L1 --repeats 1 --questions Q01,Q21,Q36,Q46,Q56: one arm, once, five questions
# without --arms, --repeats and --questions: all 12 cells; the same command again resumes
uv run python -m memory.across.report memory/runs/across-<name>     # report.md and report.json
```

The operator holds the GPU lock (`holder: agent-benchmark`); the driver only reads it. It stops before the next turn
when `<run-dir>/STOP` exists, the lock is gone or the battery is below 20% (no cell starts below 40%). A stopped cell
continues in its own HOME on the next run; a cell that died in a turn is set aside (`cells/<cell>.aborted-N`) and run
again. The journal `cells/<cell>/turns.jsonl` keeps per turn the answer, `turn.memory` (`used`, `remembered`,
`refused`, `recall`; also `result.memory_recall`), the wall time from the POST to `turn.finished`, the time to the first
event, the gateway's own event clock and the turn's window in epoch seconds. `environment.json` has `sw_vers`,
`xcodebuild -version`, the localmelo sha and the model server's `/v1/models`. **Cost of the second model:** before each cell the driver asks the server to unload the embedding model if
`/v1/models/status` shows it loaded (`POST /v1/models/{id}/unload`, then waits up to `--unload-timeout` 60 s for "not
loaded"; a model that stays loaded, or a server that does not say, stops the run like a failed preflight; the chat
model is never unloaded; `--no-cold-start` skips it), so every cell starts without the embedding model and M2's warm-up
times its load. Around each cell (before, after warm-up, after teaching, after the cell) `cell.json` records the oMLX
process rss and the models it reports loaded (`GET /api/status` and `/v1/models/status`, at the server root); the report
says whether every cell started cold and shows per arm whether the embedding model was loaded and what the cell added.

The task data was checked offline against localmelo 5f1498f4's own code (no run): each message makes exactly one note,
only the five replacing facts supersede another, no look-alike pair does, only the fillers fit the pinned window, exact
questions pass the BM25 gate on their note alone, and paraphrase, cross-language and nothing-to-recall questions pass it
on nothing (so `llm` finds no candidates for a paraphrase with no shared word, by design).
Tests: `uv run pytest -q -p no:cacheprovider memory/tests/test_across_*.py`.
