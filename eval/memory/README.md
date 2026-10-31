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

### Memory bench v2: in-chat kinds (`data/tasks_v2.yaml`)

v2 adds kinds that need a chat of their own; `--tasks memory/across/data/tasks_v2.yaml` selects the v2 protocol
(`across-run-v3`; v1 runs keep their protocol and identity). The file holds the two kinds built so far, 20 items each,
about a third Chinese. The `KINDS` table in `across/data_v2.py` says, per kind, which arms run it and what the
working window must look like at the question; a cell runs only the items whose kind lists its arm.

| kind | what it asks | arms (window rule at the question) |
|---|---|---|
| `longrange` | one chat: an opener, the fact at turn 2 or 3, 40 filler turns, then the question in the same chat. At least 14 of 20 facts are details of the conversation the note rules should not save (`note_expected: false`: "the draft I'm editing is called river.md"); the rest are note-worthy facts as a control. | N0 `out`, M2 `out`, FC `in` |
| `followup` | two facts, each taught in its own chat; then one chat with Q1 ("Who is my advisor?") and a pronoun follow-up Q2 ("And where is her office?"), deleted afterwards (unless `--keep-ask-chats`). Q1 and Q2 are scored apart. | M2, B0, N0 |
| `recall` | one question in a new chat, nothing taught: what memory holds when the cell starts (a seeded HOME and a memory import, below) is all there is. `category` is `habit`, `preference`, `project`, `abstain` (no gold: the reply must say it does not know) or `harm` (an unrelated question whose reply must not bring in remembered trivia, its `reject` strings); `workspace` puts the chat in that folder under the cell's HOME. The shipped file has none; private task files hold them. | M2, B0, N0 |

The filler is 40 turns per chat (22 tool turns that read a file or list the files, 18 plain ones), each asking for a
one-sentence answer. A tool turn is 4 messages when the model calls its tool (user, call, result, answer) and 2 when it
does not, so 124 messages follow the fact turn at best and 80 at worst. The loader enforces a floor on the worst case:
counted at 2 messages per turn, at least the window cap plus 6 (64 + 6 = 70, so 35 turns) must follow the fact; the
design aim with the tools is 110. The driver writes the small fixture files of the task file (each at most 300
characters) into `work/<item>/` for each long chat, so the read-tier tools the gateway offers (`file_read` and
`list_files`, neither of which asks for approval) have something to read; no gold string occurs in any filler message
or fixture file (the loader checks). The report's tools table shows what the chats called. A full-context chat stays
small: the first runs' biggest request was 18.7 KB at 30 filler turns, about 23 KB expected at 40, against the
100,000-character prompt cap (the loader's estimate for a chat is about 23,000 characters, limit 60,000).

**Why the floor counts 2 messages a turn.** The first real runs (`bench2-c1-*`: nine cells, 30 filler turns of which 16
were tool turns, 92 messages after the fact if every tool is called) went invalid for some items with "only 60
messages between the end of the fact turn and the question (cap 64)": the model answers a tool-meant filler turn without
calling the tool. Measured over the finished tool-meant filler turns of those runs, `tools` empty in the journal:
366 of 2,819 (13.0%); by arm M2 71 of 952 (7.5%), N0 89 of 603 (14.8%), FC 114 of 624 (18.3%), B0 92 of 640 (14.4%);
by tool `file_read` 152 of 1,948 (7.8%) and `list_files` 214 of 871 (24.6%). (61 tool-meant turns that did not finish
are left out.) Every such turn is 2 messages, not 4, so a 30-turn filler could fall to 60 messages. The set was
lengthened to 40 turns (the items, facts and asks did not change; 6 fixture files, 6 tool turns and 4 plain turns per
language were added). The task file's sha256 therefore changed: **fdf17a4d680ef1f3a1a882dcf9f1a868b8975fec0cb399acee9c0c3262fb8bfa**
(30-turn filler, the `bench2-c1-*` runs) is now **ee5fd0609503f9b556638511be8aee1386ed9859d39c6498d7788e00454b2e17**
(40-turn filler); a run made with one cannot be resumed or compared by `--compare` as the same task file.
Turns of a long chat, the question included, stop after `--long-timeout` (120 s); teaching turns keep 60 s and the
questions of follow-up chats 900 s.

**Arms.** FC (full context) is N0 with a 256-message window (the maximum) and no embedding model; the settings that
mean "memory off" are `MEMORY_OFF_X` in `across/run.py`, one place for N0 and FC. `melo serve -X agent.window_messages=N`
exits ("only for direct mode"): a gateway session takes its cap from `[agent] window_messages` of config.toml, so an
arm carries config entries (`Arm.config`; `"config": {"agent.window_messages": 256}` in an arms file) that the driver
writes into the cell's config.toml. The plan line shows them (`[config.toml: agent.window_messages=256]`) and so do
`across.json`, `cell.json` and the report's arm table. N0 and FC run in-chat kinds only (FC is not run on v1
questions). An arms file (`--arms-file`) may give an arm a `base`, the arm whose row in each kind's table it
follows: `{"M2-on": {"x": ["memory.retriever=embedding", "memory.context_query=on"], "embedding": true, "base":
"M2"}}` runs the kinds M2 runs, with M2's window rule.

**Seeded HOME and memory import.** An arm in an arms file may carry `"seed": "<folder>"` (relative to the arms file)
and `"import": true`. Before a new cell's gateway first starts, the driver copies the seed folder into the cell's HOME
at the same relative paths (`seed_home` in `across/run.py`). A `seed.json` at the folder's top maps placeholder folder
names to workspaces, `{"workspaces": {"@app": "Developer/app"}}`: a path part `@app` becomes the name a coding
assistant gives the notes folder of `<HOME>/Developer/app` under `.claude/projects/` (the absolute path with every character that is not a letter or digit
turned into `-`), and that workspace folder is created, so `.claude/projects/@app/memory/x.md` lands where a reader
of those notes looks. With `import`, the driver then calls `GET /api/memory/import/scan` and
`POST /api/memory/import` with every source the scan found, once per cell, before the first question; the scan and the
answer go to the cell's `import.json`, a summary (sources, lines kept, notes written, skipped, times, partial scan,
error) to `cell.json` and the report's import table. A failed import is recorded, not raised. The seed's fingerprint
and the import switch are part of the study's identity. The gateway never sees `CLAUDE_CONFIG_DIR`, `CODEX_HOME` or
`HERMES_HOME`, so an import reads only the seeded HOME. The recall report adds accuracy per category and, per
question, the sources of the used items.

**Valid items.** A long-range item counts for an arm only when, at the question, the fact turn is out of the window
(N0, M2) or still in it (FC). The check reads the cell's `gateway/sessions.db` after the cell ends, read-only
(`file:...?immutable=1`). Its `windows` table keeps one row per chat, the working window as the chat's newest turn
left it, so there is no row for "the turn before the question"; the row of the question's turn is used. A message
in that window was in it for every model call of the turn; a message that is absent was out of the window when the
question arrived if at least the cap (64 for N0 and M2) messages were added between the end of the fact turn and the
question (the window then held the newest 64 or fewer, and the fact is older than all of them), counted from the
stored `events`. It does not matter where in the chat the fact turn began. Otherwise the item is invalid for that arm
("the fact may still have been in the window"). FC items must also keep every request of the chat under 100,000 bytes
(localmelo's prompt cap is 100,000 characters; `req_bytes` of `proxy.jsonl`, mapped to turns by time). An item whose
fact or teaching turn did not finish, or whose cell has not finished, is invalid too. Invalid items are listed in
the report with the reason and left out of every table.

**Report** (`report.py` sends a v2 run to `report_v2.py`): accuracy per kind and arm with Wilson 95% intervals (long
range also by `note_expected`; follow-up as Q1, Q2 and Q2 given Q1 right); the class of every miss -- `not saved`
(no note from the chat that said the fact holds the gold string in `records.db`, `long_term.session_id`), `not
recalled` (saved, the chat's item is not in `turn.memory.used`), `recalled but not used`, and for N0 and FC `out of
window` or `in window but not used`; where right long-range answers came from (a used note, the chat's own turn --
a used item with source `this chat, <day>`, or the turn still in the window -- or unknown; M2's legacy step
summaries send no event, so an unexplained M2 answer is marked as possibly one); notes made against `note_expected`;
the tools the long chats called; exact McNemar between the arms of each kind; turn wall, recall, first-event time,
request bytes and recall-block characters (`recall.block_chars` when the build reports it, otherwise an estimate from
the used items, labelled ESTIMATE); and `recall.context` per question. `report ... --compare <other run dir> [arm]`
pairs the same asks of the same arm in two runs of one task file (two localmelo builds) with McNemar per kind and the
asks whose verdict differs.

```sh
cd eval
# smoke: two items per kind (L01 and L15 are long-range, P01 and P14 follow-up), all four arms, plan and preflight first
uv run python -m memory.across.run --run-dir memory/runs/bench2-smoke --localmelo <checkout> --localmelo-sha <sha> \
    --base-url http://127.0.0.1:8500/v1 --tasks memory/across/data/tasks_v2.yaml --arms M2,B0,N0,FC \
    --questions L01,L15,P01,P14 --dry-run
# the same without --dry-run runs it; with --repeats 1 (the default for v2) it is one cell per arm
# the full set: 40 items, 1 repeat, the same four arms (a second command resumes)
uv run python -m memory.across.run --run-dir memory/runs/bench2-<name> --localmelo <checkout> --localmelo-sha <sha> \
    --base-url http://127.0.0.1:8500/v1 --tasks memory/across/data/tasks_v2.yaml --arms M2,B0,N0,FC
uv run python -m memory.across.report memory/runs/bench2-<name> [--compare memory/runs/bench2-<other> M2]
```

GPU time on the bench's chat model: the follow-up kind is about 3 minutes per arm (8 teaching and 40 question turns of
a few seconds). The long-range kind is now about 860 turns per arm (20 chats of 43 turns). The first runs, at 667 turns
(30-turn filler), took M2 57-64 minutes (the embedding model adds a request per turn), N0 12 and FC 9-10 minutes; scaled
by 1.3 that is about 80 minutes for M2 and 12-16 for N0 and FC. M2 and N0 run both kinds, FC only long range, B0 only
follow-up, so the full set is about 2 hours at one repeat.
The operator holds the GPU lock as for v1; the driver never starts a model server.
