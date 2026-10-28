# localmelo's native T4 probe and the cross-task protocol (D11)

Design, 2026-10-26. Both measure localmelo's own long-term memory (R6-lite: the typed long store,
Hippo's five actions, `--admit-on-outcome`). They run after R6-lite merges (localmelo's target: the week of 11/9, after R5
around 11/3). Until then nothing here runs; the T4 assets they need exist (`memory/t4`, `memory/pab`).

The plan they come from is localmelo's reasoning-memory plan (`.melo/reasoning_memory_plan.md`,
"How stability is shown", items 3 and 4, and D11). This file is the bench side: what runs, what is
measured, and the acceptance rules, fixed before any localmelo run.

## 1. Native T4 probe

**What it measures.** The write path: what localmelo's own memory keeps from T4's timeline, and how
cleanly. The fixed reader answers from the exported store, so retrieval is not part of the score.

**One trial** = a fresh memory directory, 10 sessions (one `melo` process per day), then the reader.
A memory directory belongs to one localmelo build: from R5 on, history.db is schema v2 and an older
build refuses it, so a pre-R5 baseline and an R5 candidate never share a directory.

- Session n (1..10): direct mode, with pab as the only MCP server:
  ```
  melo --mcp-only --mcp-config <cfg> --reasoning compress --keep-decision-text [R6-lite and R5 flags at their defaults] -- "<prompt>"
  cfg: {"mcpServers": {"pab": {"command": "<eval>/.venv/bin/python",
         "args": ["-m", "memory.pab", "--run-dir", "<trial>", "--session", "n", "--task", "t4-native"]}}}
  env: LOCALMELO_PERSIST_MEMORY=1, LOCALMELO_MEMORY_DIR=<trial>/mem (the same directory every day)
  ```
  `--reasoning compress` is explicit: R6-lite's typed store and write path require a non-legacy
  reasoning mode (protocol clarification, 2026-10-28); the R5/R6-specific flags stay at their defaults.
  `--task t4-native` offers `feed_day` and `submit` only, so localmelo's own memory does the
  remembering. The records arrive as tool output, never as a user message, as in real use; third-party
  mail in the user role would turn into profile facts through the user-statement markers.
- The prompt says "remember" and states no fact, so the run's `remember` candidates are admitted at the
  end of the run (the plan's strong write signal), and nothing in it can become a fact itself:
  > Today is {date}. Read today's records with the feed_day tool and remember what they establish about
  > Jamie's plans, people, places, dates and amounts, so that you can answer questions about them later.
  > Use absolute dates, and when something changes, remember the new value. Call the submit tool with
  > status "done" when you are finished.
- Budget per session as T4: 30 model calls (the proxy's `--max-steps`), 900 s, thinking off, oQ5,
  the bench's sampling. A session that runs out is recorded, and the next day is fed anyway.
- After day 10:
  ```
  melo-retrieve --export-store <trial>/mem   > <trial>/t4/export.jsonl
  melo-retrieve --export-decisions <trial>/mem > <trial>/t4/decisions.jsonl
  python -m memory.t4.reader --run-dir <trial> --store <trial>/t4/export.jsonl
  python -m memory.t4.oracle --run-dir <trial> --store <trial>/t4/export.jsonl
  ```
  `memory/t4/store.py` already reads an export whose entries carry `text` and
  `provenance.source_ids`. With an exported store, `fabricated_source` checks every cited id against
  the ids pab's `feed_day` returned during the trial. A missing `long_term.db` means an empty store:
  retain the export command's exit code and stderr as diagnostics, write empty exports, and still
  run the reader and oracle (older pins exit 2 on a missing file; T59 and later exit 0).

**Measures** (T4's own): probe accuracy per category over the 48 answerable probes, abstention apart;
`fabricated_source`; `source_precision`; `duplicate_ratio`; `memory_tokens`. Plus Hippo's routing
(below) and, per session, model calls, wall time, and entries admitted by signal.

**Hippo's routing, against expected actions from `facts.yaml`.** Each decision in `decisions.jsonl`
carries the candidate's `source_ids`. A record "introduces" the item version whose sources first
include it (`facts.yaml` items, versions by day). For a candidate:

| Its source_ids | Expected | Also accepted |
|---|---|---|
| only distractor records | not admitted, or IGNORE | NEW or RELATED (counted as `distractor_admitted`, not an error) |
| introduce an item's first version | NEW | RELATED |
| introduce a later version that replaces a value (a rescheduled date, a corrected amount, a new room) | UPDATE | CONFLICT (counted as `conflict_left_open`) |
| introduce a later version that only adds (a payment made, who brings snacks) | UPDATE or RELATED | NEW |
| only records the item's current version already cites | IGNORE | |

The `change: replaces` and `change: adds` labels in `scenario.py` are copied into `facts.yaml`: of
fixtures v1's 16 later versions, 12 replace a value and 4 only add. These are routing metadata; the
story, gold texts, records, probes and frozen drop list are unchanged.

A version is introduced by its sources dated on that version's own day. Earlier-dated sources are
support, not introductions of future versions. A row carries the latest version each of its ids
introduces. A supporting id of a version the row carries gets no separate verdict: for example, C1
(day 1) introduces `sister` v1, and C4 (day 8) introduces `family_dinner` v1, whose sources are C1+C4.
The row `{C1}` carries only `sister` v1; `{C1,C4}` carries only `family_dinner` v1. Ids stay literal;
neither routing nor the runner invents provenance from candidate text.

The comparison tracks the **admitted store**, in decision-log order, per item. A row with an entry
and action NEW, UPDATE, RELATED or MERGE advances that item's stored version; a refusal, IGNORE or
CONFLICT does not. With no stored version, NEW is expected (RELATED accepted), whatever the carried
version. With an older stored version, apply the replaces/adds rows above. With an equal or newer
stored version, expect IGNORE. Report a verdict for each (row, item), per-pair and per-row totals,
`conflict_left_open`, `distractor_admitted`, and `missed_versions`: timeline versions never admitted
by day 10. Missing source ids, ids absent from the fixtures and non-distractor records that map to
no item are unscorable, reported separately; ids pab never returned are also checked by the oracle.

**Comparators.** Memory off scores 0 on the answerable probes by T4's construction (the filter keeps
only probes an empty store cannot answer), so it is not a comparator.

1. **Fixed targets** (set now, from the reader's ceiling of 12/12 in every category on the gold store):
   per trial, at least 9/12 extraction, 8/12 knowledge_update, 6/12 cross_session, 6/12 temporal and
   10/12 abstention, as the mean of k = 3 trials; and no `fabricated_source` in any trial.
2. **Store everything**: every candidate Hippo received, admitted with no policy, rebuilt from
   `decisions.jsonl` (`--keep-decision-text` keeps each candidate's text there; off by default) and read by the same reader. It
   costs reader calls only, and it places Hippo on the frontier of accuracy against `memory_tokens`.
3. **Hermes on `pab.memory_*`**: T4's own arm (`memory/t4/task.yaml`), same timeline, model and budget.
   It needs T4's runner, which is built before the native probe runs.

**Acceptance** (all must hold): the fixed targets; against store-everything, at most 2 probes lower on
extraction and on cross_session (means of 3 trials), at least 1 probe higher on knowledge_update, fewer
`memory_tokens`, and a lower `duplicate_ratio`; no `fabricated_source`.

**Cost.** About 1-2 h of oQ5 per arm at k = 3 (10 sessions of a few calls each, thinking off), plus
the reader: 2.9 s per probe on the 1.4K-token gold store, about 3 min per store. oMLX reported no cached
prompt tokens on these calls although every probe shares the store prefix, so a larger store costs its
full prefill on every probe: a 15K-token store-everything export would take about 15-20 s per probe, or
15-20 min per store. Why the prefix cache misses is to be checked before the probe runs.

## 2. D11: the cross-task protocol

**Question.** Does admitting the final attempt digest at the end of a run (`--admit-on-outcome on`)
help later runs, at bounded memory growth? D2 makes `on` the default only if it does.

**Tasks.** The 12 Aider exercises of `aider12-v0`: transpose, proverb, pig-latin, list-ops,
variable-length-quantity, robot-name, grade-school, affine-cipher, scale-generator, wordy, phone-number,
react. Direct mode, localmelo only; each run gets a fresh copy of the exercise, so only memory carries
over.

**Design.**
- Two arms, `on` and `off`, each with one memory directory for all its runs
  (`LOCALMELO_MEMORY_DIR=<d11>/mem-<arm>` through `bench.harness.run(extra_env=...)`, and
  `--admit-on-outcome <arm>` through `harness_args`).
- 5 passes over the 12 exercises. Within a pass, the order is a permutation seeded by the pass number,
  the same for both arms. Each exercise runs in both arms back to back, and the arm that goes first
  alternates by pass, so drift falls on both. 120 runs in all.
- Pass 1 starts from empty memories in both arms and is the baseline; passes 2-5 are the comparison
  (48 pairs per arm).
- Setting as `aider12-v0`: oQ5, thinking off, the bench's sampling, 600 s and 30 model calls per run;
  R5's recall at its merged defaults.

**Measures**, per arm and pass: pass rate; model calls (`n_llm`, from the proxy) and `wall_s`; entries
admitted, by signal (`admitted_by` in the decision log); long-store entries written per run; recall
hits split into same-exercise and cross-exercise, from R5's recall records (each pass is a new task, and
`exclude_task` does not hide earlier passes of the same exercise, so same-exercise hits are reported
apart; R5/R6-lite's recall records carry each recalled entry's origin task id for this). bench.harness's per-run `history.db` step count reads `<run>/mem`, which the shared memory
directory replaces. Step counts come from the proxy, or from the shared `history.db`: R1's
`llm_calls` rows carry the task id, so a run's model calls are its task's rows.

**Decision rule** (D2), on passes 2-5, all must hold for `on`:
- pass rate no more than 10 points below `off` (non-inferiority);
- fewer model calls or less wall time than `off` (medians over the paired runs);
- bounded growth: at most 3 long-store entries written per run on average, and no run writes more
  than 8.

Paired tests over the 48 (pass, exercise) pairs: McNemar for pass/fail, a bootstrap CI for the
differences in model calls and wall time. These are reported; the rule above decides.

**Cost.** About 3 h of oQ5 per arm, 6 h in all: one overnight window under the GPU lock.

**Not in this protocol.** A session variant, in which staged candidates can be rehearsed across turns,
needs a session entry point the bench can drive: the gateway's session API (U3: create a session, post a
turn, SSE events). U2's REPL needs a pty, so it is not one. The bench drives `melo "task"` in direct mode
today.

## 3. To build when R6-lite merges

- `memory/t4/native.py`: the 10-session driver above (bench.harness.run per day, pab config,
  shared memory directory), then export, reader and oracle; and the store-everything rebuild.
- The routing check: labels on `scenario.py`'s later versions, and the comparison with
  `decisions.jsonl`.
- `memory/d11/run.py`: the 120-run schedule, through agent_ability's task definitions and
  bench.harness.run.
- T4's runner for Hermes on `pab.memory_*` (comparator 3).
