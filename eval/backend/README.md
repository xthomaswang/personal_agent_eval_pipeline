# backend (agent-model-eval)

This pipeline evaluates one local model as an agent backend on a Mac, with one command, and
puts it in a table next to the reference model (Qwen3.8-27B oQ5 + MTP). Each run has two legs.
The speed leg measures what an agent waits for. The capability leg measures whether agents
built on the model get work done, using three agent harnesses. [PROTOCOL.md](PROTOCOL.md) lists
the fixed setting, and [DESIGN.md](DESIGN.md) explains why.

```bash
uv sync                                  # Python 3.12 venv with modeleval + the copied agentbench harness
uv sync --project envs/capability        # Python 3.13: the capability runner and the recorder proxy
uv sync --project envs/agent-tools       # pytest only: the agents' PATH and the checkers
uv run modeleval fixtures                # once: fixtures/generated (gitignored), incl. the 110K M3 prompt; CPU only
uv run modeleval selftest                # offline: tests, path and hygiene checks (no server, no GPU, no lock)
uv run modeleval models                  # the registry and what each model has
uv run modeleval plan <model>            # units, time estimates, what is already done
uv run modeleval run <model>             # both legs; see "Running" below
uv run modeleval report                  # rebuild RESULTS.md
uv run modeleval publish                 # the results page's rows under docs/benchmark (see below)
```

`<model>` is a registry key from `models.toml` (for example `nanbeige4-3-33b-oq5`) or a served
id.

The fixtures are sized in the reference model's (Qwen3.8) tokens. At startup `modeleval` sets
`$AGENTBENCH_TOKENIZER_DIR`, when it is unset, to the reference checkpoint (`models.toml` `path`, with the
config's `[paths]`), if that directory has the tokenizer files; otherwise the harness looks for an HF snapshot
of `Qwen/Qwen3.8-27B` under `$HF_HOME`. `modeleval fixtures` builds into a temporary directory, compares the
build with the committed `fixtures/manifest.json` (every file's sha256), and only then replaces
`fixtures/generated`; the committed fixture files are never written. It prints the tokenizer directory it used.

## On another Mac

Where things are comes from `config.defaults.toml` (example defaults under `~/mlsys`). Put
the keys that differ, in their tables, into `config.toml` next to it (gitignored); it overrides the
defaults key by key, and `uv run modeleval selftest` prints what it resolved. `pins.toml` and
`models.toml` refer to these paths as `${name}` (for example `path = "${models}/oq/<served id>"`).

```toml
# config.toml
[paths]
mlsys = "~/work/mlsys"            # moves every default that hangs off it
datasets = "~/data/datasets"
[lock]
mode = "off"                      # no other sessions share this GPU
```

- `[paths]`: the oMLX checkout and base path, the model directories, the agent workspace root (part of
  every task prompt, so another value is another harness epoch), the agent configs, Claude Code's
  versions directory, the task data, the HF cache and the GPU lock file.
- `[lock] mode`: the GPU-lock protocol with other sessions (lock file, `LOCK-NOTICE` and its wait, lock
  purposes). `auto` (default) turns it on when the lock file's directory exists; `on`/`off` force it.
- `[machine] id`: the results path's machine level; empty derives it from the hardware
  (`<chip>-<GPU cores>c-<RAM>gb`, for example `m5max-40c-128gb`).

The quiet thresholds were measured on the owner's machine; PROTOCOL.md says how to calibrate them for
another one. A new machine needs its own reference run: tables never mix machines.

## Adding a model

1. Add a `[models.<key>]` table to `models.toml`. Fill in: the served id (the directory name oMLX
   sees), the checkpoint path, the HF source with an exact 40-hex revision, the quantization,
   `spec` (`mtp` or `none`), `vision`, the model's recommended agent sampling, the proxy preset,
   and `server_log_expect`. `server_log_expect` holds regexes that must match the server log
   after the warm-up and prove that the intended speculative or offload mode loaded. The
   registry refuses a per-model `max_tokens`, a second reference and duplicate served ids.
2. Prepare the checkpoint:
   - `uv run modeleval download <model>` runs `hf download <repo> --revision <sha>` into
     `$HF_HOME`. It takes no lock. It is refused while the lock is held for anything not marked
     `untimed`, because a download disturbs timed units; download during a capability leg instead.
   - `uv run modeleval convert <model>` runs `scripts/oq_convert.py` with the oMLX interpreter.
     It needs the GPU lock (purpose `accuracy only (untimed), convert`), no server on 8500 or
     8703, and free disk for the output + 20 GB. It applies the registry's tokenizer fixes to
     the output copy, then registers the model.
   - `uv run modeleval register <model>` creates the `${models_dir}/<served
     id>` link. A checkpoint that another session owns is only linked, never written.
3. Run `uv run modeleval plan <model>`, then `uv run modeleval run <model>`.

## Running

```
uv run modeleval run <model> [--legs speed,capability] [--capability gate|full|full-force] [--unverified]
                             [--resume <run-id>] [--notice-wait 600] [--settle 600] [--dry-run]
```

- `--dry-run` prints the plan. It takes no lock, starts no server and writes no file.
- `--capability gate` runs only the gate set. The default `full` runs the gate set, compares it
  with the reference's finished gate set of the same harness epoch (DESIGN §6.6), and runs the
  ranking set unless the gate decision is "worse" (`GATE-DECISION` says which). The reference
  model, or a model without a reference gate set in its epoch, always goes on to the ranking set.
  `full-force` runs the ranking set in any case. A `--resume` whose ranking set has started goes
  on without deciding again.
- `--unverified` is a mechanics run. Everything goes under `results/_unverified/` (gitignored):
  the run never waits for quiet, every unit runs once, and verdicts are recorded but never
  retried. The lock is taken with an untimed purpose.
- `--resume <run-id>` continues a stopped or interrupted run. It skips speed units with a kept
  result (clean, or minor flags only) and capability pairs with a valid verdict, stops the orphans of a crashed attempt
  (from the run's pidfile), and numbers its launch after the previous ones. It is refused while
  another orchestrator of the same run id lives (an flock on `runs/<model>/<run-id>/orchestrator.pid`),
  when the macOS or Xcode build changed since the speed leg ran (start a new run, or resume with
  `--legs capability` when only that leg is left), and when its capability rows are from another
  harness epoch. A run that is resumed after its speed leg finished takes an untimed lock.
- To make a running run pause at its next boundary while it keeps the lock, create
  `runs/hold`. To let it go on, delete the file.

The orchestrator never messages other sessions. It prints machine-readable progress lines for the
driving session, which relays them to the peers.

### Progress lines

Each progress line is `<HH:MM:SS> <TAG> key=value ...` on stdout.

| Tag | Meaning |
|---|---|
| `LOCK-NOTICE` | The run will take the GPU lock at `start` (after `--notice-wait` seconds). Timed work ends around `eta` and lasts `timed_minutes`; `purpose` is the lock purpose. Relay it to the peers |
| `UNIT-WAIT` | A pre-unit check (or `unit=lock`, or `unit=launch`) is waiting for quiet; `reason` says why. A repeat appears only when the reason changes |
| `UNIT-START` / `UNIT-DONE` | One attempt of a timed unit, with its launch number. `verdict` is `clean`, `minor` (only minor disturbance flags; kept with them, not repeated, DESIGN §5.4), `disturbed` (a major disturbance; repeated after the next passing check), `incomplete` (a missing latency sample or an error; repeated the same way) or `invalid` (a cold request hit the cache; not repeated). `reason` names the flags. After 3 attempts without a kept one the unit is `disturbed x3`, `incomplete x3` or `disturbed/incomplete x3` |
| `QUIET-WINDOW-END` | Timed work is over, and the lock purpose now says `accuracy only (untimed)`. Peers may resume light work |
| `TASK-START` / `TASK-DONE` | One capability task (printed by the capability runner); `memory_kills` counts the processes the memory watchdog killed during it (DESIGN §6.1) |
| `GATE-DECISION` | After the gate set: `decision` (DESIGN §6.6 against the reference, or `-` with `why`), `action` (`ranking` or `skip`), wins, losses, the reference run and the number of excluded pairs |
| `BATTERY-PAUSE` / `BATTERY-READY` / `RESUMED` | The battery rule: the server is stopped and the lock released; the battery reached 80 %; the model is relaunched and warmed up |
| `HOLD` / `HOLD-RELEASED` | `runs/hold` exists; the run waits at a boundary and keeps the lock |
| `ERROR` | Something to read: the reason a leg stopped, a stale child that was killed, a lost server that is relaunched, a restarted quiet monitor, or a cleanup problem |
| `RUN-DONE` | Final line: `status`, exit `code`, `run_id`, and whether `--resume` makes sense |

Every line also goes to `runs/<model>/<run-id>/progress.log`. If the reader of stdout dies (a
closed terminal, a killed `| tee`), the run points stdout at /dev/null and still stops its children
and releases the lock.

### Exit codes

| Code | Status | Meaning |
|---|---|---|
| 0 | `complete` | Both legs finished and every speed unit has a kept result (clean, or kept with minor flags) |
| 1 | `error` | An unexpected error (see `ERROR`) |
| 3 | `gate-fail` | The functional block failed on a live server (G0, G1, G3 tool use or G4); nothing timed ran. A server that died or failed in transport during the block is relaunched and the block repeated once; a second loss is `stopped` |
| 4 | `stopped` / `incomplete` | The run can be resumed. For example, the machine never settled, a unit stayed disturbed (major), incomplete or invalid, the capability runner yielded on epoch drift (`ERROR` names the reason from its `yield.json`), the server was lost repeatedly, or the lock could not be retaken after a battery pause |
| 5 | `refused` | Nothing was measured: the lock was held elsewhere, port 8500 was up, the model is not prepared or registered, less than 148 GB is free under `runs/` (the SSD cache's 128 GB cap + 20 GB), the generated fixtures are missing (speed leg), the runner interpreter or the harness epoch is not ready (capability leg), or a resume would mix builds or epochs. All but the lock are checked before `LOCK-NOTICE`; `plan` shows them as `ready` lines, and lists SSD caches that crashed runs left behind |
| 130 | `interrupted` | A signal arrived. Children were stopped and the lock released after port 8703 was free |

## Where things go

A verified run goes to `docs/benchmark/<machine>/<family>/<quant>/backend/<run-id>/` at the repository root
(the results contract in `docs/benchmark/README.md`); `<family>` and `<quant>` are the registry's `family`
and `quant_dir`, `<machine>` is derived from the hardware (for example `m5max-40c-128gb`). Below it is
written `results/<model>/<run-id>/`. After a run:

```
uv run modeleval report                  # RESULTS.md (this directory): the protocol tables
uv run modeleval publish [<model> [<run-id>]]   # summary.json rows for the results page
python3 scripts/build_docs.py            # at the repository root: writes docs/index.html (the results page)
```

`publish` writes `summary.json` next to the run's `manifest.json` (suite `backend`, protocol `speed-1`)
and `docs/benchmark/<machine>/<family>/<quant>/agent_ability/<run-id>/summary.json` (protocol
`capability-N` of the run's harness epoch, `CAP_PROTOCOLS` in `src/modeleval/publish.py`), from the numbers RESULTS.md
shows. A run's write-up is its directory's `README.md`.

```
results/<model>/<run-id>/       committed (after scripts/redact.py --check passes in the pre-commit hook)
  manifest.json                 settings, launches, memory, machine and repo SHAs; ~ paths only
  speed/units.json              verdict per unit and attempt, metrics, memory peaks
  speed/<unit>-<n>.jsonl        raw files of the kept attempts
  speed/monitor.tsv, memory.tsv quiet monitor (5 s) and memory sampler (2 s)
  capability/results.json       per (harness, task) verdicts, exported from the allowlist
results/_unverified/            in this directory, gitignored: attempts with a major disturbance, --unverified runs
runs/<model>/<run-id>/          gitignored: server.log, pidfile, orchestrator.pid, lock.log, progress.log,
                                speed-stdout.log (the harness's tables), ssd-cache/ (deleted at the end of
                                every invocation, whatever its status), capability spec, state and raw runs
RESULTS.md                      generated by `modeleval report`
```

A run that is refused before its first launch writes only `runs/<model>/<run-id>/manifest.json`.
The oMLX base path is config `[paths] omlx_base` (by default `${omlx_repo}/.modeleval-home`), and the pipeline
rewrites its `settings.json` and `model_settings.json` before every launch.

## Rules this pipeline keeps

- **GPU lock** (when the protocol is on, config `[lock]`). Model servers and conversions run only
  under the shared GPU lock (`scripts/gpu_lock.sh`). The lock records the run id, and a run releases
  only its own lock, and only after nothing listens on 8703. Ports 8500 and 8600-8699 belong to the
  agent-benchmark session and are never used.
- **Peer trees are read only.** This covers the repository's `eval/bench` and `eval/agent_ability`, `agent/`,
  `serving/` (except `.modeleval-home`), `~/.omlx*` and the model directories. The only git
  commands run on them are `rev-parse`, `--no-optional-locks status --porcelain`, `diff-index
  --quiet HEAD`, to create a harness epoch's snapshot in `vendor/`, `archive`, and, to check a declared
  epoch equivalence (DESIGN §6.5), `diff` between two tree objects and `cat-file blob`.
  `PYTHONDONTWRITEBYTECODE=1` and `GIT_OPTIONAL_LOCKS=0` are exported for every child.
- **Tests are offline.** They use fake `omlx`, lock, quiet-monitor, speed and capability
  processes on ephemeral ports in temporary directories; `tests/integration` runs the real
  orchestrator with the real speed module against agentbench's mock server launched as `omlx`.
  Run them with `uv run pytest tests/core -q`, or the whole suite with `uv run modeleval selftest`
  (`cd harness && uv run pytest -q` for the copied harness; its fixture and template tests need
  `$AGENTBENCH_TOKENIZER_DIR` set to the reference checkpoint, and skip without it).
