# Module contracts (internal)

The interfaces the modules of DESIGN.md §10 agree on. They were first written so the modules could be
built in parallel; since the integration on 2026-10-26 they describe the code as it is. Keep this file
in sync with the code; DESIGN.md has the why. `tests/integration/` checks the seams end to end.

## Paths and constants (`src/modeleval/paths.py`, owned by core)

```python
REPO = Path(__file__).resolve().parents[2]            # eval/backend/
MONOREPO                                              # the nearest parent with docs/benchmark
CONFIG = load_config((REPO / "config.defaults.toml", REPO / "config.toml"))   # tables, merged key by key
def expand(value, base=REPO) -> Path                  # ${name} of CONFIG["paths"], ~, relative to base
class ConfigError(ValueError)
MLSYS, OMLX_REPO, BASE, MODELS, MODELS_DIR, WS_ROOT, GPU_LOCK   # CONFIG [paths] mlsys, omlx_repo, omlx_base,
                                                      # models, models_dir, ws_root, gpu_lock
OMLX_PY = OMLX_REPO / ".venv/bin/python"; OMLX_EXE = OMLX_REPO / ".venv/bin/omlx"
HF_HOME = $HF_HOME or CONFIG [paths] hf_home
PORT = 8703; BASE_URL = "http://127.0.0.1:8703/v1"
RESULTS = MONOREPO / "docs/benchmark"; SUITE = "backend"
UNVERIFIED = REPO / "results/_unverified"; RUNS = REPO / "runs"; VENDOR = REPO / "vendor"
FIXTURES = REPO / "fixtures"; SCRIPTS = REPO / "scripts"; HOLD_FILE = RUNS / "hold"
FOREIGN_PORTS = (8500,) + tuple(range(8600, 8700))    # other sessions' servers: never used here
PROTOCOL_VERSION = 1
def machine_id() -> str                               # $MODELEVAL_MACHINE, CONFIG [machine] id, or <chip>-<cores>c-<ram>gb
def results_dir(family, quant_dir, suite=SUITE, machine=None) -> Path   # RESULTS/<machine>/<family>/<quant_dir>/<suite>
def run_parts(run_dir) -> (machine, family, quant, suite, run_id)
def lock_enabled() -> bool                            # CONFIG [lock] mode: on | off | auto (the lock file's directory exists)
def lock_env() -> dict                                # GPU_LOCK_PATH (+ QUIET_LOCK=off) for gpu_lock.sh and quietlib
def tilde(p) -> str                                   # the home directory as ~
```

`results_dir` is the only place the results layout is spelled out; `Model.results_dir` is
`results_dir(family, quant_dir)`. Without the GPU-lock protocol, `run.Run.lock_sh` and `prepare._lock`
succeed without calling gpu_lock.sh, `notice()` emits no LOCK-NOTICE and does not wait, and quietlib
(`QUIET_LOCK=off`) requires no lock.

Every path printed or written to a committed file goes through `paths.tilde()`.

## Registry (`registry.py`)

```python
@dataclass(frozen=True)
class Model:
    key: str; display: str; served_id: str; path: Path; quant: str
    family: str; quant_dir: str   # docs/benchmark/<machine>/<family>/<quant_dir>/backend/<run-id>
    spec: Literal["mtp", "none"]; vision: bool; reference: bool
    sampling: dict            # {"temperature":..., "top_p":..., "top_k":...}
    proxy_sampling: str       # bench proxy preset
    source: dict; model_settings: dict = {}; server_log_expect: list[str] = []
    prepare: dict = {}; notes: list[str] = []
    def to_dict(self, tilde: bool = True) -> dict      # tilde=False only for spec files under runs/
    link: Path                                          # property: MODELS_DIR / served_id
    results_dir: Path                                   # property: paths.results_dir(family, quant_dir)
class RegistryError(ValueError)
def parse_models(data: dict) -> dict[str, Model]
def load_models(path: Path | None = None) -> dict[str, Model]     # REPO / "models.toml"
def reference(models) -> Model
def get(models, name: str) -> Model                   # registry key or served id
def load_pins(path: Path | None = None) -> dict       # REPO / "pins.toml"; {} when absent
def epoch_id(pins: dict) -> str | None                # `[epoch] id` (or a top-level `epoch = "<id>"`)
```

## Progress lines (stdout, one per line, `print(..., flush=True)`)

`<HH:MM:SS> <TAG> <key=value ...>` with TAG in: LOCK-NOTICE, UNIT-WAIT, UNIT-START, UNIT-DONE, QUIET-WINDOW-END,
TASK-START, TASK-DONE, BATTERY-PAUSE, BATTERY-READY, RESUMED, HOLD, HOLD-RELEASED, RUN-DONE, ERROR.
Helper: `modeleval.progress.emit(tag, **kv)`. During `modeleval run` the orchestrator's stdout carries only these
lines (and `--dry-run`'s plan): the agentbench suites' printed tables go to `runs/<model>/<run-id>/speed-stdout.log`.

## Quiet guard (`quiet.py`, owned by core)

Stdlib only (the capability runner imports `battery`, `boundary` and `hold_requested` under Python 3.13).

```python
class QuietGuard:
    def __init__(self, run_dir: Path, allow_pids: list[int], pidfile=None): ...   # always exempts its own PID
    def start_monitor(self) -> None                      # scripts/quietlib.py monitor, 5 s, log run_dir/speed/monitor.tsv
    def wait_quiet(self, unit: str, settle_s: int = 600) -> bool   # quietlib check --major-only; emits UNIT-WAIT;
                                                                     # False = give up
    def judge(self, t_start: float, t_end: float) -> UnitVerdict    # epoch seconds; waits until the monitor covers t_end+5
    def flags(self, t_start: float, t_end: float) -> dict
        # for capability tasks: {"disturbed" (any flag), "severity", "disturbed_samples", "major_samples",
        #                        "minor_samples", "reasons", "peak_foreign_cpu", "samples", "battery_min"}
    def stop_monitor(self) -> None
@dataclass
class UnitVerdict: clean: bool; reasons: list[str]; peak_foreign_cpu: float; samples: int
                   disturbed_samples: int = 0; battery_min: int | None = None
                   severity: "clean" | "minor" | "major" = "clean"   # not clean without a severity -> "major"
                   major_reasons: list[str]; minor_reasons: list[str]; major_samples: int; minor_samples: int
                   major: bool                                       # property: severity == "major"
def battery() -> tuple[int, bool]                         # percent (-1 without a battery), on_ac
def boundary(kind: Literal["unit", "long-unit", "harness-block", "task"]) -> Literal["go", "pause", "stop"]
    # the user's rule: pause below 50 % (60 % for long-unit / harness-block), stop below 20 % for tasks
def hold_requested() -> bool                              # paths.HOLD_FILE exists
def prelaunch_check(timed: bool, settle_s: int = 600, log: Path | None = None) -> CheckResult
    # timed: the whole quiet rule with --max-gpu-gb 6 and up to settle_s; untimed: one sample, only
    # LAUNCH_BLOCKERS (foreign model resident, lock, port 8500, power, battery) refuse
PAUSE_PCT = {"unit": 50, "long-unit": 60, "harness-block": 60, "task": 50}; STOP_PCT = 20; RESUME_PCT = 80
```

`scripts/quietlib.py` (modified copy, owned by scripts; stdlib only, run as `/usr/bin/python3 -I -B`) keeps its
CLI `check` (+ `--major-only`: fail only on major conditions, print minor ones on a `minor:` line; the default is
the full rule) and `monitor` (+ `--count N`) and `battery`; the old `verify` is gone. The monitor adds the columns
`pressure`, `swapouts_delta`, `foreign_gpu_ms`, `foreign_rss_delta_mb`; the column list is in `MONITOR_COLUMNS`
and in the TSV header line; the state column is DISTURBED (a major flag), MINOR, SPIKE or QUIET. Every flag has
a severity (DESIGN §5.4): major (the unit is repeated) or minor (kept with the flag). Importable:

```python
judge_window(tsv, t_start, t_end, pad_s=5.0, gpu_floor_ms=GPU_FLOOR_MS, major_gpu_ms=MAJOR_GPU_MS) -> dict
    # {"clean", "severity" ("clean" | "minor" | "major"), "reasons", "major_reasons", "minor_reasons",
    #  "peak_foreign_cpu", "samples", "disturbed_samples", "major_samples", "minor_samples"}; "clean" = no flag at
    # all; "reasons" / "disturbed_samples" cover both levels; a coverage problem (no samples, or a gap longer than
    # both 3 intervals and 15 s) is major
window_flags(tsv, t_start, t_end, ...) -> dict    # {"disturbed_samples", "major_samples", "minor_samples",
                                                  #  "peak_foreign_cpu", "battery_min", "samples"}
read_monitor(tsv) -> rows; row_flags(row, prev, ...) -> list[Flag(severity, text)]
MAJOR_FOREIGN_CPU = 300 ($QUIET_MAJOR_FOREIGN_CPU); MAJOR_GPU_MS = 500 ($QUIET_MAJOR_GPU_MS); GPU_FLOOR_MS = 250
MAJOR_SUSTAIN_SAMPLES = 6 ($QUIET_MAJOR_SUSTAIN_SAMPLES); MAJOR_SHARE = 25 % ($QUIET_MAJOR_SHARE)  # CPU/GPU major
    # only when sustained (a run of 6 window samples) or frequent (25 % of them); else minor (DESIGN §5.4)
```

## Server (`server.py`, owned by core)

```python
class Server:
    def __init__(self, model: Model, run_dir: Path, launch_no: int, *, models: list[Model] | None = None,
                 pidfile: Pidfile | None = None, caffeinate: bool = True): ...
    def write_settings(self) -> None      # BASE/settings.json + BASE/model_settings.json (every registry model) from
                                          # DESIGN §2; refuses while PORT listens
    def start(self) -> None               # launch, wait ready (GET /v1/models), 8-token warm-up, check server_log_expect
    def stop(self) -> None                # SIGINT, wait, SIGKILL; port free
    def record(self) -> dict              # this launch for manifest.json
    pid: int; base_url: str               # "http://127.0.0.1:8703/v1"
    log_path: Path                        # runs/<model>/<run-id>/server.log
    effective_settings: dict              # parsed from the startup log (+ "log_expect": the matched lines)
    written_settings: dict                # settings.json as written, minus the auth secret
class MemorySampler:                      # thread, 2 s; writes run_dir/speed/memory.tsv (columns MEMORY_COLUMNS)
    def __init__(self, run_dir: Path, interval: float = 2.0, launch_no: int = 1)   # takes the vm_stat baseline
    def start(self, server_pid: int) -> None; def mark(self, label: str) -> None; def stop(self) -> dict
class Pidfile:                            # runs/<model>/<run-id>/pidfile: pgid, role, marker per child group
    def add(self, pgid, role, marker); def remove(self, pgid); def kill_all(self, timeout=15.0) -> list[str]
```

`MEMORY_COLUMNS = (iso, epoch, launch, label, source, tree_bytes, wired_delta, compressor_delta, file_backed_delta,
pressure, swapouts_delta)`; the deltas are against the launch's pre-launch baseline.

## Speed (`speed.py`, owned by speed)

```python
async def run_functional(ctx: SpeedContext) -> FunctionalResult   # G0, G1, G3, G4 (+G2, G2b with vision), M7
async def run_unit(ctx: SpeedContext, unit: str, attempt: int) -> UnitResult
    # unit in: "g5+m3-64k", "m3-16k", "m3-32k", "m3-110k", "m3-think-16k", "m1", "m4-1", "m4-2", "m4-3"
def timed_units(model: Model) -> list[str]                          # in order; m3-think-16k only if spec != "none"
def required_prompts(model: Model) -> list[str]                     # generated prompts the units need
def finalize(run_dir: Path) -> dict                                 # memory peaks into units.json, after run.py's last write
FIXTURES_DIR = paths.FIXTURES                                       # tests point it at a small tree
@dataclass
class SpeedContext: model: Model; base_url: str; server_pid: int; run_dir: Path; tokenizer_dir: Path; unverified_dir: Path
@dataclass
class UnitResult: unit: str; attempt: int; t_start: float; t_end: float; raw_file: Path; metrics: dict; valid: bool; note: str
@dataclass
class FunctionalResult: ok: bool; stop_reason: str | None; items: dict; m7: dict | None; files: list[Path]
                        t_start: float; t_end: float
```

- `base_url` ends in `/v1`; speed strips it, since agentbench adds `/v1/...` itself.
- Every request: the model's sampling (OpenAI and Anthropic bodies), thinking off through
  `chat_template_kwargs.enable_thinking` only, text-only prompts, `cold_nonce="system+tool"`, fallback token counts
  with the model's own tokenizer (`tokenizer_dir` = the checkpoint).
- `metrics["cold_cached_tokens"]` lists the cached tokens of the requests that must be cold (G5, the M1 cold
  trial, M4 step 0); run.py checks each against one cache block (256).
- The functional block refuses with `ok=False` when a generated prompt is missing; run.py's preflight finds the
  same thing before the lock (`run.missing_prompts`).

`run.py` owns the retry loop: `wait_quiet` -> `run_unit` -> `judge` -> keep a clean or minor attempt (move nothing)
or, for a major verdict, move the raw file into `RESULTS/_unverified/<model>/<run-id>/speed/` and retry (at most 2
retries). An attempt that is invalid (a cold request hit the cache, missing samples, an error) is kept out of the
table and not retried, since repeating cannot fix it; `--unverified` keeps every attempt and never retries.

`speed/units.json` has two writers, in one process and never concurrently. speed records each attempt as soon
as it ends (`{"attempt", "t_start", "t_end", "raw_file", "valid", "note", "metrics", "verdict": null, "kept":
false}`) and the functional block (`{"passed", "time", "result"}`); run.py then merges, by attempt number and
always as a read-modify-write of the file, the quiet verdict (`clean | minor | disturbed | incomplete | invalid`;
`minor` is kept, `disturbed` means major), `kept`, the launch, the quiet reasons (`severity`, `reasons`,
`major_reasons`, `minor_reasons` and the sample counts) and the unit verdict (`clean`, `minor`, `invalid`,
`disturbed x3`), and calls `finalize` after its last write. An attempt left without a verdict (a crash) makes
`--resume` run the unit again; a unit whose verdict is `clean` or `minor` is done.

## Capability (`capability/`, owned by capability)

Before the lock, when the capability leg runs, the orchestrator calls `capability.epoch.ensure_epoch()` (vendor/
is gitignored; a missing snapshot is recreated with `git archive` from the pinned commits, and pinned inputs
that no longer hash to the pinned id refuse the run). It then starts the runner as a subprocess:

```
envs/capability/.venv/bin/python -I -B -m modeleval.capability.runner --spec <runs/.../capability/spec.json>
```

with `cwd=REPO`, an allowlisted environment (DESIGN §6.2; built by `capability.runner.allowed_env(os.environ, extra)`,
which the orchestrator imports, with this repo's own `.venv/bin` removed from PATH; `MODELEVAL_AGENT_PROC_LIMIT_GB`
and `MODELEVAL_AGENT_TREE_LIMIT_GB` override the memory watchdog's limits in GiB, and a bad value is a preflight
refusal), and `sys.path` coming from the
environment's editable install of `src/` (`-I` drops the working directory). `spec.json`:

```json
{"model": {...Model.to_dict(tilde=False)...}, "base_url": "http://127.0.0.1:8703/v1", "server_pid": 123,
 "sets": ["gate", "ranking"], "run_root": "<abs runs/<model>/<run-id>/capability>",
 "results": "<abs results/<model>/<run-id>/capability/results.json>", "ws_root": "<abs WS_ROOT>",
 "epoch": "<epoch id>", "vendor": "<abs vendor/<epoch>>", "wall_s": 3600, "max_steps": 30, "max_turns": 40,
 "thinking": "off", "resume": true,
 "launch": 1, "server_log": "<abs runs/.../server.log>", "pidfile": "<abs>", "monitor": "<abs>"}
```

`launch` is recorded per row and `server_log` is scanned per task for template errors; the runner ignores
`pidfile` and `monitor`. Optional keys for debugging and tests: `harnesses` (a subset), `only` (fnmatch patterns
over task ids), `selftest`. `runner.spec_problems(spec)` is the runner's own check of this file.

Runner exit codes: 0 done; 75 yield, with `<run_root>/yield.json` = `{"reason": "battery" | "battery-stop" |
"hold" | "drift", "detail", "at", "exit_code"}` (state saved; the orchestrator pauses for the battery, waits for
the hold, or stops resumably on drift, then restarts with resume); 76 server lost (the orchestrator relaunches,
then restarts with resume); 2 preflight refused; 1 error. The runner prints TASK-START / TASK-DONE progress
lines (TASK-DONE with `memory_kills=N`), keeps every run in `<run_root>/state.json`, and writes `results.json` (rows, superseded attempts and
warm-ups) and each run's allowlisted files after every task, through `capability/export.py`.

Rows carry `flags: null` and `t_start`/`t_end`; when the memory watchdog (`capability/memwatch.py`, DESIGN §6.1)
killed something during the task, `flags` starts as `{"memory_kills": [{"pid", "cmdline", "footprint_gib", "t",
"limit": "process" | "tree"}]}`. After every runner exit the orchestrator adds `QuietGuard.flags(t_start, t_end)`
to every row whose flags have no `disturbed` yet, keeping `memory_kills`, into `state.json` (so the runner's later
rewrites keep them) and into `results.json` (through `export.write_results`).

`capability/epoch.py`: `create_epoch(sha, *, pins_raw=None, vendor_root=VENDOR) -> str` (pins.toml `[snapshot]`:
`repo` relative to MONOREPO, `sha`, and the paths of `bench`, `smoke`, `shims` at it; archived as
`vendor/<epoch>/personal-agent-bench/bench`, `harness-smoke/{smoke,shims}`; the record's `snapshot` is
`{repo, sha, bench, smoke, shims}`),
`load_epoch(id, vendor_root=VENDOR)`, `verify_live(epoch) -> list[str]` (problems; empty = ok),
`ensure_epoch(pins_path, vendor_root) -> str`, `snapshot_problems(epoch)`, `index_stamps(repos)`, `load_pins(path)`,
all using only the git commands allowed by DESIGN §6.1 (plus `archive` into vendor/). `capability/export.py`:
`is_allowlisted(rel)` is the export allowlist; `scripts/redact.py` mirrors it (it runs under the system Python
in the pre-commit hook and cannot import the package).

## Stats and report (`stats.py`, `report.py`, owned by speed)

```python
def sign_test(wins: int, losses: int) -> float                    # two-sided exact
def per_task_scores(results: list[dict], set_name: str) -> dict[str, int]   # task -> harnesses solved (0-3)
def per_harness(results: list[dict], set_name: str) -> dict[str, dict[str, bool]]
def compare(new: dict, ref: dict, *, set_name=None, new_by_harness=None, ref_by_harness=None) -> dict
    # per-task diffs, wins/losses, p, MDD, decision; the per-harness maps decide the gate rule's
    # "consistent in at least two harnesses" clause
def time_to_solution(new_results, ref_results, set_name) -> dict; def mdd(...); def wilson(k, n)
def build_report(results_root: Path = RESULTS) -> str              # RESULTS.md text
def run_dirs(results_root) -> list[Path]          # .../backend/<run-id>/ with manifest.json or speed/units.json
# publish.py: rows(run) -> ([(path, summary)], messages); publish(root, model, run_id) -> lines
#   SPEED_PROTOCOLS / CAP_PROTOCOLS: protocol id -> (protocol version, builds or epoch, title)
```

`report.py` reads `speed/units.json`, `speed/memory.tsv`, `manifest.json` (its `machine` for the macOS and Xcode
builds) and `capability/results.json` (`rows`, `finished`, `sets`).

## Orchestrator (`run.py`, owned by core)

`modeleval run` exit codes: 0 complete, 1 error, 3 functional gate FAIL, 4 stopped or incomplete (`--resume`
continues), 5 refused before anything was measured, 130 interrupted. Preflight, before LOCK-NOTICE: checkpoint
and registration, port 8703 free, the generated fixture prompts (speed leg), the runner interpreter and
`ensure_epoch()` (capability leg). `plan` and `run --dry-run` print the same checks read-only.

## Files written per run

`results/<model>/<run-id>/` is `paths.results_dir(family, quant_dir) / run_id`, under the repository's
docs/benchmark; `publish` adds `summary.json` there and in the sibling `agent_ability/<run-id>/`.

```
results/<model>/<run-id>/manifest.json, speed/units.json, speed/<unit>-<attempt>.jsonl, speed/gate-<n>.jsonl,
                         speed/m7-<n>.jsonl, speed/monitor.tsv, speed/memory.tsv,
                         capability/results.json, capability/<harness>/<run>/{calls.jsonl, summary.json, check.txt}
                         (speed/*.log are gitignored)
eval/backend/results/_unverified/<model>/<run-id>/speed/...   major-disturbed, incomplete and invalid attempts;
                                                               --unverified runs
runs/<model>/<run-id>/   server.log, pidfile, orchestrator.pid, lock.log, quiet-checks.log, speed-stdout.log,
                         ssd-cache/, manifest.json (refused runs only), capability/{spec.json, state.json,
                         yield.json, selftest.json, scratch/, <harness>/<run>/...} (raw)
```
