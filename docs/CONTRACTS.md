# Module contracts (internal)

The interfaces the modules of DESIGN.md §10 agree on, so they can be built in parallel. Keep this file
in sync with the code; DESIGN.md has the why.

## Paths and constants (`src/modeleval/paths.py`, owned by core)

```python
REPO = Path(__file__).resolve().parents[2]            # agent-model-eval/
MLSYS = Path.home() / "Developer/mlsys"
OMLX_REPO = MLSYS / "serving/omlx_0.7"
OMLX_PY = OMLX_REPO / ".venv/bin/python"
OMLX_EXE = OMLX_REPO / ".venv/bin/omlx"               # whatever serve.sh's fw_exe omlx resolves to
BASE = OMLX_REPO / ".modeleval-home"                  # pipeline-owned oMLX base path
MODELS_DIR = MLSYS / "models/hf-bench"
PORT = 8703
WS_ROOT = MLSYS / "experiments/agent-model-eval-ws"
RESULTS = REPO / "results"; RUNS = REPO / "runs"; VENDOR = REPO / "vendor"
GPU_LOCK = MLSYS / "experiments/.gpu-lock"
PROTOCOL_VERSION = 1
```

Every path printed or written to a committed file goes through `paths.tilde()`.

## Registry (`registry.py`)

```python
@dataclass(frozen=True)
class Model:
    key: str; display: str; served_id: str; path: Path; quant: str
    spec: Literal["mtp", "none"]; vision: bool; reference: bool
    sampling: dict            # {"temperature":..., "top_p":..., "top_k":...}
    proxy_sampling: str       # bench proxy preset
    source: dict; model_settings: dict; server_log_expect: list[str]
    prepare: dict; notes: list[str]
def load_models(path: Path = REPO / "models.toml") -> dict[str, Model]
def reference(models) -> Model
def load_pins(path: Path = REPO / "pins.toml") -> dict
```

## Progress lines (stdout, one per line, `print(..., flush=True)`)

`<HH:MM:SS> <TAG> <key=value ...>` with TAG in: LOCK-NOTICE, UNIT-WAIT, UNIT-START, UNIT-DONE, QUIET-WINDOW-END,
TASK-START, TASK-DONE, BATTERY-PAUSE, BATTERY-READY, RESUMED, HOLD, HOLD-RELEASED, RUN-DONE, ERROR.
Helper: `modeleval.progress.emit(tag, **kv)`.

## Quiet guard (`quiet.py`, owned by core)

```python
class QuietGuard:
    def __init__(self, run_dir: Path, allow_pids: list[int]): ...
    def start_monitor(self) -> None                      # scripts/quietlib.py monitor, 5 s, log run_dir/speed/monitor.tsv
    def wait_quiet(self, unit: str, settle_s: int = 600) -> bool   # quietlib check; emits UNIT-WAIT; False = give up
    def judge(self, t_start: float, t_end: float) -> UnitVerdict    # epoch seconds; waits until the monitor covers t_end+5
    def flags(self, t_start: float, t_end: float) -> dict           # for capability tasks: disturbed samples, peak foreign CPU, battery min
    def stop_monitor(self) -> None
@dataclass
class UnitVerdict: clean: bool; reasons: list[str]; peak_foreign_cpu: float; samples: int
def battery() -> tuple[int, bool]                         # percent, on_ac
def boundary(kind: Literal["unit", "long-unit", "harness-block", "task"]) -> Literal["go", "pause", "stop"]
    # the user's rule: pause below 50 % (60 % for long-unit / harness-block), stop below 20 % for tasks
def hold_requested() -> bool                              # REPO / "runs/hold" exists
```

`scripts/quietlib.py` (modified copy, owned by scripts) keeps its CLI (`check`, `monitor`, `battery`) and adds
monitor columns `pressure`, `swapouts_delta`, `foreign_gpu_ms`, `foreign_rss_delta_mb`; the column list is
in its `MONITOR_COLUMNS` constant and in the TSV header line.

## Server (`server.py`, owned by core)

```python
class Server:
    def __init__(self, model: Model, run_dir: Path, launch_no: int): ...
    def write_settings(self) -> None      # BASE/settings.json + BASE/model_settings.json from DESIGN §2; refuses while PORT listens
    def start(self) -> None               # launch, wait ready (GET /v1/models), 8-token warm-up, check server_log_expect
    def stop(self) -> None                # SIGINT, wait, SIGKILL; port free
    pid: int; base_url: str               # "http://127.0.0.1:8703/v1"
    log_path: Path
    effective_settings: dict              # parsed from the startup log
class MemorySampler:                      # thread, 2 s; writes run_dir/speed/memory.tsv
    def start(self, server_pid: int) -> None; def mark(self, label: str) -> None; def stop(self) -> dict
```

## Speed (`speed.py`, owned by speed)

```python
async def run_functional(ctx: SpeedContext) -> FunctionalResult   # G0, G1, G3, G4 (+vision probes), M7
async def run_unit(ctx: SpeedContext, unit: str, attempt: int) -> UnitResult
    # unit in: "g5+m3-64k", "m3-16k", "m3-32k", "m3-110k", "m3-think-16k", "m1", "m4-1", "m4-2", "m4-3"
def timed_units(model: Model) -> list[str]                          # in order; m3-think-16k only if spec != "none"
@dataclass
class SpeedContext: model: Model; base_url: str; server_pid: int; run_dir: Path; tokenizer_dir: Path; unverified_dir: Path
@dataclass
class UnitResult: unit: str; attempt: int; t_start: float; t_end: float; raw_file: Path; metrics: dict; valid: bool; note: str
```

`run.py` owns the retry loop: `wait_quiet` -> `run_unit` -> `judge` -> keep (move nothing) or move the raw file
into `RESULTS/_unverified/<model>/<run-id>/` and retry (max 2 retries).

## Capability (`capability/`, owned by capability)

The orchestrator starts the runner as a subprocess:

```
envs/capability/.venv/bin/python -I -B -m modeleval.capability.runner --spec <runs/.../capability/spec.json>
```

with `cwd=REPO`, an allowlisted environment (DESIGN §6.2; built by `capability.runner.allowed_env(os.environ, extra)`,
which the orchestrator imports), and `sys.path` set inside the runner. `spec.json`:

```json
{"model": {...Model as dict...}, "base_url": "http://127.0.0.1:8703/v1", "server_pid": 123,
 "sets": ["gate", "ranking"], "run_root": "<abs runs/<model>/<run-id>/capability>",
 "results": "<abs results/<model>/<run-id>/capability/results.json>", "ws_root": "<abs WS_ROOT>",
 "epoch": "<epoch id>", "vendor": "<abs vendor/<epoch>>", "wall_s": 3600, "max_steps": 30, "max_turns": 40,
 "thinking": "off", "resume": true}
```

Runner exit codes: 0 done; 75 yield (battery pause or hold or epoch drift: state saved, orchestrator decides and
restarts with resume); 76 server lost (orchestrator relaunches, then restarts with resume); 2 preflight refused;
1 error. The runner prints TASK-START / TASK-DONE progress lines and writes `results.json` after every task.

`capability/epoch.py` provides `create_epoch(pab_sha, smoke_sha) -> str`, `load_epoch(id)`, `verify_live(epoch) -> list[str]`
(problems; empty = ok), all using only the git commands allowed by DESIGN §6.1.

## Stats and report (`stats.py`, `report.py`, owned by speed)

```python
def sign_test(wins: int, losses: int) -> float                    # two-sided exact
def per_task_scores(results: list[dict], set_name: str) -> dict[str, int]   # task -> harnesses solved (0-3)
def compare(new: dict, ref: dict) -> dict                          # per-task diffs, wins/losses, p, MDD, decision
def build_report(results_root: Path = RESULTS) -> str              # RESULTS.md text
```

## Files written per run

```
results/<model>/<run-id>/manifest.json, speed/units.json, speed/*.jsonl, speed/monitor.tsv, speed/memory.tsv,
                         capability/results.json
runs/<model>/<run-id>/   server.log, pidfile, ssd-cache/, capability/<harness>/<task>/... (raw), spec.json
```
