# agent-model-eval: design

Status: draft of 2026-10-26, written before any code. The implementation follows this file;
where code and this file disagree, fix one of them in the same commit.

## 1. Purpose

One command evaluates a local model as an agent backend on this Mac (MacBook Pro M5 Max, 128 GB)
and puts it in a table next to the reference model, under one fixed setting:

- **speed**: what an agent waits for (prompt prefill, cached-prefix TTFT, decode, a 20-step
  trajectory, memory);
- **capability**: whether agents built on it get work done, through three agent harnesses
  (Claude Code, Hermes Agent, localmelo) on tasks with objective checkers.

Reference model: Qwen3.8-27B oQ5 + MTP (`Qwen3.8-27B-oQ5-mtp`), the user's current agent backend.
First new models: Nanbeige4.3-33B (dense Llama, text only, no MTP) and Qwen3.8-Flash-Next
(qwen4_exp MoE, VLM, MTP).

### Why a new pipeline

The Qwen3.8-27B serving campaign (`../qwen3.8-27b_20261017-2258_benchmark`, 2026-10-17..22) used
51.8 machine-hours over 3+ days. Its own logs show where the time went:

| Sink | Hours | What this design does instead |
|---|---|---|
| Whole suites discarded on any disturbance (22 of 34 verdicts rested on one 10 s sample; only 6 % of their requests overlapped it) | 10.8 | judge and repeat the smallest unit (one M3 cell group, one M4 run, the gate, M1) |
| M3's full grid (thinking caps 1000/2000, 64K cells) | 16.5 | the trimmed grid the conclusions actually cited |
| Variant sprawl: 8-bit rows, MTP-off rows, temperature-0 duplicates, extra M6 soaks, M8 on candidates | ~20 | one configuration per model: the one agents use |
| Full metric sets for configurations that had already failed the gate | 14.7 | stop after a failed gate |
| One server launch per tag (78 launches for ~22 configurations), fixed 45 s settles, 120 s cool-downs | ~4.5 | one launch per model, no fixed sleeps |
| Battery quarantine and pauses | 5.0 | check per unit, start a speed block at >= 50 % |

Capability came from another session's private bench, measured separately per checkpoint (~1 h each).
Here it is a leg of the same run, on the same server launch, with pinned harness versions.

Target cost per new model: prepare (download + convert) ~20-30 min unattended, speed ~25-40 min in a
quiet window, capability ~1-2.5 h that needs no quiet machine.

## 2. The fixed setting

Everything in this section is the same for every model. A change here is a new **protocol version**
(`PROTOCOL_VERSION` in code); results from different protocol versions are never put in one table.

| Item | Value |
|---|---|
| Machine | MacBook Pro M5 Max (40-core GPU), 128 GB; AC power, High Power mode; `sw_vers` build and `xcodebuild -version` recorded in every result |
| Server | oMLX 0.7.1rc1, `~/Developer/mlsys/serving/omlx_0.7` at `7a5a3d75` (clean), its `.venv`; base path `serving/omlx_0.7/.bench-home` (HOME isolated to `.bench-home/home`), model dir `~/Developer/mlsys/models/hf-bench`, port **8703**, `OMLX_HF_CACHE_ENABLED=false` |
| Context window | `max_context_window` 131072 for every model (per-model `model_settings.json`) |
| Speculative decoding | the model's own head when it ships one (MTP: `mtp_enabled=true`); none otherwise. Recorded per model, never a separate row |
| Quantization | oQ5 (oMLX oQ, level 5) converted locally from the BF16 release when possible; otherwise the closest published oQ build, disclosed (Flash-Next: Jundot's oQ4e, BF16 is ~250 GB) |
| Sampling | the model's recommended **agent** sampling, in every request of both legs: Qwen3.8 family 1.0 / 0.95 / top_k 20; Nanbeige 1.0 / 0.95 / no top_k (oMLX `top_k` 0). Stated in the table footer |
| Thinking | **off** for every request of both legs (agent steps emit tool calls), except the M3 thinking cell. A thinking-on capability arm exists as an option (§6.6) |
| Images | core comparison is **text-only** for every model (screenshots replaced by their equal-size text equivalents); VLMs may add an image-mode extra |
| Prompts | byte-identical across models (same fixtures, same task prompts); token counts differ by tokenizer and are reported per model |
| Harness pins (capability) | personal-agent-bench, harness-smoke, localmelo bench worktree `c295ce2`, Hermes `v0.22.2` (`d923a5eb48`), Claude Code `2.1.302`, Aider polyglot `7e0611e`, MCPMark `cd45b7f`. The set of pins is the **harness epoch** (§7) |

## 3. Model registry: `models.toml`

One table per model. Keys:

```toml
[models.qwen3-8-27b-oq5]
display      = "Qwen3.8-27B oQ5 + MTP"
served_id    = "Qwen3.8-27B-oQ5-mtp"              # oMLX model id = directory name under models/hf-bench
path         = "~/Developer/mlsys/models/oq/Qwen3.8-27B-oQ5-mtp"
source       = { hf_repo = "Qwen/Qwen3.8-27B", revision = "<sha>", made_by = "oq_convert level 5, MTP kept" }
quant        = "oQ5"
spec         = "mtp"                               # "mtp" | "none"
vision       = true
reference    = true
sampling     = { temperature = 1.0, top_p = 0.95, top_k = 20 }
proxy_sampling = "inject:qwen3.8-gen-config"       # bench proxy preset for the capability leg
thinking_kwarg = "enable_thinking"
notes        = []
```

`prepare` fields (only for models built here): `convert = { level = 5, preserve_mtp = "auto" }`,
`tokenizer_fixes = ["drop-bos-post-processor"]`. Models that are only linked
(`link_from = "<dir>"`) are read, never written.

Nanbeige4.3-33B entry, for the record:
`hf_repo = "Nanbeige/Nanbeige4.3-33B"`, revision `c1093c501cc6460427c0ac634db7223d0450c0ee`,
`spec = "none"`, `vision = false`, `sampling = {1.0, 0.95, top_k 0}`, `proxy_sampling = "inject:omlx-default"`,
`tokenizer_fixes = ["drop-bos-post-processor"]` (§4.2).

## 4. Prepare (`modeleval prepare <model>`)

Unattended, no quiet window, but it needs the GPU lock for the conversion (it loads the BF16 model).

1. **Download** into the HF cache (`$HF_HOME`, `~/Developer/ml_source/huggingface`) with `hf download
   <repo> --revision <sha>`. Never into the repo or `models/`.
2. **Convert** with oMLX's oQ quantizer (the logic of `../qwen3.8-27b_.../scripts/oq_convert.py`,
   generalized): level from the registry, `preserve_mtp` true only if the source has `mtp.*` tensors.
   Output `~/Developer/mlsys/models/oq/<served_id>`. Refuse while any model server runs (oQ needs
   ~88 GB free for the full-precision sensitivity pass, otherwise it silently falls back to a proxy).
3. **Tokenizer fixes** on the converted copy only, each recorded in `<out>/agent-model-eval.json`:
   - `drop-bos-post-processor`: oMLX renders the chat template to text and then calls
     `tokenizer.encode(text)` with special tokens on. A tokenizer whose post-processor prepends a BOS
     that the template already writes gets it twice. Nanbeige: `<|im_start|>` is both the BOS and the
     template's first token, so every prompt starts `<|im_start|><|im_start|>` (verified 2026-10-26 with
     the oMLX venv: 30 vs 29 tokens). Fix: `tokenizer.json` `post_processor` without the BOS, and
     `add_bos_token: false` in `tokenizer_config.json` (the flag alone does not change the
     post-processor in transformers 5.17). The reference stacks (Transformers `apply_chat_template`,
     vLLM) tokenize with `add_special_tokens=False`, so the fix matches them.
4. **Register**: link `models/hf-bench/<served_id>` -> the checkpoint dir; set `max_context_window`
   131072 and `mtp_enabled` in `.bench-home/model_settings.json` (refuse while 8703 listens: a running
   server rewrites that file).
5. **Offline checks** (no server): config loads; template renders with and without tools, thinking on
   and off; the **prompt-parity count** (§5.1 G0) for the fixture messages is computed and stored.

## 5. Speed leg

One server launch per model per run. Order: warm-up -> gate -> M1 -> M3 -> M4 (all timed, quiet
window) -> M7 (untimed) -> capability leg (untimed) -> stop.

### 5.1 Suites (the copied `agentbench` harness, `harness/`)

| Unit | Content | Notes |
|---|---|---|
| gate | G0 prompt parity (new), G1 tool call (OpenAI, streamed), G3 Anthropic `/v1/messages` (text, `tool_use`), G4 thinking control, G5 one cold 64,000-token prompt (TTFT + server footprint); G2/G2b/G3-image only if `vision` | a gate FAIL on G0, G1, G3-tool_use, G4 or G5 stops the run before M1 |
| M1 | 1 warm-up + 5 warm trials + 1 cold trial; ~15K-token agent prefix (44 tools + system prompt), each trial adds a canned call, a tool result and a screenshot text equivalent | cold trial nonce placed by the template (§5.3) |
| M3 cell groups | `16000 x cap 150` (3 trials), `32000 x cap 150` (3), thinking `16000 x cap 1000` (3); each group has its own warm-up | the cells the campaign's conclusions cited |
| M4 runs | 20-step trajectory, cap 150, text-only; 2 runs, a 3rd if the two totals differ by more than 5 % | the headline agent-latency number |
| M7 | 24 tool-call cases x 5 trials (120) at the model's sampling | correctness, untimed; runs after the quiet window |

**G0 prompt parity (new).** For three fixed requests (no tools; 44 tools; 44 tools with a tool-call
history), compare oMLX's `usage.prompt_tokens` with the count from the model's own tokenizer and
template via `apply_chat_template(..., tokenize=True)` (thinking off). Equal: PASS. Different: FAIL
with the first diverging token ids. This catches doubled BOS tokens, dropped system prompts and
template-kwarg mismatches before anything is timed.

Dropped by default (each still callable from `harness/`): M2 (cold prefill is covered by G5 and M1
cold), M5, M6 (its 30-min throughput was predictable from M4 within 2-4 %; memory comes from G5 and
the launch-wide sampler), M8 (framework-level), M3 64K and caps 50/300/2000, spec-off rows,
temperature-0 rows.

### 5.2 Memory

The server's footprint (`footprint` of the PID tree, RSS fallback) is sampled every 2 s for the whole
launch. Reported: after load (idle), peak during G5 (64K), peak overall.

### 5.3 Fixtures

The committed fixtures of the old harness are reused byte for byte for every model (the agent
prefix, tool results, trajectory, M7 cases, gate prompts, the screenshot text equivalents). Prompt
sizes were built with the Qwen3.8 tokenizer ("16,000 tokens" means 16,000 Qwen3.8 tokens); each
result records the server's own `prompt_tokens`.

Nonce placement for cold requests becomes template-aware. Today a nonce goes into the first tool's
description because the Qwen template renders tools before the system prompt. Nanbeige renders the
system prompt first, so that position leaves the system prompt cacheable and a "cold" trial would be
partly warm. `--nonce-position auto` renders the request with two different nonces at each candidate
position (start of the first tool description; start of the system message; start of the first user
message) using the model's tokenizer and template, and picks the position whose first differing token
index is smallest. The chosen position and index are recorded.

### 5.4 Disturbance policy (per unit)

A monitor (the old `quietlib` sampler) runs for the whole launch at a 5 s interval and logs every
sample. A sample is **disturbed** if any of:

- the GPU lock is not ours, or port 8500 listens (another model server);
- not on AC power, or not High Power mode, or battery below 20 %;
- a build, convert or download tool runs (compilers, `hf`/`huggingface-cli`, `git clone`, `uv sync`, ...);
- foreign CPU in total above **200 %** in two consecutive samples;
- memory pressure level >= 2 (`kern.memorystatus_vm_pressure_level`) or a swap-out burst.

A single foreign process is no longer judged by itself. Evidence, the campaign's M8 on oMLX: one fully
busy core cost +2.9 % on M1, -0.9 % on M3, +2.0 % on M4; eight busy threads cost +27 % / +17.5 % / +9.4 %.
Two cores therefore stay well inside the ~10 % the user accepts. Every unit still records the peak
foreign CPU seen during it.

Unit verdict: a unit is disturbed if a disturbed sample falls in `[t_start - 5 s, t_end + 5 s]`. A
disturbed unit is moved to `results/_unverified/` and repeated at once, at most twice. A unit that is
still disturbed leaves its metric empty in the table ("disturbed x3") instead of blocking the run.
Thermal pressure is logged, not judged (every sustained run reaches "heavy" on its own).

Before the server starts: lock held, no foreign model resident (GPU memory in use <= 6 GB), battery
>= 50 %, AC, High Power. Before each unit: battery >= 40 %; otherwise print `BATTERY-PAUSE`, wait
charging with the server idle (it stays loaded), and continue at 80 %.

The orchestrator prints machine-readable progress lines (`UNIT-START`, `UNIT-DONE <verdict>`,
`QUIET-WINDOW-END`, `BATTERY-PAUSE`, `BATTERY-READY`) so the driving session can relay them to the
peer sessions.

## 6. Capability leg

Runs on the same launch after `QUIET-WINDOW-END`. It needs the GPU but no quiet machine, and is never
repeated because of a disturbance: verdicts are pass/fail and a slower step only moves wall time.

### 6.1 Harnesses and how they are driven

The three harnesses come from personal-agent-bench's `bench.harness` and harness-smoke's `smoke`
package, imported read-only from their directories (other sessions own both; nothing there is
written). The runner executes with personal-agent-bench's interpreter
(`personal-agent-bench/.venv/bin/python -I`), because `bench.harness` starts its proxy with
`sys.executable -m bench.proxy`. Environment: `PYTHONDONTWRITEBYTECODE=1`, `GIT_OPTIONAL_LOCKS=0`.

Adaptations, all by module attributes at run time (harness-smoke's own end-to-end tests do the same):

| Problem (file:line in the research notes) | Adaptation |
|---|---|
| run.py's preflight requires a lock held by `agent-benchmark`; run.py writes into harness-smoke (`ws/`, `runs/`) | call `smoke.runner.run_suite(bench.harness, Settings, run_root, ws_root)` directly with our own `run_root` and `ws_root`; our own preflight |
| Claude Code's model id lives in `agent/configs/claude-code/claude-code.env:19-23` (and `CLAUDE_BIN` at :14, compaction window at :41) and `agent/homes/claude-code/settings.json:3`, sourced inside `sh -c` | write per-run copies with the model's `served_id`, `CLAUDE_BIN` = 2.1.302, window 131072; set `bench.harness.CLAUDE_ENV` / `CLAUDE_HOME` to them |
| `bench.harness._start_proxy` never passes `--sampling` | replacement that adds `--sampling <proxy_sampling>` |
| import-time constants (`MODEL`, `UPSTREAM`, manifest defaults on port 8500) | set `BENCH_MODEL`, `OMLX_BASE_URL=http://127.0.0.1:8703/v1`, `OMLX_REPO`, `OMLX_BASE_PATH`, `BENCH_MODEL_PATH`, `BENCH_LOCALMELO_REPO` before import, and the module attributes after |
| localmelo's 120 s request timeout | `localmelo_args = ["--request-timeout", "1800", "--timeout", "1200", "--output-format", "json"]` |

### 6.2 Tasks and budgets

- Aider polyglot Python, the 12 "extended" exercises (`transpose`, `proverb`, `pig-latin`, `list-ops`,
  `variable-length-quantity`, `robot-name`, `grade-school`, `affine-cipher`, `scale-generator`,
  `wordy`, `phone-number`, `react`), checked by pytest against pristine tests.
- MCPMark filesystem easy, 10 tasks, checked by their `verify.py`.
- One unscored warm-up per harness. One run per task. Harness-major order (Claude Code, Hermes,
  localmelo), sequential, so prompts stay byte-identical across harnesses (same `ws/<task>` path).
- Per task: 30 proxy steps, 40 harness turns, **1200 s** wall (the old 600 s budget would turn a slower
  decoder's time into failures; the step budget is the capability budget, and wall time is reported
  separately).

66 scored runs per model (36 Aider + 30 MCPMark) plus M7's 120 tool calls.

### 6.3 Checks after every run

From `calls.jsonl`: every call's `model` is the target's `served_id`; `sampling_forwarded` matches the
registry; `thinking_forwarded` says off and no call returned reasoning; no call hit HTTP >= 500. A run
that fails a check is `invalid` (reported, excluded from the denominator, and re-run once).

### 6.4 Metrics

Per model: pass counts per harness and task family with Wilson 95 % intervals; total of 66; M7 of 120;
median wall time per solved task and calls per solved task. Against the reference: paired by
(harness, task), the counts solved only by each side and an exact McNemar p.

### 6.5 Pins

`pins.toml` lists the epoch's pins (repo SHAs and clean trees for personal-agent-bench and
harness-smoke, the localmelo worktree SHA, Hermes tag and SHA, the Claude Code version, the data
commits). The runner refuses to start if any pin differs (`--accept-new-epoch` writes a new epoch
instead, see §7).

### 6.6 Optional arms (off by default)

- `--thinking on` capability arm, with 4x wall budgets. It needs the reference in the same arm.
- `--vision-extra` for VLMs: M1 and M4 in image mode (G2b decides whether tool-result images work).
- `--tasks aider-all` (34 Python exercises) for a tighter capability estimate (~3x time).

## 7. Reference rows and epochs

- A comparison table only contains rows with the same protocol version and, for capability, the
  same harness epoch.
- The reference model is run through this pipeline once per protocol version (speed) and once per
  harness epoch (capability). It is not re-run for each new model.
- Old campaign numbers are cited in a separate "earlier measurements" note, never mixed into the
  table: they used image-mode M1/M4, 3 M4 runs at temperature 0, and an older localmelo
  (`5c009a8-dirty`).

## 8. Results and reports

```
results/<model>/<run-id>/            run-id = YYYYmmdd-HHMMSS
  manifest.json                      model entry, protocol version, epoch, server version, launch
                                     command, sw_vers/xcodebuild, pins, git SHA of this repo
  speed/<unit>-<attempt>.jsonl       agentbench raw files (provenance header + requests)
  speed/monitor.tsv                  the launch-wide monitor
  speed/memory.tsv
  speed/units.json                   verdict per unit and attempt
  capability/<harness>/<task>/...    bench.harness outputs (no workspace contents of MCPMark data)
  capability/results.json
results/_unverified/                 gitignored: disturbed units, mechanics checks
RESULTS.md                           generated: one speed table, one capability table, footnotes
```

`modeleval report` rebuilds `RESULTS.md` from `results/`. Each row uses the newest complete run of
the model in the current protocol version and epoch.

Public-repo hygiene: no absolute home paths (`~` only), no bodies of agent requests, `scripts/redact.py
--check` before every commit.

## 9. Command line

```
uv run modeleval models                  # list the registry and what each model has
uv run modeleval prepare <model>         # download, convert, fix, register (idempotent)
uv run modeleval plan <model>            # units, estimated minutes, what is already done
uv run modeleval run <model> [--legs speed,capability] [--unverified] [--resume <run-id>]
uv run modeleval report                  # rebuild RESULTS.md
uv run modeleval selftest                # offline: mock servers, fixtures, pins, registry
```

`run` takes and releases the GPU lock itself (`scripts/gpu_lock.sh`, holder line
`holder: local_ai-benchmark session`); the driving session messages the peers on the progress lines.
`--resume` skips units that already have a clean result in that run.

## 10. Implementation layout

```
agent-model-eval/
  DESIGN.md  README.md  PROTOCOL.md  RESULTS.md  resume.md
  models.toml  pins.toml  pyproject.toml  uv.lock  .gitignore
  harness/                 copy of agentbench from ../qwen3.8-27b_.../harness at c15a0a8, own uv project
  src/modeleval/
    registry.py            models.toml + pins.toml loading and validation
    prepare.py             download, convert (oq), tokenizer fixes, register, offline checks
    server.py              oMLX launch/stop/ready, memory sampler
    quiet.py               monitor wrapper, disturbance policy, unit verdicts, battery, lock
    speed.py               speed units on top of agentbench's in-process API
    capability/
      runner.py            runs under personal-agent-bench's interpreter; adapters of §6.1
      checks.py            calls.jsonl checks
    report.py              tables and statistics
    run.py                 orchestrator: plan, units, retries, progress lines, resume
    cli.py
  scripts/                 gpu_lock.sh, lib.sh, quietlib.py (copied), redact.py
  tests/                   offline only (mock servers, fake harness, temporary dirs)
```

Tests never start a real model server, never take the GPU lock, never write outside temporary
directories, and never import personal-agent-bench or harness-smoke in a way that writes there.
