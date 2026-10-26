# agent-model-eval: design

Status: revision 2 of 2026-10-26, after a four-lens review (methodology, efficiency, integration,
constraints; 26 confirmed findings). The implementation follows this file; where code and this file
disagree, fix one of them in the same commit.

## 1. Purpose

One command evaluates a local model as an agent backend on this Mac (MacBook Pro M5 Max, 128 GB) and
puts it in a table next to the reference model, under one fixed setting:

- **speed**: what an agent waits for: cached-prefix TTFT, cold 64K prefill, decode from 16K to ~110K
  context, a 20-step trajectory, memory;
- **capability**: whether agents built on it get work done, through three agent harnesses (Claude Code,
  Hermes Agent, localmelo) on tasks with objective checkers.

Reference model: Qwen3.8-27B oQ5 + MTP (`Qwen3.8-27B-oQ5-mtp`), the user's current agent backend.
First new models: Nanbeige4.3-33B (dense Llama, text only, no MTP) and Qwen3.8-Flash-Next (qwen4_exp
MoE, VLM, MTP).

### Why a new pipeline

The Qwen3.8-27B serving campaign (`../qwen3.8-27b_20261017-2258_benchmark`, 2026-10-17..22) used 51.8
machine-hours over 3+ days. Its logs show where the time went:

| Sink | Hours | What this design does instead |
|---|---|---|
| Whole suites discarded on any disturbance (22 of 34 verdicts rested on one 10 s sample; 6 % of their requests overlapped it; 6.6 h ran after the first bad sample) | 10.8 | wait for quiet before each small unit; judge and repeat only that unit (one M3 cell, one M4 run, G5, M1) |
| M3's full grid | 16.5 | the cells the conclusions cited, plus the long-context cells that differ across architectures |
| Variant sprawl: 8-bit rows, MTP-off rows, temperature-0 duplicates, extra M6 soaks, M8 on candidates | ~20 | one configuration per model: the one agents use |
| Full metric sets for configurations that had already failed the gate | 14.7 | stop after a failed gate |
| One server launch per tag (78 launches for ~22 configurations), fixed 45 s settles, 120 s cool-downs | ~4.5 | one launch per model, no fixed sleeps |
| Battery quarantine and pauses | 5.0 | the user's battery rule at every unit and task boundary, before anything is measured |

Capability came from another session's private bench, measured separately per checkpoint. Here it is
a leg of the same run, on the same server launch, with snapshotted harness code.

## 2. The fixed setting (protocol version 1)

Everything in this section is the same for every model. A change here is a new **protocol version**
(`PROTOCOL_VERSION` in code); rows from different protocol versions never share a table. The macOS
build (`sw_vers`) and Xcode build are part of the speed comparability key: a changed build triggers one
re-run of the reference speed leg.

| Item | Value |
|---|---|
| Machine | MacBook Pro M5 Max (40-core GPU), 128 GB; AC power, High Power mode; `sw_vers` and `xcodebuild -version` in every result |
| Server | oMLX 0.7.1rc1, `~/Developer/mlsys/serving/omlx_0.7` at `7a5a3d75` (clean, never modified), its `.venv`; **pipeline-owned base path** `serving/omlx_0.7/.modeleval-home` (HOME isolated to `.modeleval-home/home`), model dir `~/Developer/mlsys/models/hf-bench`, port **8703**, `OMLX_HF_CACHE_ENABLED=false` |
| Server settings | written to `.modeleval-home/settings.json` before every launch, with the server stopped. Mirrors the user's production backend (`~/.omlx-0.7/settings.json`, read only): `scheduler.max_concurrent_requests` 8, `chunked_prefill` true, `prefill_priority` context, `decode_fairness` true; `cache.hot_cache_max_size` 20GB; memory guard `balanced` (soft 0.85, hard 0.95); `burst_decode_mode` balanced; global `max_tokens` 32768, temperature 1.0, top_p 0.95, top_k 0. Plus: `cache.ssd_cache_dir` a fresh `<run scratch>/ssd-cache` per run, deleted at the end with its peak size logged; `ssd_cache_max_size` fixed at 128GB. The effective values are parsed from the startup log into the manifest |
| Per-model settings | `.modeleval-home/model_settings.json`, rewritten before every launch from the registry: `max_context_window` 131072; `mtp_enabled` per registry; never a per-model `max_tokens` (Hermes and localmelo send none, so every model gets the global 32768) |
| Speculative decoding | the model's own head when it ships one (MTP); none otherwise. Recorded per model, never a separate row |
| Quantization | oQ5 (plain oMLX oQ, level 5, full-precision sensitivity pass) converted locally from the BF16 release when possible; otherwise the closest published build, disclosed (Flash-Next: Jundot's oQ4e, built with oMLX 0.7.3.dev1; its BF16 is ~250 GB) |
| Sampling | the model's recommended **agent** sampling in every generation request of both legs, including the Anthropic probes: Qwen3.8 family 1.0 / 0.95 / top_k 20; Nanbeige 1.0 / 0.95 / top_k 0 (off). Stated in the table footer |
| Thinking | **off** (`enable_thinking: false`) for every request of both legs, except one M3 cell for models with speculative decoding (§5.1). The thinking cell sends `enable_thinking: true` only, no `reasoning_effort`, so every template renders its own default. This matches the decided setting of the user's agent bench. A thinking-on capability arm exists as an option (§6.8) |
| Images | the core comparison is **text-only** for every model (screenshots replaced by their equal-size text equivalents); Nanbeige's row says "vision: none" |
| Prompts | byte-identical across models (same fixtures; same task prompts, which embed one fixed workspace path); token counts differ by tokenizer and are reported per model, with each model's token ratio on the fixture corpus in the manifest |
| Harness epoch | the snapshotted harness code and configs of §6.5 |

## 3. Model registry: `models.toml`

```toml
[models.qwen3-8-27b-oq5]
display        = "Qwen3.8-27B oQ5 + MTP"
served_id      = "Qwen3.8-27B-oQ5-mtp"            # oMLX model id = directory name under models/hf-bench
path           = "~/Developer/mlsys/models/oq/Qwen3.8-27B-oQ5-mtp"   # read only: hard-linked with a peer's copy
source         = { hf_repo = "Qwen/Qwen3.8-27B", revision = "<sha>", made_by = "oq_convert level 5, MTP kept" }
quant          = "oQ5"
spec           = "mtp"                             # "mtp" | "none"
vision         = true
reference      = true
sampling       = { temperature = 1.0, top_p = 0.95, top_k = 20 }
proxy_sampling = "inject:qwen3.8-gen-config"       # bench proxy preset for the capability leg
server_log_expect = ["Lightning MTP enabled"]      # the run is refused if a line is missing
```

- Nanbeige4.3-33B: `hf_repo = "Nanbeige/Nanbeige4.3-33B"`, revision
  `c1093c501cc6460427c0ac634db7223d0450c0ee`, `served_id = "Nanbeige4.3-33B-oQ5"`, `spec = "none"`,
  `vision = false`, sampling 1.0 / 0.95 / 0, `proxy_sampling = "inject:omlx-default"`,
  `prepare = { convert = { level = 5 }, tokenizer_fixes = ["drop-bos-post-processor"] }`.
- Flash-Next: `path` is a link into the agent-benchmark session's `~/.omlx-0.7/models` (read only),
  `quant = "oQ4e"`, `spec = "mtp"`, `vision = true`, `model_settings = { qwen4_ple_ssd_offload = true,
  moe_expert_offload_enabled = false }`, `server_log_expect = ["PLE mode for .*: mmap", "Lightning MTP
  enabled"]`. A resident PLE would need a global wired-limit change (sudo), which only the user makes.
- A model's checkpoint is never written in place by `prepare` or `run` after conversion. Files with a
  link count above 1 or symlinks are refused for any edit.

## 4. Prepare

Split by resource:

- `modeleval download <model>`: `hf download <repo> --revision <sha>` into `$HF_HOME`. No lock. Refuses
  while a speed leg runs (another session's lock with a timed purpose, or our own run in its quiet
  window); allowed during a capability leg.
- `modeleval convert <model>`: `scripts/oq_convert.py` (already used for Nanbeige4.3-33B-oQ5 on
  2026-10-26, 126 s). Needs the GPU lock with an `accuracy only (untimed), convert` purpose and no
  model server (8500, 8703). Refuses when oQ's own memory probe says `requires_proxy`. Applies the
  registry's tokenizer fixes to the output copy and records them in `<out>/agent-model-eval.json`.
  Checks free disk >= output size + 20 GB first.
- `modeleval register <model>`: the `models/hf-bench/<served_id>` link (refused while 8703 listens).

Tokenizer fix `drop-bos-post-processor`: oMLX renders the chat template to text and then encodes it
with special tokens on. Nanbeige's `<|im_start|>` is both its BOS and the template's first token, so
every prompt started `<|im_start|><|im_start|>` (verified with the oMLX venv: 30 vs 29 tokens). The fix
removes the BOS from the `tokenizer.json` post-processor and sets `add_bos_token: false` (the flag alone
does not change the post-processor in transformers 5.17). The reference stacks (Transformers
`apply_chat_template`, vLLM) tokenize chat prompts with `add_special_tokens=False`, which this matches.

## 5. Speed leg

One server launch per model per run (§6.7 covers relaunches). Order:

1. **Before the lock**: print `LOCK-NOTICE <start> <eta> <timed-minutes>` and wait 10 minutes (the
   driving session relays it to the peers with the rules below). Then acquire the lock with purpose
   `modeleval <model> speed (timed)`.
2. **Pre-launch check**: `quietlib check --max-gpu-gb 6 --settle 600` (no foreign model resident,
   lock ours, no build/convert/download tools, AC, High Power), battery >= 60 %.
3. Launch, warm-up (8 tokens), check the server log against `server_log_expect`.
4. **Functional block** (not timed, no quiet requirement, fails fast): G0, G1, G3 text + tool_use,
   G4 (+ G2/G2b/G3 image probes if `vision`), and M7. A FAIL on G0, G1, G3.tool_use or G4 stops the run.
5. **Quiet window** (timed units, §5.4): G5 + M3 64K, M3 16K, M3 32K, M3 110K, M3 thinking 16K (spec
   models only), M1, M4 x 3.
6. `QUIET-WINDOW-END`: the orchestrator rewrites the lock purpose to
   `modeleval <model> capability, accuracy only (untimed)` (`gpu_lock.sh update-purpose`) and starts the
   capability leg on the same launch.

### 5.1 Units

| Unit | Content | Timed |
|---|---|---|
| G0 prompt parity (new, in `modeleval`) | (1) `POST /v1/tokenize` for three requests (no tools; 44 tools; 44 tools + a tool-call history), `chat_template_kwargs {enable_thinking: false}`, `add_special_tokens` unset. (2) The no-tools request must match an independent render (`apply_chat_template(tokenize=True)` with the model's own tokenizer, in the oMLX venv) id for id; report the first differing index. (3) For the tools requests: ids[0:2] are not a doubled BOS; the system prompt appears verbatim; the thinking-off generation suffix is present; the tool-call and tool-response turns are rendered. (4) The real G1 request's `usage.prompt_tokens` equals the tokenize length | no |
| G1, G3 (text, tool_use), G4 | the old gate probes, driven one by one by `speed.py` (not `gate.run()`, which writes `capabilities.md` and always sends image probes). In the copied harness, G3 is split so that its image and tool_result-image probes run only with `vision=true`; with `vision=false` G3 passes on text + tool_use and the image parts are N/A | no |
| M7 | 22 text tool-call cases x 5 trials (110) at the model's sampling; the 2 image cases are N/A in text-only mode. Reported as a per-case gate: a case fails if it misses in >= 3 of 5 trials; failing cases are listed. Not part of the capability score | no |
| G5 + M3 64K | the 64,000-token M3 prompt (thinking off) as the cold G5 probe: TTFT, server footprint, system memory; then the M3 64K cell (cap 150, 3 trials) reusing that prefix (record `cached_tokens` on trial 0; fall back to a normal warm-up if it was not cached) | yes |
| M3 cells | thinking off, cap 150, 3 trials each, at 16,000, 32,000 and 110,000 tokens (the 110K fixture is new; ~114K Nanbeige tokens, inside 131072). Each cell's warm-up uses `max_tokens` 16. Thinking cell 16,000 x cap 1000 only for `spec != none` (without speculative decoding, per-token work does not depend on thinking) | yes |
| M1 | 1 warm-up + 5 warm trials + 1 cold trial on the ~15K-token agent prefix | yes |
| M4 | 20-step trajectory, cap 150, text-only, **3 runs**, median reported | yes, each run is a unit |

Reported decode speed: tok/s per context (a curve from 16K to 110K, and the ratio to the reference at
each context), plus seconds per 100 output tokens. Token counts on code differ between tokenizers; M3's
output is JSON-shaped, where the ratio is about 1.

**M4 reporting.** Its total mixes byte-identical prefill with each model's own output length, so a
terser model wins regardless of speed. Headline = sum of TTFT over the 21 steps + 21 x 100 / R, where R
is the run's decode rate (completion tokens / decode time over steps with server usage), and the footer
says so. Also reported: the TTFT sum alone; the raw total; mean completion tokens per step; cap hits as
k/21 (`finish_reason=length` plus `incomplete_tool_call`). A stream that ends in `incomplete_tool_call`
after its first token is a valid latency sample (TTFT and e2e valid, no usage). Fallback token counts use
the model's own tokenizer, never Qwen's.

**Cold-request nonces.** The old fixtures put a cold nonce in the first tool's description, because
Qwen's template renders tools before the system prompt. Nanbeige renders the system prompt first, so its
5.7K-token system prompt would be a cache hit in "cold" requests. Rule: every request that must be cold
and carries tools (the M1 cold trial, step 0 of every M4 run) puts the same fresh nonce at the start of
the system message **and** at the start of the first tool's description. Whichever renders first breaks
the cache match; prompts stay byte-identical across models. Runtime check: on those requests the
server's `cached_tokens` must be below one cache block (256), else the unit is invalid.

Dropped by default (each still callable in `harness/`): M2, M5, M6 (its 30-min throughput was
predictable from M4 within 2-4 %), M8 (framework-level), M3 caps 50/300/2000, spec-off rows,
temperature-0 rows.

### 5.2 Memory

Sampled every 2 s for the whole launch: the server PID tree's `footprint` (RSS fallback), and
system-wide `vm_stat` (wired, compressor-occupied, file-backed pages) as a delta from the pre-launch
baseline, plus `kern.memorystatus_vm_pressure_level` and the swap-out delta. Reported: after load, peak
during G5, peak during the 110K cell, peak overall, and whether the server itself drove the system into
pressure or swap. Footnote for Flash-Next: its 29.8 GiB PLE table is mmapped and file-backed, so
`footprint` excludes it; the system-wide delta includes it.

### 5.3 Fixtures

The committed fixtures of the old harness are reused byte for byte (agent prefix, tool results,
trajectory, M7 cases, gate prompts, screenshot text equivalents), plus one new generated M3 prompt of
110,000 tokens (`M3_OFF_SIZES`). Prompt sizes were built with the Qwen3.8 tokenizer ("16,000 tokens"
means 16,000 Qwen3.8 tokens); every result records the server's own `prompt_tokens`.

### 5.4 Quiet rule (per unit)

The disturbance rule is the old campaign's, unchanged in its thresholds (METHODOLOGY section 3): a
foreign process above 25 % CPU in two consecutive samples, foreign CPU in total above 150 % in two
consecutive samples, a lock/port/power/battery (< 20 %) event, or a build/convert/download tool. Two
additions: foreign GPU time (per-process `accumulatedGPUTime` from `ioreg`, outside the server tree,
WindowServer and kernel_task, above a floor taken from an idle trace) in two consecutive samples; and
memory pressure >= 2 or swap-out **only when foreign RSS also grew** in that window (pressure the server
causes itself is a result, §5.2). pmset thermal warnings and thermal pressure are logged, not judged.
The thresholds could be relaxed only after a calibration run on the reference (`B cpu2 B cpu4 B` with
`interference.py`, ~30 min), which is not part of this protocol version.

What changes is the unit and the order:

1. Before every timed unit and before every retry: `quietlib check --allow-pid <server> --allow-pid
   <orchestrator> --settle 600`. While it settles, print `UNIT-WAIT <reason>`. If it does not settle,
   stop the speed leg cleanly (`--resume` continues later) instead of burning attempts.
2. The monitor (a modified copy of quietlib's sampler, 5 s interval, extra columns: pressure level,
   swap-out delta, foreign GPU time, foreign RSS delta) runs for the whole launch.
3. A unit is disturbed if a disturbed sample falls in `[t_start - 5 s, t_end + 5 s]`, judged from the
   orchestrator's own timestamps and the logged columns (not the old `verify`, which judged whole files).
   A disturbed unit goes to `results/_unverified/` and is repeated after the next passing check, at most
   twice. A unit still disturbed leaves its metric empty ("disturbed x3").

No downloads, builds, conversions or test suites by the driving session or its subagents during a quiet
window.

### 5.5 Battery (the user's rule, both legs)

At every speed unit, every capability task, and between legs: below 50 % (60 % before the quiet window,
before M3, before M4, and before each harness block) the run yields: stop the server (no model left
resident), release the lock, print `BATTERY-PAUSE`, wait for 80 %, print `BATTERY-READY`, give the
10-minute notice, retake the lock, relaunch, warm up again (the speed warm-up, or the harness warm-up),
print `RESUMED`. Every relaunch is a numbered launch in the manifest; every unit and task records its
launch. The 40 % (unit start) and 20 % (any sample) limits stay as speed validity guards; 20 % is a hard
stop for capability tasks (macOS caps power near 7 %).

A `hold` file makes the run wait at the next boundary while keeping the lock (the old `logs/hold`).

## 6. Capability leg

Runs on the same launch after `QUIET-WINDOW-END`, under an `accuracy only (untimed)` lock purpose, so the
peers' light work continues. It needs the GPU but no quiet machine. Nothing is repeated because of a
disturbance: every task is flagged with the disturbed samples and battery minimum that overlapped it.

### 6.1 How the harnesses are driven

The three harnesses come from personal-agent-bench's `bench.harness` (and its proxy) and harness-smoke's
`smoke` package. Other sessions own both repos; the pipeline never writes there.

- **Snapshot, not live trees.** At epoch creation, `git archive <sha>` of personal-agent-bench `bench/`
  and harness-smoke `smoke/` + `shims/` into the pipeline's gitignored `vendor/<epoch>/` (both repos are
  private and have no remote; the snapshot is never committed). Every path that `bench.manifest`,
  `bench.harness` and `smoke.paths` derive from their file location (`MLSYS_ROOT`, `AGENT_ROOT`,
  `CLAUDE_ENV`, `CLAUDE_HOME`, `HERMES_BIN`, `HERMES_HOME`, `SMOKE_ROOT`, `PAB_ROOT`, `GPU_LOCK`, ...) is
  set explicitly after import, and a selftest asserts each resolved path.
- **Interpreters.** Three pipeline-owned environments, all locked by the pipeline's `uv.lock`s:
  `.venv` (Python 3.12: `modeleval` + `agentbench`); `envs/capability` (Python 3.13: httpx, pyyaml,
  starlette, uvicorn, for the runner and the proxy, which `bench.harness` starts as
  `sys.executable -m bench.proxy`); `envs/agent-tools` (pytest only: the Python the agents see on PATH
  and the checker's interpreter, set through `smoke.paths.SMOKE_VENV_BIN` / `SMOKE_PYTHON`). The runner
  runs as `python -I -B` with `sys.path` set explicitly.
- **Task loop owned by the runner.** The runner writes the env file once (`adapters.write_env_file`),
  iterates over (harness, task) and calls `smoke.runner.run_one(...)` per task under its own workspace
  lock, writing each record as soon as it finishes. This gives yield points between tasks and lets
  `--resume` skip pairs that already have a valid verdict. Retries get a distinct run name.
- **Health checks.** Before each task: server PID tree alive and `GET /v1/models` answers; battery; hold.
  After each task: scan its `calls.jsonl` for upstream errors. A dead server stops the leg, relaunches
  the same configuration with a warm-up, marks the task `server_lost` and runs it again.
- **Claude Code config.** Per-run copies of `agent/configs/claude-code/claude-code.env` and
  `agent/homes/claude-code/settings.json` from the epoch snapshot, byte copies with only these variables
  replaced by name: `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`,
  `CLAUDE_CODE_SUBAGENT_MODEL` (all = `served_id`), `CLAUDE_BIN` (2.1.302),
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (131072). The copy must still contain `CLAUDE_CONFIG_DIR`, a non-empty
  placeholder `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY=""`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`,
  `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`; otherwise the run refuses.
- **Proxy sampling.** A replacement for `bench.harness._start_proxy` that adds `--sampling
  <proxy_sampling>`.
- **localmelo.** `--request-timeout 1800 --timeout <wall - 60> --output-format json`; a localmelo JSON
  run status `timeout` maps to the `timeout` verdict (as Claude Code and Hermes, which are killed).
- **Writes into peer trees: none.** `PYTHONDONTWRITEBYTECODE=1` and `GIT_OPTIONAL_LOCKS=0` are exported
  by the orchestrator at startup for every child. `bench.manifest._git` is replaced by a version that runs
  only `git rev-parse` and `git --no-optional-locks status --porcelain`. The only git commands run on a
  peer repo anywhere in the pipeline are `rev-parse`, `--no-optional-locks status --porcelain` and
  `diff-index --quiet HEAD`. Hygiene audit: the mtime and inode of each pinned repo's `.git/index` (and
  worktree index) are recorded before and after our own operations; a change marks the run with a
  hygiene failure. Known, disclosed exception: Hermes's per-run home symlinks the shared `tools/` and
  `installs/` stores.

### 6.2 Environment and workspace

- The runner starts with an **allowlisted environment**: PATH, HOME, USER, LOGNAME, SHELL, TMPDIR, TERM,
  LANG, LC_*, HF_HOME, NVM_DIR, the pipeline's BENCH_*, OMLX_*, PAB_ROOT, HARNESS_SMOKE_DATA,
  PYTHONDONTWRITEBYTECODE, GIT_OPTIONAL_LOCKS. `DEVELOPER_DIR` is dropped (the manifest records the
  global Xcode). At start it asserts that no variable matching `CLAUDE*`, `ANTHROPIC*`, `OPENAI*`,
  `*TOKEN*`, `*_KEY` is present, and records the sorted variable names. `CLAUDE_CONFIG_DIR` and
  `HERMES_HOME` point to scratch directories for the whole process (the manifest's version probes
  otherwise write `~/.hermes/source-checks`).
- **ws_root** = `~/Developer/mlsys/experiments/agent-model-eval-ws`, one fixed path for every model and
  run (it is part of every prompt, so of `prompt_sha256`). Preflight refuses if `git -C <ws_root>
  rev-parse` succeeds or if any of Hermes's project markers (pyproject.toml, setup.py, setup.cfg,
  requirements.txt, package.json, Makefile, Dockerfile, AGENTS.md, CLAUDE.md, .cursorrules, ...) exists in
  ws_root or its parents below `$HOME`. `GIT_CEILING_DIRECTORIES=<ws_root>` is added for the agents.
  Shims make `pip`, `pip3`, `uv`, `uvx` fail on the agents' PATH.
- Benchmarked `claude -p` instances open their own `/tmp/cc-socks` sockets and appear in the driving
  session's agent list during the leg; the driving session messages peers only by their full listed name.

### 6.3 Tasks (two sets, reported separately, never pooled)

- **Gate set** (non-inferiority), always: the 12 small Aider polyglot Python exercises used so far
  (`transpose`, `proverb`, `pig-latin`, `list-ops`, `variable-length-quantity`, `robot-name`,
  `grade-school`, `affine-cipher`, `scale-generator`, `wordy`, `phone-number`, `react`) and the 10
  MCPMark filesystem easy tasks, x 3 harnesses = 66 runs.
- **Ranking set** (is it better?), by default after the gate passes: the other 22 Aider Python exercises
  of the pinned data (`7e0611e`), fixed in advance by the data, never chosen from the reference's past
  failures, x 3 harnesses = 66 runs. `--capability gate` skips it.
- One unscored warm-up per harness block. One run per task. Harness-major order (Claude Code, Hermes,
  localmelo), sequential.
- Budgets per task: 30 proxy steps (the capability budget), 40 harness turns, wall cap 3600 s (so that
  decode speed does not turn into failures; a hung run is ended earlier by the 1800 s request timeout).

### 6.4 Scoring and checks

- **Solved** = the checker passes (it also runs after a wall timeout) within the 30-step budget. A second
  column, "solved within 1200 s", keeps the reference-equivalent time budget visible. For a timed-out run,
  the model time (sum of `calls.jsonl` `duration_s`) is recorded next to the wall time, so slow decoding
  can be told from a hang.
- **Invalid** (excluded, re-run once) only for pipeline mismatches: a call's `model` is not the target's
  `served_id`, `sampling_forwarded` or `thinking_forwarded` differs from the registry. A 5xx, a reasoning
  leak while thinking is off, or a template fallback in the server log counts as that model's failure, is
  flagged, and appears in a per-model anomaly count.
- Every task records the disturbed samples, peak foreign CPU and battery minimum that overlapped it
  (flags only).

### 6.5 Harness epoch

Defined by content, so that other sessions' commits to docs, results or scripts never force a new epoch:
the git tree hashes of the snapshotted `bench/`, `smoke/`, `shims/` (`git rev-parse <sha>:bench`, ...);
sha256 of the config copies (claude-code.env, settings.json, the Hermes home without logs, sessions and
caches; `memories/` and `skills/` must be empty) and of Hermes's `tools/facts.json`; the uv.lock hashes of
the pipeline environments; the localmelo worktree SHA (`c295ce2`), the Hermes tag and SHA (`v0.22.2`,
`d923a5eb48`) with a clean `agent/hermes-agent`, the Claude Code version (2.1.302), and the data commits
(Aider polyglot `7e0611e`, MCPMark `cd45b7f`). The live parts (Hermes, localmelo, the Claude Code binary)
are re-checked before every task; on a mismatch the leg stops resumably. `pins.toml` holds the epoch.

### 6.6 Statistics

Analysis is per task, not per run: for each task, a model's score is the number of harnesses that solved
it (0-3). Against the reference, per set: the per-task differences, a two-sided sign test across tasks,
and the task counts where each model is ahead. RESULTS.md prints, per set, the minimum detectable
difference at the current sizes and the decision rule, fixed in advance:

- gate set: "not worse" unless the new model loses on >= 5 tasks more than it wins, with the sign
  consistent in at least two harnesses;
- ranking set: "better" / "worse" only at sign-test p < 0.05; otherwise "no detectable difference".

Time-to-solution is paired: over the (harness, task) pairs both models solved, the sums and ratio of
wall time and of model time, and steps per pair.

### 6.7 Resume and relaunch

`--resume <run-id>` skips speed units with a clean result and (harness, task) pairs with a valid verdict,
removes a stale workspace lock of our own, and merges the chunk records. Every child process group
(oMLX, proxies, harness PGIDs) is recorded in a run pidfile and killed on exit or signal; the lock is
released only after nothing listens on 8703; the lock carries the run id, and release checks it.

### 6.8 Optional arms (off by default)

- `--thinking on`: capability with thinking on (4x wall budget); needs the reference in the same arm.
- `--vision-extra`: M1 and M4 in image mode for VLMs. Tool-result images (G2b) need the patched oMLX
  (`57620a3f`), which belongs to another session; until a pipeline-owned build exists this arm reports
  user-message images only.

## 7. Reference rows and epochs

- A table only contains rows with the same protocol version and macOS/Xcode builds (speed) or the same
  harness epoch (capability).
- The reference runs through this pipeline once per protocol version (speed) and once per harness epoch
  (capability), not once per new model.
- The footer prints the measured cross-night band of the old campaign (M1 and M3 up to ~20 %, M4 ~9 %)
  and calls differences inside it "no difference".
- Old campaign numbers appear only in a separate "earlier measurements" note: they used image-mode M1/M4,
  temperature 0 for some rows, three M4 runs with whole-run exclusion of cut steps, and an older
  localmelo (`5c009a8-dirty`).

## 8. Results, raw data and the public repo

```
results/<model>/<run-id>/              committed after the export and the redaction check
  manifest.json                        model entry, protocol version, epoch, server version and effective
                                       settings, launches, pins, sw_vers/xcodebuild, repo SHA; ~ paths only
  speed/units.json                     verdict per unit and attempt, peak foreign CPU, memory peaks
  speed/<unit>-<attempt>.jsonl         agentbench raw files of clean attempts
  speed/monitor.tsv, speed/memory.tsv
  capability/results.json              per (harness, task): verdict, check, steps, turns, wall and model
                                       time, call checks, flags, prompt_sha256, launch
results/_unverified/                   gitignored: disturbed units, mechanics checks
runs/<model>/<run-id>/                 gitignored: raw harness run dirs (transcripts, homes, workspaces,
                                       proxy logs), per-run config copies, ssd-cache, pidfile
RESULTS.md                             generated
```

Only an allowlisted exporter writes capability files under `results/`: `calls.jsonl` (refused if bodies
were logged), `summary.json` with paths rewritten to `~` or relative, the results row, and the check
output as `check.txt` with home paths rewritten. Never exported: `prompt.md` (its sha256 is kept),
`manifest.json` of bench.harness (needed fields are folded into ours), stdout, `claude-home/`,
`hermes-home/`, `mem/`, `workspace/`, the env file.

`scripts/redact.py --check` (run on `git diff --cached --name-only` by a pre-commit hook) fails on
`/Users/`, the current hostname, private IPv4 addresses, machine/user/session-id UUIDs, non-UTF-8 files,
any symlink, and any file under `results/*/*/capability/` not on the allowlist; it still replaces
third-party app names.

## 9. Command line

```
uv run modeleval models                 # registry, and what each model has
uv run modeleval download|convert|register <model>
uv run modeleval plan <model>           # units, per-model time estimates, what is already done
uv run modeleval run <model> [--legs speed,capability] [--capability gate|full] [--unverified]
                                        [--resume <run-id>]
uv run modeleval report                 # rebuild RESULTS.md
uv run modeleval selftest               # offline: mock servers, fake harness, registry, paths, hygiene
```

Progress lines for the driving session: `LOCK-NOTICE`, `UNIT-WAIT`, `UNIT-START`, `UNIT-DONE <verdict>`,
`QUIET-WINDOW-END`, `TASK-DONE`, `BATTERY-PAUSE`, `BATTERY-READY`, `RESUMED`, `HOLD`, `HOLD-RELEASED`,
`RUN-DONE`. The orchestrator never messages other sessions itself.

`plan` estimates times per model from the registry (weight bytes, KV bytes per token, spec) and, after a
first launch, from the model's own G5 and M3 numbers. Rough expectations: Qwen3.8-27B oQ5 quiet window
~30 min, gate set ~1 h, ranking set ~1.5-2.5 h; Nanbeige oQ5 quiet window ~45-55 min, gate set ~2-2.5 h,
ranking set ~3-5 h.

## 10. Implementation layout

```
agent-model-eval/
  DESIGN.md  README.md  PROTOCOL.md  RESULTS.md  resume.md
  models.toml  pins.toml  pyproject.toml  uv.lock  .gitignore
  harness/                 agentbench, copied from ../qwen3.8-27b_.../harness at c15a0a8. Changes against
                           the copy: G3 vision split; cold-nonce rule (system + first tool); M3 110K size
                           and warm-up cap; sampling merged into Anthropic bodies; per-model tokenizer
                           for fallback counts. Each change is listed in harness/CHANGES.md
  fixtures/                copied byte for byte from c15a0a8 (+ the generated 110K prompt, gitignored)
  src/modeleval/
    registry.py            models.toml + pins.toml
    prepare.py             download, convert, register (wraps scripts/oq_convert.py)
    server.py              settings writing, oMLX launch/stop/ready, log checks, memory sampler, pidfile
    quiet.py               monitor wrapper, unit verdicts, pre-unit checks, battery/hold, lock calls
    speed.py               G0, the gate probes, M7, the timed units, on agentbench's in-process API
    capability/
      epoch.py             snapshot, hashes, live re-checks
      runner.py            runs in envs/capability; task loop, adapters of §6.1-6.2
      checks.py            calls.jsonl checks, verdict mapping
      export.py            allowlisted export into results/
    stats.py               per-task sign tests, minimum detectable difference, Wilson intervals
    report.py              RESULTS.md
    run.py                 orchestrator: launches, units, retries, progress lines, resume
    cli.py
  envs/capability/  envs/agent-tools/   pipeline-owned uv projects
  scripts/                 gpu_lock.sh (+ update-purpose, run-id), lib.sh, quietlib.py (modified
                           sampler), quiet_check.sh, redact.py (extended), oq_convert.py, pre-commit
  tests/                   offline only
```

Tests never start a real model server, never take the GPU lock, never write outside temporary
directories, and never import the live personal-agent-bench or harness-smoke trees in a way that writes
there (they use a temporary `git archive` snapshot or fakes).
