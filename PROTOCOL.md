# Protocol version 1

This page describes the fixed setting behind every row of `RESULTS.md`. Each run records it in
`results/<model>/<run-id>/manifest.json`. Rows from different protocol versions never share a
table. Speed rows also share a table only when their macOS and Xcode builds match, and
capability rows only when their harness epoch matches. [DESIGN.md](DESIGN.md) explains why each
choice was made.

## Machine and server

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M5 Max (40-core GPU), 128 GB. On AC power in High Power mode. `sw_vers` and `xcodebuild -version` (the global Xcode) are recorded in every result |
| Server | oMLX 0.7.1rc1, stock checkout at `7a5a3d75`. The pipeline owns its own base path, and `HOME` is isolated inside it. Listens on 127.0.0.1:8703; the HF cache lookup is off |
| Global server settings | at most 8 concurrent requests; chunked prefill on; prefill priority `context`; decode fairness on. Hot (in-memory) prefix cache 20 GB. Memory guard `balanced` (soft limit 0.85, hard limit 0.95). Burst decode `balanced`. `max_tokens` 32768. Default sampling 1.0 / 0.95 / top_k 0 (each request carries its own sampling, see below) |
| SSD prefix cache | a fresh directory per run with a fixed 128 GB cap (never "auto"). It is deleted after a complete run, and its peak size goes into the manifest |
| Per-model settings | `max_context_window` 131072. `mtp_enabled` only for models that ship an MTP head. There is never a per-model `max_tokens`: Hermes and localmelo send none, so every model gets the same 32768 |
| Launches | one launch per model and run. A battery pause or a lost server starts a new launch with its own number, and every unit and task records which launch it ran on |
| Load check | the run is refused when a line the registry expects is missing from the server log after the warm-up. These lines prove that the speculative-decoding mode and the offload mode really loaded, for example `Speculative backend selected ...: Lightning MTP (..., active)` or `PLE mode for ...: mmap` |

The settings the server reports at startup and model load are copied into the manifest:
memory guard, SSD cache directory and cap, hot cache, default `max_tokens`, the loaded model's
size, the speculative backend and the PLE mode. Next to them is the `settings.json` the
pipeline wrote, minus its auth section.

## Models

| Item | Rule |
|---|---|
| Quantization | oQ5 (oMLX oQ level 5 with the full-precision sensitivity pass), converted locally from the BF16 release when possible. Otherwise the closest published build is used and disclosed (Qwen3.8-Flash-Next: oQ4e) |
| Speculative decoding | the model's own MTP head when it ships one, otherwise none. It is never reported as a separate row |
| Sampling | the model's recommended agent sampling, used in every generation request of both legs. Qwen3.8 family: temperature 1.0, top_p 0.95, top_k 20. Nanbeige: 1.0 / 0.95 / top_k 0 |
| Thinking | off (`enable_thinking: false`) in every request of both legs. The one exception is the M3 thinking cell for models with speculative decoding: it sends `enable_thinking: true` and nothing else |
| Images | the core comparison is text-only for every model. Screenshots are replaced by text equivalents of the same size |
| Prompts | byte-identical across models. Token counts differ by tokenizer; each result reports the server's own `prompt_tokens` |

## Speed leg

The run gives a 10-minute notice and then takes the shared GPU lock for timed work. Before the
launch it requires that no other model is resident on the GPU, the lock is held, the machine is
on AC power and quiet, and the battery is at 60 % or more. The functional block (G0 prompt
parity, G1, G3 text + tool use, G4, and M7 tool-call accuracy) is not timed. A FAIL on G0, G1,
G3 tool use or G4 ends the run.

The timed units, in this order:

| Unit | What it measures |
|---|---|
| G5 + M3 64K | cold time to first token (TTFT) for a 64,000-token prompt, server memory, then decode speed at 64K (cap 150 tokens, 3 trials) on that cached prefix |
| M3 16K, 32K, 110K | decode speed at each context (cap 150 tokens, 3 trials) |
| M3 thinking 16K | decode speed with thinking on (cap 1000 tokens); only for models with speculative decoding |
| M1 | TTFT on a ~15K-token agent prefix, warm (5 trials) and cold (1 trial) |
| M4 x 3 | a 20-step agent trajectory (cap 150 tokens per step), run 3 times; the median run is reported |

**Quiet rule, per unit: two levels.** Following the project owner's instruction of 2026-10-26 (small
disturbances have little effect: skip them, no re-run), only a major disturbance repeats a unit; a
minor one keeps the result and records the flag. A monitor takes a sample every 5 s, and a unit is
judged from the samples in `[start - 5 s, end + 5 s]`.

- **Major** (the unit is repeated): our lock is missing; a peer's server port (8500, 8600-8699)
  listens; the machine is not on AC power in High Power mode; the battery is below 20 %; foreign CPU
  in total is at or above 300 % in two consecutive samples, sustained or frequent (below); foreign GPU
  time is at or above 500 ms per 5 s sample (10 % of GPU time) in two consecutive samples, sustained
  or frequent; memory pressure reaches level 2, or the system swaps out, while foreign memory is
  growing; the monitor has a gap in the window.
- **Minor** (kept, flagged): one foreign process is above 25 % CPU in two consecutive samples;
  foreign CPU in total is above 150 % in two consecutive samples, below 300 % or at it in a short
  burst; foreign GPU time is above 250 ms in two consecutive samples, below 500 ms or at it in a
  short burst; a build, convert or download tool is running.

Sustained or frequent (since 2026-10-26): CPU or GPU at its major line counts as major only when it
lasts at least 6 consecutive samples (30 s) or when samples at the line make up at least 25 % of the
unit's samples; a shorter burst is minor. On 2026-10-26, four 10-15 s bursts of macOS daemons (the
owner granting app permissions) had each repeated a unit, although such a burst moves a 7-minute M4
run by well under 1 % and an M3 cell reports the median of 3 trials.

Why there: the owner accepts ~10 % error between runs. In the old campaign's M8 on oMLX, one busy
core moved M1 / M3 / M4 by +2.9 % / -0.9 % / +2.0 % against block 1 (about +8.5 % on M1 against
the stable baselines), and eight busy threads by +27 % / +17.5 % / +9.4 %, so M1 reaches ~10 % at
about three busy cores. Foreign GPU time takes time slices from the server directly, so a 10 % share
costs up to ~10 %. Pressure and swapping that the server causes itself are results, not
disturbances. Thermal warnings are logged and not judged.

Before every unit and every retry, the pre-unit check waits up to 10 minutes for the major
conditions to clear (minor ones are only logged); if they do not, the speed leg stops and
`--resume` continues it later. A unit with a major disturbance is moved out of the results and
repeated, at most 3 attempts in total. So is an incomplete one (a missing latency sample, an error,
a lost server, which is relaunched first). A unit that stays disturbed or incomplete has no value;
the table shows "disturbed x3" or "incomplete x3". A unit kept with a minor flag records its
reasons, and its cells in `RESULTS.md` carry a ‡ footnote. A cold request that reuses 256 or more
cached tokens makes its unit invalid, which a repeat cannot fix. A response that opens with a tool
call takes the server's own time to first token: oMLX streams a tool call in one piece after
generating it, so the client would see its first token only after the whole call.

**Reported.** Decode speed in tok/s at 16K, 32K, 64K and 110K, as a curve, with the ratio to the
reference at each context and the time per 100 output tokens. TTFT and memory: after load, peak
during G5, peak during the 110K cell, peak overall, and whether the system reached memory
pressure or swapped. The M4 headline is the sum of TTFT over the 21 steps + 21 x 100 / R, where
R is the run's own decode rate. This keeps a terser model from winning on output length. Also
shown: the TTFT sum alone, the raw total, the mean output length and the number of capped steps.

## Battery rule (both legs)

At every unit, every capability task and between the legs, the run checks the battery. Below
50 % it yields; the threshold is 60 % before the quiet window, before M3, before M4 and before
each harness block. Yielding means: stop the server, release the lock, wait for 80 %, give the
notice again, take the lock back, relaunch and warm up. Below 20 % no capability task starts. A
`runs/hold` file makes the run wait at the next boundary while it keeps the lock.

## Capability leg

The capability leg runs on the same launch, after the quiet window, under an untimed lock
purpose, so peers may do light work meanwhile. It uses three agent harnesses (Claude Code,
Hermes Agent, localmelo), pinned as a harness epoch. Each task runs once, harness by harness.
Each task's budget is 30 proxy steps, 40 harness turns and a 3600 s wall cap. A run counts as
solved when its checker passes within the step budget. A second column also requires that it
finished within 1200 s.

| Set | Tasks | Decision rule (fixed in advance) |
|---|---|---|
| Gate | 12 Aider polyglot Python exercises + 10 MCPMark filesystem easy tasks | "not worse" unless the model loses on at least 5 more tasks than it wins, with the sign consistent in at least two harnesses |
| Ranking | the other 22 Aider Python exercises of the pinned data | "better" or "worse" only at a sign-test p < 0.05; otherwise "no detectable difference" |

The gate set runs first. The ranking set follows unless the gate decision against the reference
(same harness epoch) is "worse"; `--capability full-force` runs it anyway.

The analysis is per task: a model's score on a task is the number of harnesses that solved it
(0 to 3), counted over the harnesses that have a valid result for both models. A (harness, task)
pair excluded on either side (invalid twice) is listed and scored on neither side; a decision with
excluded pairs is marked provisional. Disturbances never repeat a capability task. They only add
flags: the flagged samples (major and minor counts), peak foreign CPU and the battery minimum
during the task.

## Comparability

The reference row (Qwen3.8-27B oQ5 + MTP) is measured once per protocol version, per macOS or
Xcode build (speed), and once per harness epoch (capability). The old campaign measured
differences between nights of up to ~20 % for M1 and M3 and ~9 % for M4. Differences inside
that band are reported as "no difference".
