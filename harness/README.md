# agentbench (harness)

Benchmark client used unchanged against every server in this experiment. It implements the
gate G1–G5 and the metrics M1–M7 of [../METHODOLOGY.md](../METHODOLOGY.md); see
[STATUS.md](STATUS.md) for what is verified and what still needs a live check.

Python 3.12, uv project, offline. The tokenizer and chat template come from a local directory:
`$AGENTBENCH_TOKENIZER_DIR`, else the oQ5 checkpoint `~/Developer/mlsys/models/oq/Qwen3.8-27B-oQ5-mtp`,
else an HF snapshot of `Qwen/Qwen3.8-27B` or `mlx-community/Qwen3.8-27B-8bit`. The fixtures were
built from the 8-bit snapshot, and the oQ5 files rebuild them byte for byte
(`src/agentbench/tokenizer.py`).

```bash
cd harness
uv sync                              # harness/.venv from uv.lock
uv run pytest -q                     # mock servers only; never a real model server
uv run agentbench fixtures build     # deterministic fixtures in ../fixtures (about 20 s)
```

**The harness never takes the GPU lock and never starts or stops a server.** Follow
METHODOLOGY §3 (lock, port check) before any timed run. It refuses base URLs on 8500 and
8600–8699 (the agent benchmark's ports). Mock servers bind an ephemeral port or 8790–8799.

## Fixtures

`agentbench fixtures build [--out DIR] [--seed N]` writes everything from code and a seed
(default 20261017); `agentbench fixtures check` rebuilds into a temporary directory and
compares every sha256 with `fixtures/manifest.json`.

| Committed (`fixtures/`) | Content |
| --- | --- |
| `system_prompt.md` | agent system prompt (instructions, project context file, skills, memory, environment) |
| `tools.openai.json`, `tools.anthropic.json` | the same 44 generic tools in both formats |
| `m1_m5.json` | M1 task, canned assistant call, 24 tool results (≈250 tokens, `{{REQUEST_ID}}` placeholder) |
| `m4_trajectory.json` | 20 canned steps: assistant tool call, tool result, screenshot id |
| `m7_suite.json` | 24 tool-call cases with expectations and reference answers |
| `gate.json` | G1 prompt and expected call, G2 prompt and image id |
| `manifest.json` | sha256 of every file, measured token sizes, image metadata, the G2 code |

Generated (`fixtures/generated/`, gitignored): 47 fake-UI screenshots (1280×1024 PNG), their
text equivalents (accessibility-tree text of exactly 1282 tokens, the image's token count),
cold prompts, M3 prompts and M5 helper prompts.

Measured with the Qwen3.8 tokenizer and template (thinking off unless noted):

| Item | Tokens |
| --- | --- |
| agent prefix (system turn: 44 tools + system prompt) | 14,905 |
| image 1280×1024 (1280 pads + 2 vision markers) = text equivalent | 1,282 |
| tool result | 250 each |
| M1 warm-up prompt / trial prompt / new tokens per trial | 14,949 / 16,590 / 1,641 |
| M4 canned assistant turn (min / median / max) | 126 / 144 / 167 |
| M4 new tokens per step / step-20 prompt | 1,686–1,727 / 49,082 |
| cold prompts (M2, G5) | 16,000 / 32,000 / 64,000 exactly (+ 2,000 for `--quick`) |
| M3 prompts, thinking off / on | 16,000 / 32,000 / 64,000 and 16,000 / 32,000 exactly |
| M5 helper prompts | 2,000 |

Rules the fixtures follow:

- **Nonces.** Digits only (`[run 123456789012]`); every digit is one token, so a nonce never
  changes a prompt's size. With `tools`, the nonce goes at the start of the first tool's
  description, because the Qwen template renders the tools block before the system prompt;
  without tools it goes at the very start of the first message.
- **Exact prefixes.** The M1 warm-up (prefix + task, ending with a user message) is an exact
  text prefix of every M1 trial; M4 step k is an exact prefix of step k+1 (canned assistant
  turns render with the same empty `<think>` block as the generation prompt). Tests assert both.
- **Uniqueness.** Each M1/M5 trial gets fresh `request_id` digits and its own screenshot; each
  M4 run gets a fresh nonce. Digits differ between runs, everything else is byte-identical
  across frameworks (`prompt.sha256` in the raw records zeroes the digits).
- Sizes that are not pre-built (for example `--sizes 128000`) are generated on the fly with
  the same procedure, but are not covered by the manifest.

## Profiles

`--profile` shapes every request as the last step (after `--extra-body`), and records what it
changed in each raw record (`shaping`).

| Profile | Port | Output dir | Shaping and expectations |
| --- | --- | --- | --- |
| `mlxlm` | 8701 | `mlx-lm/` | omits `model` (any value reloads the model); drops `seed`; reasoning in `reasoning` |
| `llamacpp` | 8702 | `llama.cpp/` | cached tokens from `prompt_tokens_details` or `timings.cache_n` |
| `omlx` | 8703 | `omlx/` | pass-through |
| `lithos` | 8704 | `lithos-metal/` | strips `chat_template_kwargs`, `reasoning_effort` other than `none`, Anthropic `thinking` other than `disabled`; probes that need them are N/A with the reason; 429s recorded; cached tokens from `--server-log` |
| `vllm` | 8705 | `vllm-metal/` | reasoning in `reasoning` |
| `mlc` | 8700 | `mlc-llm/` | tool calls via structured output (below); strips thinking fields (thinking is fixed by the served model config); metric suites default to `--text-only` |
| `generic` | — | `--framework` | pass-through (needs `--base-url` and `--framework`) |

All profiles: reasoning is read from `reasoning` and `reasoning_content` (and inline
`<think>`), and every streaming request carries exactly `stream_options: {"include_usage": true}`.

### Structured-output tool calls (`--profile mlc`)

MLC-LLM has no Qwen tool-call parser, so the `mlc` profile never sends `tools`. Instead it
puts the tool schemas (compact JSON: name, description, parameters) and a one-paragraph format
instruction at the start of the system message, flattens tool history (assistant calls become
the JSON step object, tool results become user messages), and constrains the reply with
`response_format`. The step schema is a union of `{"tool": <const name>, "arguments": <the
tool's parameters>}` per tool, `{"tool_calls": [...]}` (two or more parallel calls) and
`{"final": <string>}`, with one `$defs` entry per tool. The JSON reply is parsed back into the
normal `tool_calls` record, so the gate, M4 and M7 run unchanged; every raw record carries
`tool_call_mode: "structured_output"` (other profiles: `"tool_calls"`), and the gate reports G1
as `PASS via structured output (non-standard)` only if the content parses and the arguments
validate against the tool schema. A nonce moves to the very start of the system message.

| Option | Default | Why |
| --- | --- | --- |
| `--structured-format mlc\|openai\|none` | `mlc` = `{"type": "json_object", "schema": "<schema as a JSON string>"}` | MLC's `RequestResponseFormat.type` is `Literal["text", "json_object"]` with a string `schema` (mlc-llm main 8978ea9a); the OpenAI `{"type": "json_schema", ...}` shape fails validation. `none` = instruction only, no grammar. |
| `--structured-union anyOf\|oneOf` | `anyOf` | MLC's pinned xgrammar (d4f57c4) does not implement `oneOf`: a top-level `oneOf` compiles to "any JSON". The branches are mutually exclusive, so `anyOf` is equivalent. |

If G1 gets HTTP 400/404/422, the gate also sends the same request without `response_format`
(`G1.no-grammar`) to show whether the grammar request itself was rejected.

## Common options (gate, m1–m7, run-all)

| Option | Meaning |
| --- | --- |
| `--profile`, `--base-url`, `--model` | server under test; the base URL defaults to the profile's port |
| `--framework DIR` | output directory under the repo root (default from the profile) |
| `--tag TAG` | row label, e.g. `8bit-spec-off`; part of every file name |
| `--framework-version`, `--launch-flags` | recorded in the provenance header |
| `--server-pid PID` | memory sampling (G5, M6) of the PID tree via `footprint` (RSS fallback) |
| `--server-log PATH` | lithos: read `'cached_prompt_tokens': N` from the server log after each request |
| `--context-config N` | configured context, recorded for G5 (otherwise read from `/v1/models` or `/props`) |
| `--text-only` | replace screenshots with their equal-token text equivalents (M1, M4, M5, M6); M7 image cases become N/A |
| `--screenshot-placement user\|tool` | screenshot in a user message after the tool result (default) or inside the tool message |
| `--thinking-fields hermes\|kwargs\|effort\|none` | how metric suites express thinking (default Hermes: both `reasoning_effort` and `chat_template_kwargs.enable_thinking`) |
| `--quick` | smoke mode: minimal trials and sizes, for integration checks only |
| `--trials N` | trials per cell (suite defaults otherwise) |
| `--extra-body JSON`, `--api-key` | merged into every OpenAI body / bearer token (or `AGENTBENCH_API_KEY`) |
| `--structured-format`, `--structured-union` | `mlc` profile only, see above |
| `--preflight` | METHODOLOGY's quiet-machine rule: run `scripts/quiet_check.sh --allow-pid <server> --allow-pid <harness>` before each suite and store its output in the header; if it fails, the results go to `<framework>/raw/_unverified/` (gitignored, left out of `summary.md`) with a warning (`--preflight-cmd` substitutes another command) |
| `--unverified` | write results to `raw/_unverified/` regardless (exploratory runs) |
| `--interference JSON` | M8: a JSON object stored as the header's `interference` record (scenario, parameters, allowances, load PID, block, run); `scripts/run_m8.sh` builds it with `interference.py meta` |
| `--read-timeout`, `--total-timeout` | seconds (defaults 1800 / 3600; a 64K cold prefill can take minutes) |
| `--out-root`, `--fixtures` | redirect outputs and fixtures (tests use temporary directories) |

Every suite writes `<framework>/raw/<suite>-<tag>-<YYYYmmdd-HHMMSS>.jsonl`. Line 1 is the
provenance header (harness commit and dirty flag, `sw_vers`, `xcodebuild -version` default and
beta, chip, memory, framework version, launch flags, all CLI args; home and `$HF_HOME` paths
redacted). Then one `request` line per request (`metrics`, `prompt`, `shaping`, the full
`record` with per-chunk timestamps), plus `memory`, `skip`, `gate_item` lines as relevant, and a
final `summary` line.

## Subcommands

### `gate` — capability gate G1–G5

```bash
uv run agentbench gate --profile omlx --model Qwen3.8-27B-oQ5-mtp --tag oq5mtp \
    --framework-version 0.7.1rc1 --launch-flags "<flags>" --server-pid <pid>
```

- G1: Hermes-shaped request (`reasoning_effort: "none"` + `enable_thinking: false`) with the
  full agent prefix; the streamed call must assemble to `fs_read_file {"path": "docs/setup.md"}`
  with no markup in `content`. Records whether the call arrived incrementally or as one delta.
  If the Hermes shape gets HTTP 400/404/422, a plain request (no thinking fields) is tried too.
- G2: base64 PNG with a random code; passes only if the answer contains the code. Images are
  always probed, even with `--text-only`.
- G3: Claude Code-style `/v1/messages` (system blocks with `cache_control`): text with
  `thinking: adaptive` and without a thinking field, `tool_use` with the agent tools, an image
  block. Never sends `thinking: disabled`. Servers that think by default (vLLM, llama.cpp,
  oMLX) get `max_tokens=4096`; a text probe whose default thinking streams until `max_tokens`
  still counts as streaming that works, and says so.
- G4: Hermes off/on, `enable_thinking` alone, `reasoning_effort` alone, and the efforts the
  template rejects with thinking on (`high`, `minimal`) to see whether the server maps or
  rejects them. PASS / PARTIAL (only `enable_thinking` works) / FAIL.
- G5: the 64,000-token cold prompt with `max_tokens=8`, memory sampled every second if
  `--server-pid` is given, configured context from `--context-config`, `/v1/models` or `/props`.

Prints the pass/fail table, writes `raw/gate-<tag>-<ts>.md`, and upserts the same fragment into
`<framework>/capabilities.md` between `<!-- agentbench:begin gate <tag> -->` markers (text
outside the markers is kept).

### `m1` — cached-prefix TTFT

```bash
uv run agentbench m1 --profile vllm --model qwen3.8-27b-8bit --tag 8bit --trials 5 --cold-trials 3
```

Warm-up = prefix + task (ends with a user message, `max_tokens=1`). Trials append the canned
call, a unique tool result and a unique screenshot (about 1.6K new tokens). Cold trials use the
same shape with a nonce in the first tool's description. `--rewarm` re-sends the warm-up
before each trial; `--m1-max-tokens` (192) caps the reply. TTFT follows METHODOLOGY (first
content, reasoning or tool-call chunk); `first_kind` and e2e are recorded because oMLX and
mlx-lm emit a tool call only when it is complete.

### `m2` — cold prefill

```bash
uv run agentbench m2 --profile llamacpp --tag 4bit --sizes 16000,32000,64000 --trials 3
```

Fresh nonce at the very start, `max_tokens=1`, one warm-up. Throughput = prompt tokens / TTFT
(server count if reported, local count otherwise; e2e is used and flagged if no token chunk arrives).

### `m3` — short structured decode

```bash
uv run agentbench m3 --profile llamacpp --tag 4bit-mtp --spec-label on
```

Thinking off: caps 50/150/300 at 16K/32K/64K. Thinking on (`--think-effort xhigh`, sent
top-level and in `chat_template_kwargs`): caps 1000/2000 at 16K/32K. The prompt asks for a
150-element JSON array in `content` (no `tools`), so every cap is exceeded; `hit_cap` flags
cells that stopped early. Options: `--contexts`, `--caps`, `--think-contexts`, `--think-caps`,
`--no-thinking-bucket`, `--bust-cache` (default: the warm-up primes the prompt cache).
`--spec-label` is required and only labels the rows (launch the server accordingly).

### `m4` — 20-step trajectory

```bash
uv run agentbench m4 --profile omlx --model Qwen3.8-27B-oQ5-mtp --tag oq5mtp --runs 3
```

Step 0 = prefix + task; step k appends canned call k, tool result k and screenshot k
(`--m4-max-tokens` 150 output, discarded). Per step: TTFT, e2e, cached and prompt tokens; the
summary adds the TTFT slope over steps and the total per run. Each run gets a fresh nonce
(`--no-bust` to disable). `--steps` (20), `--runs` (3).

### `m5` — concurrency

```bash
uv run agentbench m5 --profile omlx --model <id> --tag 8bit --aux 1,2 --aux-lead 0.5
```

Main step = M1 trial shape; helpers = 2K-token summarization requests with fresh nonces and
256-token outputs. Helpers start first and the main step follows `--aux-lead` seconds later
(0 = together). The prefix is re-warmed before each trial (`--no-rewarm` to skip). Concurrent
requests are never retried; 429s are counted per role (lithos-metal).

### `m6` — memory and stability

```bash
uv run agentbench m6 --profile vllm --model qwen3.8-27b-8bit --tag 8bit --minutes 15 --server-pid <pid>
```

Footprint at 64K (cold 64,000-token prompt, sampled every second), then M4 trajectories back to
back for `--minutes`, sampled every `--mem-interval` seconds (5). Reports memory start / peak /
end and growth, latency drift (same step, first vs last complete run) and errors. Without
`--server-pid` only system memory (`vm_stat`) is sampled.

### `m7` — tool-call correctness

```bash
uv run agentbench m7 --profile omlx --model <id> --tag 8bit [--cases m7-03,m7-18]
```

24 streamed cases on top of the agent prefix: single call, multiple args, nested objects,
arrays, enums, numbers and booleans, Unicode/Chinese, quotes and newlines, long strings, two
parallel calls, similar tools, no call needed, calls after an image (file name read from a
screenshot; a click on a Save button, accepted in pixels or 0–1000 normalized coordinates, with
the convention recorded). Success = expected calls with the right names, schema-valid
arguments (jsonschema), matching values, no spurious call, no markup leak.

### `run-all`

```bash
uv run agentbench run-all --profile omlx --base-url http://127.0.0.1:8703 --model <id> \
    --framework omlx --tag 8bit-spec-off --server-pid <pid> \
    --framework-version 0.7.1rc1 --launch-flags "<flags>" \
    --suites gate,m1,m2,m3,m4,m5,m7 --spec-label off --preflight
```

Runs the chosen suites in the canonical order (gate, m1 … m7) against one server, validates the
arguments first (`m6` needs `--minutes`; without `--spec-label`, M3 rows are labelled `off` and a
note is printed), keeps going when one suite fails, switches the remaining suites to
`--text-only` when the server rejects G2's image with an HTTP error (an image that is accepted but
misread is still measured with images; `--no-auto-text-only` keeps images regardless), and
finally upserts `<framework>/summary.md` (section `summary <tag>`) and prints it (`--no-summary`
only prints it: M8 blocks stay out of `summary.md`). `--trials` applies to every suite in the run,
M7 included.

### `summarize`

```bash
uv run agentbench summarize --framework omlx [--tag 8bit-spec-off] [--write] [--all-runs] [--files ...]
```

Markdown tables per suite and cell with `median (min-max)`: gate table, M1 warm/cold, M2 per
size, M3 per (spec, thinking, context, cap) with decode rate, gaps and tokens per chunk, M4 per
step with cached share, M5 per helper count with the ratio to solo and 429 counts, M6 memory and
drift, M7 per case with reasons. Uses the latest file per suite and tag unless `--all-runs`.
`--write` upserts into `<framework>/summary.md`.

### `m8-report`

```bash
uv run agentbench m8-report [--framework llama.cpp ...] [--tag 4bit-mtp-on] [--details] [--unverified]
```

M8 (METHODOLOGY.md, section 4): reads the `<suite>-<tag>-m8-<block>` raw files that
`scripts/run_m8.sh` writes and prints one markdown table: per framework and tag, the baseline
medians, the baseline drift (median change between consecutive baselines: the noise floor) and,
per scenario, the slowdown in M1 warm TTFT, M3 decode (as time per token) and the M4 run total.
Each scenario block is compared with the mean of the baseline blocks on either side of it on the
same server launch (`*` when it is not bracketed by both). `--details` adds one row per block and
suite; `--unverified` reads `raw/_unverified/` instead (mechanics checks).

### `mock-server`

```bash
uv run agentbench mock-server --port 8790 --behavior lithos --log ../logs/mock-lithos.log
```

Without `--behavior`: the legacy scenario mock used by `tests/test_client.py`. With
`--behavior generic|llamacpp|omlx|vllm|lithos|mlxlm|mlc`: a programmable responder that imitates each
framework's documented quirks (strict schema and 429s for lithos, 404 for images and
`/v1/messages` on mlx-lm, `reasoning` vs `reasoning_content`, single-delta tool calls,
`timings.cache_n`, log-only cached tokens, MLC's rejection of `tools` and its JSON-object
`response_format`) and a message-level prefix cache. Point any suite at it
with `--base-url http://127.0.0.1:8790` for an integration check.

### `info`, `mem`

`agentbench info` prints the snapshot and the image-token arithmetic; `agentbench mem --pid P`
samples a PID tree's footprint.
