# Model-call recording proxy

`bench/proxy/` sits between a harness and oMLX. The harness talks to the
proxy, the proxy relays every request to oMLX unbuffered and appends one JSON
line per call to `<run-dir>/calls.jsonl`. It also makes the inference
settings the same for every harness: one thinking level, one sampling policy
and one step budget per run.

```
harness ──> proxy (127.0.0.1:86xx) ──> oMLX (127.0.0.1:8500)
                 └─> <run-dir>/calls.jsonl, <run-dir>/bodies/ (opt-in)
```

The run-level records (manifest, events, results) are described in
`docs/run-records.md`; this page covers the proxy and its per-call record.

## Start

One proxy instance per run, on a port in 8600-8699:

```sh
uv run python -m bench.proxy --port 8600 --harness hermes --thinking off \
    --max-steps 40 --run-id <run-id> --run-dir runs/<run-id>
```

| Option | Default | Meaning |
|---|---|---|
| `--upstream URL` | `OMLX_BASE_URL` without `/v1`, else `http://127.0.0.1:8500` | model server root; a trailing `/v1` is stripped |
| `--host` / `--port` | `127.0.0.1` / `8600` | listen address; `--port 0` picks a free port and prints it |
| `--run-id` | generated `YYYYmmdd-HHMMSS-xxxxxx` | written into every record; a request header `X-Bench-Run-Id` overrides it per call |
| `--run-dir` | `$BENCH_RUNS_DIR/<run-id>`, else `runs/<run-id>` | where `calls.jsonl` and `bodies/` go |
| `--harness` | none | label written into every record |
| `--thinking` | `passthrough` | see [Thinking modes](#thinking-modes); the runner passes the arm's level |
| `--sampling` | `inject:qwen3.8-gen-config` | see [Sampling policy](#sampling-policy) |
| `--max-steps N` | no budget | see [Roles and the step budget](#roles-and-the-step-budget) |
| `--budget-mode` | `reject` | `reject` refuses main calls beyond the budget, `record` only flags them |
| `--log-bodies` | off | also store full request and response bodies |
| `--[no-]inject-stream-usage` | on | add `stream_options.include_usage=true` to streaming OpenAI requests |
| `--read-timeout` | 3600 s | seconds without upstream data before a call fails |
| `--random-seed` | off | a fresh random `seed` on every OpenAI generation request; see [Sampling policy](#sampling-policy) |
| `--one-reasoning-field` | off | an assistant message with both `reasoning_content` and `reasoning` keeps one; see below |
| `--quiet` | off | no per-call summary line on stderr |
| `--upstream-token-cmd CMD` | none | command printing a bearer token for the upstream; see below |

The proxy only writes `calls.jsonl` and `bodies/`.

**An upstream that needs a token.** Some servers, such as oMLX with its API key set (since
2026-10-24) or Recursive's rec engine (removed on 2026-10-26), answer only requests with
`Authorization: Bearer <token>`. With `--upstream-token-cmd '<command>'` (for oMLX, one that reads
`auth.api_key` from its `settings.json`; for rec it was `rec auth token`) the proxy runs the command once at
startup, keeps its output in memory, and sends it upstream in place of any
`Authorization` header the harness sent, so the harness keeps a placeholder key.
The token is never printed or recorded: `bodies/NNNNNN-headers.json` holds the
client's headers, redacted. `bench.harness` passes the option when
`BENCH_UPSTREAM_TOKEN_CMD` is set, and leaves that variable out of the harness's
environment. Each run starts its own proxy, so each run fetches the token afresh.

**rec and replayed reasoning (`--one-reasoning-field`).** rec is gone, but the option stays for any
upstream with the same rule. rec 0.1.0 answered HTTP 400 "Use one
reasoning field per message" when an assistant message in the history carries both
`reasoning_content` and `reasoning`, even with the same text; either one alone is accepted. Hermes
replays both to local endpoints (`_LOCAL_CARRIERS` in its `agent/message_sanitization.py`). With
this option the proxy keeps one of them, the non-empty one with `reasoning_content` first, and
records a warning on each call it changed. localmelo replays no reasoning, so the option changes
nothing there.

## Pointing the harnesses at the proxy

The proxy forwards paths unchanged, so each harness keeps its usual path
layout and only the host and port change.

- **Hermes**: in `config.yaml`, set `providers.<name>.api` to
  `http://127.0.0.1:8600/v1` (instead of `http://127.0.0.1:8500/v1`).
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8600` (no `/v1`;
  Claude Code appends `/v1/messages`). Its startup probe `HEAD /api/hello`
  is answered locally with 200.
- **localmelo**: `base_url = "http://127.0.0.1:8600/v1"` (the
  OpenAI-compatible provider posts to `<base_url>/chat/completions`).

To tag calls of one run when several share a proxy, send
`X-Bench-Run-Id: <id>`; the header is not forwarded to oMLX. Per-run state
(`main_index`, the step budget, `images_new`) is kept per run id.

## Thinking modes

`--thinking` rewrites `/v1/chat/completions`, `/v1/messages` and
`/v1/messages/count_tokens` bodies:

| Mode | OpenAI request | Anthropic request |
|---|---|---|
| `passthrough` | unchanged | unchanged |
| `off` | `chat_template_kwargs.enable_thinking=false` | `thinking={"type":"disabled"}` + same `chat_template_kwargs` |
| `on` | `enable_thinking=true`, no `reasoning_effort` (template default, xhigh for Qwen3.8) | `thinking={"type":"enabled"}` + same |
| `effort:X` | `enable_thinking=true`, `reasoning_effort=X` | `thinking={"type":"enabled"}` + same |

Other keys of `chat_template_kwargs` are kept. In every mode except
`passthrough`, `chat_template_kwargs` becomes the only thinking control: the
proxy removes top-level `reasoning_effort`, `reasoning` and `thinking_budget`
(OpenAI) and `output_config.effort` (Anthropic); `thinking.budget_tokens`
and `thinking.display` are dropped with the old `thinking` value.

Why this works on oMLX 0.3.9: `MessagesRequest` has a `chat_template_kwargs`
field (`omlx/api/anthropic_models.py:186-187`); `/v1/messages` merges it
(`omlx/server.py:3393-3397`) before it maps `thinking.type` onto
`enable_thinking` (`3399-3406`), so `reasoning_effort` survives and reaches the
template (`3504`, `3510-3514`).

The Qwen3.8 chat template accepts `reasoning_effort` only as `xhigh`,
`medium` or `low` while thinking is on; any other value, including `high`,
makes oMLX answer 400 `Chat template error`. The proxy accepts
`effort:high` (other models may support it) but warns at startup.

## Sampling policy

`--sampling` applies to generation requests (`/v1/chat/completions`,
`/v1/messages`). It acts on the top-level fields `temperature`, `top_p`,
`top_k`, `min_p`, `typical_p`, `presence_penalty`, `frequency_penalty`,
`repetition_penalty`, `xtc_probability`, `xtc_threshold` and `seed`:

| Policy | Effect |
|---|---|
| `passthrough` | the client's values are forwarded unchanged |
| `strip` | all of them are removed, so the server defaults apply |
| `inject:qwen3.8-gen-config` (default) | removed, then `temperature=1.0, top_p=0.95, top_k=20` (the model's `generation_config.json`) |
| `inject:omlx-default` | removed, then `temperature=1.0, top_p=0.95, top_k=0` (oMLX's global defaults) |

`sampling_original` and `sampling_forwarded` record the fields before and
after. A proxy instance for a bench-owned call that sets its own sampling,
such as the T4 reader (temperature 0), must run with `--sampling
passthrough`; the default policy would replace temperature 0 with 1.0.

**oMLX 0.7.1rc1 takes every injected field from the request**, ahead of the
model's and the global settings unless `force_sampling` is set
(`get_sampling_params`, `omlx/server.py:1811`). `/v1/chat/completions`
(`create_chat_completion`, `4912`) passes `temperature`, `top_p`, `top_k`,
`min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty` and
`xtc_*`; `/v1/messages` (`create_anthropic_message`, `7486`) passes
`temperature`, `top_p`, `top_k` and `repetition_penalty`. So the injected
`top_k=20` takes effect on both endpoints. oMLX 0.3.9 never read `top_k` or
`repetition_penalty` from a request, and its `/v1/messages` used only
`temperature` and `top_p`.

**`--random-seed`** sets `seed` to a fresh random integer below 2^31 on every
`/v1/chat/completions` request, after the policy has run. `sampling_forwarded`
records the value and `sampling_policy` gains `+random-seed`. It is for servers
whose default seed is fixed: Recursive's rec engine samples every unseeded
request with the same seed, so the same prompt gives the same output even at
temperature 1.0 (local_ai-benchmark, 2026-10-22). oMLX does not read `seed`.

## Roles and the step budget

Every generation request gets a role:

- `main` when the request carries a non-empty `tools` list;
- `aux` otherwise (title generation and similar side calls).

`count_tokens`, `/api/hello` and other paths have `role: null`; they are
never steps. Main calls are numbered per run: `main_index` is 1 for the
first main call, and counts every main call, including failed or refused ones.

A **step** is a main call that succeeded: HTTP 200, no error, and for a
stream the end marker was seen. This is the design's `n_llm`.

With `--max-steps N`, a main call that arrives when the run's steps plus
its main calls still in flight already reach `N` is beyond the budget:

- `--budget-mode reject` (default): the proxy answers it itself with HTTP 400
  without contacting oMLX. The error code `bench_budget_exhausted` is
  `error.code` for OpenAI clients and the start of `error.message` (type
  `invalid_request_error`) for Anthropic clients. The record has
  `budget_exhausted: true`, `handled_locally: true`, and
  `error.type: "bench_budget_exhausted"`.
- `--budget-mode record`: the call is forwarded and only flagged with
  `budget_exhausted: true`.

A failed main call gives its step back, so upstream errors do not use up the
budget. Aux calls never count.

`bench/proxy/steps.py` computes the run's counts from `calls.jsonl` with the
same definitions:

```python
from bench.proxy.steps import load_calls, step_counts, new_image_long_sides

records = load_calls("runs/<run-id>")             # a run dir or calls.jsonl
step_counts(records, t0=t0_mono, t_end=t_end_mono)
# {"n_llm", "n_llm_aux", "n_llm_err", "n_llm_post_submit", "n_tool", "images_new_sum"}
new_image_long_sides(records, t0=t0_mono, t_end=t_end_mono)   # for the GUI check
```

`t0` and `t_end` are `time.monotonic()` values and a call is in the window
when its `mono_start` is. `n_llm + n_llm_aux + n_llm_err` covers every
generation request in the window once: `n_llm_err` takes failed calls of
either role, including budget refusals. `n_tool` sums `tool_calls` over the
`n_llm` calls. `n_llm_post_submit` counts steps that started after `t_end`.

`request_tool_names` is the sorted list of tool names in the request (names
only, no schemas), and `request_tools_sha256` is the sha256 of that list as
compact JSON, for example `["Bash","Read"]`. Compute other hashes that are
compared with it (an allowlist) with `bench.proxy.request_info.tool_names_sha256`.
`request_tools_raw_sha256` hashes the `tools` array exactly as sent (request order,
key order and schemas). When it changes between two calls of one run, the harness
reordered or edited its tools, and the server's prefix cache breaks at the tools.

## Images

For each image in a request (OpenAI `image_url` / `input_image` parts and
Anthropic `image` blocks, including those inside `tool_result` content), the
record has its sha256 (of the decoded image bytes) in `image_sha256` and
`[width, height]` in `image_wh`. Sizes are read from the PNG, JPEG, GIF, WebP
or BMP header. Images given by URL or file id cannot be identified without
fetching them, and have `null` in both lists.

`images_new` is the number of distinct images that no earlier generation
request of the same run contained, and `images_new_long_side` lists their
long sides (`max(width, height)`), which the GUI comparability check expects
to be 1280. `count_tokens` requests neither count nor mark images as seen,
since Claude Code may count a new screenshot before sending it. A call
refused by the budget does not mark its images as seen.

## Call record (`calls.jsonl`)

One JSON object per line, written when the call ends, so concurrent calls
can appear out of `call_index` order. Times `ts_*` are Unix seconds
(`time.time()`), `mono_*` are `time.monotonic()` seconds; differences of
`mono_*` values are the reliable durations.

| Field | Meaning |
|---|---|
| `schema_version` | 2 |
| `run_id`, `run_id_source` | run id; `"cli"` or `"header"` (from `X-Bench-Run-Id`) |
| `call_index` | 0-based arrival order within this proxy instance |
| `harness` | `--harness` label |
| `protocol` | `"openai"`, `"anthropic"`, or `null` for other paths |
| `endpoint` | `chat_completions`, `messages`, `count_tokens`, `hello` (answered locally), `other` |
| `role` | `main`, `aux`, or `null` for non-generation endpoints |
| `method`, `path`, `query` | request line (`query` e.g. `beta=true` from Claude Code) |
| `upstream`, `handled_locally` | upstream root; `true` for `/api/hello` and budget refusals |
| `model`, `stream`, `max_tokens` | from the request body |
| `request_bytes`, `forwarded_bytes` | body size as received and as sent upstream (after rewrites) |
| `messages`, `tools` | counts |
| `request_tool_names`, `request_tools_sha256` | sorted tool names of the request and their hash |
| `request_tools_raw_sha256` | hash of the `tools` array as sent (order and schemas included) |
| `images`, `image_sha256`, `image_wh` | images in the whole request (history included), with hash and size each |
| `images_new`, `images_new_long_side` | images first seen in this run, and their long sides |
| `user_agent`, `anthropic_beta` | client identification headers |
| `thinking_mode`, `thinking_original`, `thinking_forwarded`, `thinking_rewritten` | the `--thinking` value; thinking settings before and after (`{}` = none sent); whether they differ |
| `sampling_policy`, `sampling_original`, `sampling_forwarded` | the `--sampling` value; sampling fields before and after |
| `usage_injected` | the proxy added `stream_options.include_usage` |
| `main_index` | 1-based number of this main call in its run |
| `max_steps`, `budget_exhausted` | the configured budget; whether this main call was beyond it |
| `status`, `content_type`, `response_bytes` | upstream response (or the proxy's own 400/502/504) |
| `usage` | normalized `{prompt_tokens, completion_tokens, cached_tokens}`; for Anthropic `prompt_tokens = input + cache_creation + cache_read` |
| `usage_raw` | usage object as returned (OpenAI streams include oMLX timing fields) |
| `tool_calls`, `tool_names` | tool calls in this response (OpenAI `tool_calls` deltas, Anthropic `tool_use` blocks) |
| `stop_reason` | OpenAI `finish_reason` or Anthropic `stop_reason` |
| `output_text_chars`, `output_reasoning_chars` | generated characters of text and of thinking |
| `counted_input_tokens` | result of `count_tokens` |
| `sse_events`, `sse_keepalives`, `stream_complete` | streams only; `stream_complete` = `[DONE]` / `message_stop` seen |
| `client_disconnected` | the client closed the connection before the proxy finished |
| `error` | `null` or `{source: upstream|upstream_stream|proxy|client, status?, type?, message?}` |
| `warnings` | e.g. body not JSON, so forwarded unchanged |
| `ts_start` / `mono_start` | request arrived at the proxy |
| `ts_first_byte` / `mono_first_byte` | first response byte from upstream |
| `ts_first_output` / `mono_first_output` | first generated token of any kind (thinking, text or tool call); streams only |
| `ts_first_content` / `mono_first_content` | first token that is not thinking |
| `ts_end` / `mono_end` | upstream response finished (or the call failed) |
| `ttfb_s`, `ttft_s`, `duration_s` | derived from `mono_*`: first byte, first output, end |
| `ttft_valid` | `true` when the first output was thinking or text; `false` when it was a tool call; `null` when not streamed or nothing was generated |
| `bodies` | paths of stored bodies, with `--log-bodies` |

Notes on reading the numbers:

- `ttfb_s` is a keepalive timer, not a time to first token. oMLX 0.7.1rc1 answers a non-streaming
  call with its whole body, or, once 2 s have passed, with a keepalive space first. So `ttfb_s` is
  min(`duration_s`, about 2.0 s): in the R3 scored run, the 1,094 calls of up to 2.01 s show
  `ttfb_s` = `duration_s`, and all 70 longer calls show 2.004 to 2.008 s. Streams start with a keepalive chunk or
  `event: ping`. Use `ttft_s` for time to first token; it is only available for streaming calls.
- oMLX emits tool calls only after generation has finished. When a response
  starts with a tool call (no thinking or text streamed first), `ttft_s` is
  about the whole call and `ttft_valid` is `false`; leave those calls out of
  first-token statistics.
- The OpenAI Python SDK (sync client) closes the connection right after
  `[DONE]`; such calls show `client_disconnected: true` with
  `stream_complete: true` and `error: null`. Check `error` and
  `stream_complete`, not `client_disconnected`.
- When a client disconnects mid-call, the proxy closes the upstream
  connection at once, so oMLX stops generating instead of keeping the GPU busy.
  After such an early close (including the SDK case above) the upstream
  connection is not reused, so the next call opens a new one.
- Relayed responses are always sent chunked (upstream `content-length` is
  dropped), so the record is on disk before the client sees the end of the
  response. Bytes are otherwise unchanged.
- Parsing, hashing images and rewriting a request happen before it is
  forwarded and are part of `duration_s`: well under 1 ms for a 20 KB
  request, about 30 ms for a 10 MB request with 30 screenshots.

## Privacy and bodies

By default records hold metadata only: no message text, no generated text,
no tool arguments, no image pixels, no secrets. Error messages are clipped,
and the `input` member of validation errors (which echoes request content)
is dropped.

With `--log-bodies`, the proxy also writes, per call, into `<run-dir>/bodies/`:

- `NNNNNN-request.json`: the request exactly as received;
- `NNNNNN-forwarded.json`: the body sent upstream, when the proxy changed it;
- `NNNNNN-response.{json,sse}`: the upstream response bytes as relayed;
- `NNNNNN-headers.json`: request and response headers, with
  `authorization`, `x-api-key` and cookies redacted.

These files are meant for replaying real agent request traces in speed
tests. `runs/` is git-ignored; keep it that way.

## Tests

```sh
uv run pytest
```

The tests start a fake oMLX and the proxy in-process with uvicorn on
OS-assigned ports (port 0), never 8500 or 8600-8799, and never load a model.
