# Harness status (2026-10-18)

The harness is complete. Nothing is committed (review first). Nothing has run against a real
model server: every test uses mock servers on ephemeral ports.

## Done

- **Client layer** (previous session): async streaming OpenAI + Anthropic clients, TTFT by the
  METHODOLOGY rule, usage and cached tokens, SSE parsing, decode metrics, provenance, memory
  sampler, concurrency helper. Small additions in this session: `reasoning_fields` (which field
  carried reasoning: `reasoning_content`, `reasoning`, `inline_think`, `thinking_block`),
  `t_first_data`, `messages(add_model=False)`, a final memory sample when a sampler exits, cached
  provenance calls.
- **Fixtures** (`agentbench fixtures build|check`, `src/agentbench/fixtures/`): deterministic and
  seeded; committed text in `../fixtures/`, binaries and large prompts in `../fixtures/generated/`
  (gitignored); `manifest.json` holds the sha256 of every file and the measured sizes. See the
  README for the size table (prefix 14,905 tokens; cold prompts exactly 16,000 / 32,000 / 64,000).
- **Profiles** (`profiles.py`): `mlxlm`, `llamacpp`, `omlx`, `lithos`, `vllm`, `mlc`, `generic`.
  `mlc` calls tools through structured output (`structured.py`): no `tools` field, schemas
  inlined into the system message, flattened tool history, `response_format` constrained by
  xgrammar, JSON reply parsed back into `tool_calls`; records carry `tool_call_mode`.
- **Suites** (`src/agentbench/suites/`): `gate`, `m1`–`m7`, `run-all`; `--quick`, `--text-only`,
  configurable trials and sizes; JSONL with the provenance header first.
- **M7 checker** (`m7check.py`): matchers, jsonschema validation, one-to-one matching of parallel
  calls, alternatives, click point in pixels or 0–1000 normalized coordinates.
- **`summarize`** (`summarize.py`): median (min-max) tables per suite and cell, M4 per step;
  markdown upsert between markers into `summary.md` / `capabilities.md`.
- **Mock** (`mock_smart.py`): framework behaviour presets used by the tests and by
  `agentbench mock-server --behavior`.
- **Quiet-machine preflight** (METHODOLOGY §3, added 01:48): `--preflight` runs
  `scripts/quiet_check.sh` (read-only) before each suite, records its output in the header, and
  routes the run to `raw/_unverified/` when the machine is not quiet; `summary.md` ignores it.
- **Tests**: `uv run pytest -q` → **82 passed** (16 original + 66 new) in about 70 s: fixture
  sizes and invariants (exact prefixes, nonce placement and count invariance, text-equivalent
  token parity, determinism via `fixtures check`, committed manifest in sync with the generator),
  profile shaping (unit and end to end through the gate, per framework behaviour), every suite in
  `--quick` mode end to end, lithos 429 / N/A / server-log cached tokens, `run-all` argument
  validation and auto text-only, reserved ports, M7 scoring, `summarize`, and the `mlc`
  structured-output path (request shape, step schema, parsing, end to end, rejected
  `json_schema` shape), and the preflight routing to `raw/_unverified/`.

## Design decisions

- With `tools`, the cache-busting nonce goes at the start of the first tool's description (the
  Qwen template renders tools before the system prompt); without tools, at the very start.
  Nonces and request ids are digits only, so substitution never changes token counts.
- M1 warm-up = prefix + task, ending with a user message (mlx-lm checkpoints only then); each
  trial is an exact extension of it. M4 canned turns render with the same empty `<think>` block as
  the generation prompt, so step k is an exact prefix of step k+1. Both are asserted in tests.
- TTFT is not redefined for tool-only replies. `first_kind` and e2e are recorded, because oMLX and
  mlx-lm emit a tool call as one delta when it is complete (TTFT ≈ e2e there).
- M3 output is a long JSON array in `content` with no `tools`, so chunk timing is meaningful on
  servers that hold back tool-call markup. Thinking-on uses `xhigh`, sent both top-level and in
  `chat_template_kwargs` (mlx-lm ignores the top-level field; vLLM, oMLX and llama.cpp merge the
  two without conflict, checked in their sources).
- Screenshots go in a user message after the tool result by default (`--screenshot-placement tool`
  puts them inside the tool message). Text-only equivalents have exactly the image's 1282 tokens.
- The gate always probes images (G2, G3), even with `--text-only`; `run-all` switches later suites
  to text-only automatically only when the server rejects G2's image with an HTTP error (an
  accepted-but-misread image is still measured with images). `run-all` without `--spec-label`
  labels M3 rows `off` and says so; standalone `m3` requires the flag.
- Sizes use decimal K (64K = 64,000).

## Verify on the first live run

1. **Model load in the first request.** oMLX loads the model on the first request (17 s in the
   smoke test). The gate's first probe (G1) and each suite's warm-up absorb it; G1's TTFT in the
   gate table is not a latency number.
2. **oMLX sampling defaults.** The other session reports that oMLX 0.7 applies
   `generation_config` defaults (`top_k 20`). Check that request-level `temperature: 0` is honoured.
3. **Cached-token fields.** Confirm `cached_source` in the raw records: `usage.prompt_tokens_details`
   (vLLM, oMLX, mlx-lm), `timings.cache_n` or usage (llama.cpp), `server-log` (lithos, needs
   `--server-log logs/<lithos log>`).
4. **M7 click case.** Records whether coordinates came back in pixels or 0–1000 normalized units;
   check which convention Qwen3.8 uses before reading too much into a failure.
5. **lithos M5.** Which request wins the lock is racy; 429s are recorded per role, not retried.
   lithos caches at most about 46K tokens, so M3 at 64K is a cold multi-minute prefill per trial.
6. **Timeouts.** Read timeout 1800 s, total 3600 s. Raise `--read-timeout` if a 64K cold prefill on
   lithos-metal exceeds 30 min.
7. **MLC structured output may crash the engine.** `mlc-llm/env.md` §5 says *don't use
   `response_format`*: grammar jump-forward rolls tokens back through `PopN`, which the hybrid
   model's RNN state cannot do (§6.1), and the engine loop has no try/catch. The `mlc` profile
   uses it anyway, as requested. On the first live run, try the G1 probe alone and watch the log;
   if the engine dies, rerun with `--structured-format none` (instruction only, still parsed and
   validated) and record which mode was used. MLC must run with `--prefix-cache-mode disable
   --mode interactive`, so M1/M4 measure cold prefills and M5's concurrent requests are serialized
   or rejected (both are recorded per role, never retried); cached tokens come from `usage.extra`
   (`--enable-debug`) and are always 0. The grammar schema (all 44 tools, descriptions stripped,
   about 20-30 KB) is sent with every tools request; the string is identical across requests and
   runs (the nonce lives in a description), so MLC compiles it once, which may make the first G1
   slower. A changed tool set changes the string.
8. **Core AI** (port 8700 as well) has no profile: `--profile generic --framework coreai` works for
   basic probes; it streams reasoning as `content` (split client-side at `</think>`) and sends
   no usage chunk.
