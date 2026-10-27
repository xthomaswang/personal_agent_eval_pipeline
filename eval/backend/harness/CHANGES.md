# Changes against qwen3.8-27b_20261017-2258_benchmark@c15a0a8

`harness/` was copied byte for byte from `qwen3.8-27b_20261017-2258_benchmark` at `c15a0a8` (checked
with `git archive c15a0a8 harness` and `diff -r` on 2026-10-26). These are the changes made since, for
agent-model-eval (DESIGN.md §5.1, §5.3, §10). Every new option defaults to the c15a0a8 behaviour, so
the old tests pass unchanged; `tests/test_model_eval_changes.py` covers each change.

## Gate (`suites/gate.py`)

- **Probes run on their own.** `Gate.g1`, `g2`, `g2b`, `g3`, `g4` and `g5` now return their item.
  The new `run_probes(cfg, [(name, kwargs), ...], fx=None, suite="gate")` runs a chosen subset and
  writes one raw file and a summary. It never writes `capabilities.md` or the `.md` fragment.
  *Why:* `modeleval` runs G1, G3, G4 in the functional block and G5 as a timed unit. `run()` always
  sends every probe and upserts `capabilities.md`.
- **Vision split.** `g3(vision: bool = True)`: with `vision=False`, the image block and the
  tool_result image probes (and their control) are not sent. They appear as N/A, and G3 passes on
  text + `tool_use`. The item records `vision`. `RunConfig.vision` (default True) makes `run()`
  skip G2 and G2b as well: their items are `N/A`, with `rejected=False`, so run-all's automatic
  text-only switch stays off. On the CLI, `--no-vision` also selects `--text-only` for the metric
  suites. *Why:* the old G3 failed every text-only model (Nanbeige) on its image parts, and the
  gate always sent images.
- **G5 options.** `g5(prompt: Literal["cold", "m3-off"] = "cold", nonce: str | None = None)`.
  `"m3-off"` sends the 64,000-token thinking-off M3 prompt through M3's own request path
  (`Target.chat(thinking=False)`), so its body is byte-identical to an M3 trial with the same
  nonce. The item records `prompt`, `nonce`, `metrics` (TTFT, e2e, prompt and cached tokens) and
  `peak_tree_bytes`. *Why:* DESIGN §5.1 "G5 + M3 64K" reuses G5's cold prefill as the warm-up of
  the 64K cell.

## Requests (`suites/common.py`)

- **`RunConfig.cold_nonce: Literal["tool", "system+tool"] = "tool"`.** With `"system+tool"`, the
  same fresh nonce goes at the start of the system message (`nonce_line`, `[run <digits>]\n`) and
  at the start of the first tool's description. This applies to the M1 cold trial and to every
  step of each M4 run, so the steps of a run still share their prefix (tested over all 21 steps
  with the Qwen3.8 template). `Fixtures.agent_system(nonce=None)`,
  `m1.trial_messages(..., system_nonce=None)` and `m4.step_messages(..., system_nonce=None)` take
  the nonce. Invalid values are refused before any file is written. *Why:* Qwen renders tools
  before the system prompt, Nanbeige the other way round. With the tool nonce alone, Nanbeige's
  5.7K-token system prompt was a cache hit in "cold" requests.
- **Cold checks.** The M1 summary has `cold_nonce` and `cold_checks`
  (`[{trial, cached_tokens, prompt_tokens, ok}]`). The M4 summary has
  `cold_checks: [{run, cached_tokens, prompt_tokens}]`, from step 0 of each run. Cold lines carry
  `nonce_at`. *Why:* the caller enforces `cached_tokens < 256` (one cache block) on these.
- **`RunConfig.anthropic_sampling: bool = False`.** When true, `temperature`, `top_p` and `top_k`
  from `extra_body` are merged into `/v1/messages` bodies. Other `extra_body` fields still go to
  OpenAI bodies only. *Why:* G3 ran at the server's global defaults (top_k 0), not at the model's
  agent sampling (DESIGN §2).
- **`RunConfig.tokenizer_dir: Path | None = None`.** Local fallback counts use this tokenizer.
  These are the completion tokens of a stream without usage, and `prompt.local_tokens` /
  `prompt_tokens_local`. If the directory does not load, they stay empty
  (`completion_tokens_source: null`, `prompt.local_tokens_error`). There is never a silent
  fallback to Qwen's tokenizer. `None` keeps the Qwen3.8 resolution. `Target.count_tokens` returns
  `int | None`; `Target.local` is the tokenizer in use. *Why:* cut M4 steps have no usage, and
  their counts came from Qwen's tokenizer for every model.
- **Thinking without an effort.** `Target.chat(..., omit_effort=False)` and
  `thinking_fields(..., omit_effort=False)`: with thinking on, only
  `chat_template_kwargs.enable_thinking=true` is sent (styles `hermes` and `kwargs`; `{}` for
  `effort` and `none`).
- **Header.** `mode` adds `vision`, `cold_nonce`, `anthropic_sampling` and `local_tokenizer`
  (directory, kind, and whether it loaded, with the error).
- **Unit window.** `Writer` records `t_start` (epoch seconds) after writing the header. It stamps
  `t_start` and `t_end` on every `summary` line, in place, so the summary dict a suite returns
  carries the same window. *Why:* each M3 cell, M4 run and G5 is a separately judged unit
  (DESIGN §5.4). The summary stays the last line of the file.
- `Outcome.metrics()` adds `decode_s` (`t_last - ttft`).
- **Direct tool calls.** `Outcome.metrics()` adds `server_ttft_s` and `server_generation_s` (oMLX's
  `usage.time_to_first_token` and `usage.generation_duration`, null for servers without them),
  and `ttft_eff_s` / `ttft_source` from `metrics.ttft`. *Why:* oMLX sends a tool call as one delta
  only after generating all of it. For a response that opens with a call (`first_kind:
  "tool_call"`), the client's TTFT is the end of the call's decode and `decode_s` is about 0, so
  M1 read a slower prefill and M4 dropped the step from R (old campaign raw data: client 1.85 s
  against the server's 1.31 s for a 32-token call).

## Suites

- **`suites/__init__.py`.** `run_suite` validates its options first: it raises `ValueError` with
  `validate`'s messages and creates no file. It documents the single-unit use: one M3 cell
  (`contexts=[c], caps=[150], no_thinking_bucket=True`, or `contexts=[], think_contexts=[c],
  think_caps=[1000]`), or one M4 run (`runs=1`). `validate` checks the new M3 options.
  `reuse_nonce` needs exactly one thinking-off context, 12 digits and no `bust_cache`.
  `warmup_max_tokens` must be at least 1.
- **M3 (`suites/m3.py`, `suites/opts.py`).**
  - `SuiteOpts.warmup_max_tokens: int | None = None`. `None` keeps the warm-up at `min(caps)`.
    *Why:* the thinking cell's warm-up decoded 1000 tokens; DESIGN uses 16.
  - `SuiteOpts.reuse_nonce: str | None = None`. The thinking-off cell is sent with this nonce and
    without a warm-up. If trial 0 reports a cached prefix (`cached_tokens` at least
    `REUSE_MIN_SHARE` = 0.9 of its prompt), the trials continue. Otherwise that request is
    written as the warm-up (`warmup: "fallback"`) and all trials follow it. The cell's `warmup`
    records `mode` (`normal` / `reused` / `fallback`), `max_tokens`, and trial 0's cached and
    prompt tokens.
  - `SuiteOpts.think_effort: str | None = "xhigh"`. `None` sends `enable_thinking` only, with no
    `reasoning_effort` anywhere, so each template renders its own default. CLI:
    `--no-think-effort`, `--warmup-max-tokens`, `--reuse-nonce`.
  - Warm-up lines record their nonce. Cells add `prompt_tokens` and `warmup`. The summary adds
    `think_effort` and `reuse_nonce`.
- **M4 (`suites/m4.py`).**
  - A stream cut inside a tool call (`incomplete_tool_call`) after its first token is a latency
    sample. Runs with cut steps are kept.
  - Every step line has `derived`: `latency_sample`, `cap_hit` (`length` / `incomplete_tool_call`
    / null), `decode_s`, `completion_tokens` and `completion_tokens_source`.
  - `per_step` uses latency samples (as `summarize` already did) and counts `cap_hits`. A
    successful stream without any token (no TTFT) is not a latency sample: it is listed in
    `failed_steps`, and its run has no `ttft_sum_s` or `headline_s`.
  - The summary adds `per_run` (`run_stats`): `ttft_sum_s` (null unless every step is a latency
    sample), `e2e_total_s`, `cap_hits`, `cap_hits_by_kind`, `mean_completion_tokens`,
    `usage_steps`, `usage_completion_tokens`, `usage_decode_s`, `decode_rate_tps` (R = usage
    completion tokens / usage decode time), `headline_s` = `ttft_sum_s + n_steps × 100 / R`,
    `failed_steps`, and step 0's cached and prompt tokens. It also adds per-run lists
    `ttft_sum_s`, `headline_s` and `decode_rate_tps`, the totals `cap_hits`, `step_requests` and
    `invalid_steps`, and `fixed_tokens_per_step` (100), `headline_formula`, `cold_nonce` and
    `cold_checks`. `errors` keeps its c15a0a8 meaning (requests with `ok=false`, cut steps
    included).
  - *Why:* DESIGN §5.1 M4 reporting. The old suite left cut steps out of per-step TTFT, and its
    total rewarded terse models. R follows the DESIGN's definition, tokens / (t_last - ttft); it
    is not the per-request (n-1)/span rate.
  - Direct tool calls: `run_stats` takes TTFT from `metrics.ttft` (the server's for a step that
    opens with a call) and the step's decode time from `metrics.decode_time` (the server's
    `generation_duration` for such a step), so these steps count in R. It adds `direct_calls` and
    `ttft_includes_decode` (direct calls without server timing: their client TTFT is kept but
    listed). `step_derived` adds `ttft_s`, `ttft_source`, `decode_time_s` and `direct_tool_call`;
    `per_step` TTFT uses `metrics.ttft`.
- **M1 (`suites/m1.py`).** The cold-nonce rule and `cold_checks`, as above. The summary's TTFT uses
  `metrics.ttft` and adds `ttft_sources` (counts per source).

## Metrics (`metrics.py`, `summarize.py`)

- `CAP_TRUNCATED` and `cap_truncated` moved from `summarize.py` to `metrics.py`, so the suites can
  use them; `summarize` re-exports both, and its tables are unchanged. New: `latency_sample(m)` and
  `cap_hit(m)`, and for direct tool calls `server_timing(usage)`, `direct_tool_call(m)`, `ttft(m)`
  (value and source: `client`, `server`, or `client-incl-decode` when a direct call has no server
  timing) and `decode_time(m)`.
- `decode_metrics` accepts a counter that returns `None`. The count and its source then stay
  empty. It also returns `decode_s`.

## Fixtures (`fixtures/build.py`, `fixtures/store.py`)

- `EXTRA_PROMPTS = [("m3-off", 110_000)]`. `fixtures build` writes
  `generated/prompts/m3-off-110000.json` (marked `outside_manifest`). It stays out of
  `manifest.json` and `sizes`, so the committed fixtures and `fixtures check` stay byte-identical
  to c15a0a8. `build(..., extras=True)`; `check()` rebuilds without extras. The build's return
  value (not the manifest file) lists `extra_prompts`. *Why:* DESIGN §5.1/§5.3: the 110K M3 cell
  (about 114K Nanbeige tokens, inside 131072). It is sized in Qwen3.8 tokens like every fixture.
  Fitting it takes about 1.5 s.
- `Fixtures.prompt()` caches a prompt that it generated on the fly under `generated/prompts/`
  (atomic write, `outside_manifest: true`). It still uses the Qwen3.8 tokenizer resolution.
  *Why:* the next process, or a timed unit, then loads the file instead of fitting it again.

## Tokenizer (`tokenizer.py`)

- `load_model_prompt(dir) -> (QwenPrompt | None, error | None)` loads a checkpoint's tokenizer,
  cached per directory. It needs `tokenizer.json`. The template comes from `chat_template.jinja`,
  else from `tokenizer_config.json`'s `chat_template` (a string, or the `default` entry of a
  list), as transformers resolves it. `preprocessor_config.json` is optional. `bos_token`,
  `eos_token`, `unk_token` and `pad_token` are passed to the template. `QwenPrompt` takes optional
  `template` / `template_vars`, and the Jinja environment gains HF's `strftime_now`. The default
  Qwen3.8 path renders exactly as before. *Why:* Nanbeige keeps its template in
  `tokenizer_config.json` and ships no preprocessor config, so the strict loader would fail.

- **No built-in machine path.** `snapshot_dir()` resolves `$AGENTBENCH_TOKENIZER_DIR` (which must
  hold all three files), then an HF snapshot of `Qwen/Qwen3.8-27B` or
  `mlx-community/Qwen3.8-27B-8bit` under `$HF_HOME/hub`. The c15a0a8 second step, the oQ5
  checkpoint at `~/Developer/mlsys/models/oq/Qwen3.8-27B-oQ5-mtp` (`OQ5_DIR`), is gone.
  `modeleval` sets the variable, when unset, to its registry's reference checkpoint at startup
  (when that directory has the files), and `modeleval fixtures` builds and checks the fixtures
  with it. The tokenizer is the same files on the owner's machine, so the fixtures stay
  byte-identical. *Why:* without the variable the harness looked for the checkpoint only in the
  owner's home layout; on another Mac, or with the models elsewhere (config `[paths] models`),
  runs and fixture builds failed unless an HF snapshot happened to be cached.

## CLI (`cli.py`)

New options: `--no-vision`, `--cold-nonce tool|system+tool`, `--anthropic-sampling`,
`--tokenizer-dir`, `--no-think-effort`, `--warmup-max-tokens`, `--reuse-nonce`.
`fixtures build` also prints the sizes of the extra prompts.

## Mock (`mock_smart.py`, tests only)

New `Behavior` knobs, all off by default:

- `system_first`: the prefix cache sees the system message before the tools, as with Nanbeige.
- `default_call`: a tool call for tool-bearing requests that have no canned answer.
- `cut_tool_calls`: as oMLX does, a tool call cut by `max_tokens` ends with an
  `incomplete_tool_call` error event and no usage.
- `report_usage`: `False` sends no usage chunk.
- `server_timing`: usage carries `time_to_first_token` and `generation_duration`, as oMLX's does.

Logged requests record their `sampling` fields.

## Tests

- `tests/test_model_eval_changes.py` (new, 27 tests) covers every change above, including the CLI
  options, `summarize` on new M4 files and the tokenizer resolution.
- The fixture and template tests skip without a resolvable tokenizer: set
  `$AGENTBENCH_TOKENIZER_DIR` to the reference checkpoint before `uv run pytest`.
- `tests/test_interference.py`: the 12 tests of `scripts/interference.py` and `scripts/run_m8.sh`
  skip when those scripts are missing. agent-model-eval did not copy the M8 tooling (M8 is
  dropped by default). The 5 quietlib tests skip when `scripts/quietlib.py` is agent-model-eval's
  modified copy (detected by its `MONITOR_COLUMNS`): they cover the c15a0a8 version, whose
  `verify` command and judge rules the copy replaces (DESIGN §5.4). Tests of the new quietlib
  belong with it, outside `harness/`.

## Docs

`README.md` describes the options, the in-process API and the new summary fields. `STATUS.md` (the
old campaign's note) points here.
