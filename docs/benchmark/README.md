# Benchmark results

One directory per run, under the model it measured:

```
docs/benchmark/<family>/<quant>/<suite>/<run-id>/
  summary.json      the row the results page shows (format below)
  ...               the suite's allowlisted export and write-up, if any
```

- `<family>`: the model family, lower case, for example `qwen3.8_27b`.
- `<quant>`: the checkpoint's quantization as served, for example `oq5`, `oq8` or `bf16`. oQ levels are
  mixed precision (oQ5 is about 5.5 bits per weight), so the directory names the oQ level, not a bit
  width.
- `<suite>`: `backend`, `agent_ability` or `memory`.

`scripts/build_docs.py` reads every `summary.json`, checks it, and writes `docs/index.html` and the
results block of the top-level README. Run it after adding a run, and commit its output with the run.

## summary.json

```json
{
  "schema": 1,
  "suite": "agent_ability",
  "protocol": "aider12-v0",
  "title": "Aider, 12 Python exercises x 3 harnesses, one run each",
  "model": "Qwen3.8-27B oQ5 + MTP",
  "family": "qwen3.8_27b",
  "quant": "oq5",
  "run_id": "2026-10-22-quant-compare",
  "date": "2026-10-22",
  "system": {"macos_build": "26B5101f", "xcode_build": "27A9275", "server": "oMLX 0.7.1rc1 (57620a3f)"},
  "metrics": [{"name": "Total", "value": "32/36"}],
  "notes": "optional, one line",
  "details": "optional: a write-up, relative to this directory"
}
```

- Rows share a table only when `suite` and `protocol` are equal. A protocol is a fixed setting (tasks,
  budgets, sampling, thinking, server settings); a change to any of them is a new protocol id, and its
  rows start a new table. Results from different protocols are never compared in one table.
- Within a protocol, every row has the same `title` and the same metric names in the same order.
- `value` is the display string. Keep raw numbers in the suite's own files next to it.
- The directory path must match `family`, `quant`, `suite` and `run_id`.
- Nothing personal goes here: no absolute home paths, account names, message or mail content. Each suite
  exports only allowlisted fields.
