# Benchmark results

One directory per run, under the machine and the model it measured:

```
docs/benchmark/<machine>/<family>/<quant>/<suite>/<run-id>/
  summary.json      the row the results page shows (format below)
  ...               the run's other files and write-up, if any (what is published: last bullet below)
```

- `<machine>`: chip, GPU cores and memory, lower case, for example `m5max-40c-128gb`. Speed, memory
  headroom and timeouts depend on the machine, so results from different machines never share a table.
- `<family>`: the model family, lower case, for example `qwen3.8_27b`.
- `<quant>`: the checkpoint's quantization as served, for example `oq5`, `oq8` or `bf16`. oQ levels are
  mixed precision (oQ5 is about 5.5 bits per weight), so the directory names the oQ level, not a bit
  width.
- `<suite>`: `backend`, `agent_ability` or `memory`.

`scripts/build_docs.py` reads the `summary.json` directly inside each `<run-id>/` directory (a run's own
files may hold more of them deeper down; a shallower one is an error), checks it, and writes the site,
four pages and their assets under `docs/` (served by GitHub Pages). Run it after adding a run, and
commit the generated output with the run. Source lives in `site/`; `--check` verifies every generated
page and asset, including the data file.

## summary.json

```json
{
  "schema": 1,
  "suite": "agent_ability",
  "protocol": "example-v0",
  "title": "An example protocol: 12 Python exercises x 3 harnesses, one run each",
  "model": "Qwen3.8-27B oQ5 + MTP",
  "family": "qwen3.8_27b",
  "quant": "oq5",
  "run_id": "2026-10-22-example",
  "date": "2026-10-22",
  "reference": true,
  "system": {"machine": "m5max-40c-128gb", "macos_build": "26B5101f", "xcode_build": "27A9275",
             "server": "oMLX 0.7.1rc1 (57620a3f)"},
  "metrics": [{"name": "Total", "value": "32/36"}],
  "notes": "optional, one line",
  "details": "optional: the run's write-up, or its main data file, relative to this directory"
}
```

- Rows share a table only when `system.machine`, `suite` and `protocol` are equal. A protocol is a
  fixed setting (tasks, budgets, sampling rule, thinking, server settings); a change to any of them is a new
  protocol id, and its rows start a new table. Results from different protocols are never compared in one table.
- Within a protocol, every row has the same `title` and the same metric names in the same order.
- `value` is the display string. Keep raw numbers in the suite's own files next to it. Pass counts
  (`a/b`) are shown as a percentage with the original count; partial counts stay visibly partial.
- `reference` (optional) is true on reference-model rows; it is displayed as a reference tag.
- `notes` and full metrics appear in result details; `details` must be a file under docs/benchmark,
  and the site links that report alongside the summary and run evidence.
- All fields are strings, except `schema` (1), `system` (an object), `metrics` (a list of `name`/`value`
  strings) and `reference` (a boolean).
- The directory path must match `system.machine`, `family`, `quant`, `suite` and `run_id`.
- Nothing personal goes here: no absolute home paths, account names, message or mail content. The
  capability leg exports only allowlisted files (no request or response bodies); a backend run also
  publishes its manifest and raw speed records, whose prompts are synthetic.

## Additional public evidence

For `capability-1`, the site derives per-harness pass counts and median model time/calls from the
published `backend/<run-id>/capability/results.json`. It uses the runner's final-record and solved
accessors, excludes warm-ups, and verifies that totals match these summary rows.

For `ui-real-1`, the report's sibling `results.json` holds item-sample counts. The site reads
`gold.audited`, verifies the seven summary values, and exposes screenshot, screenshot + OCR, AX-list,
and read-only original-capture arms. Strict and lenient reads remain separate. Pages, questions,
profiles, captures and responses from the private set are not part of the generated website.

New machine/protocol groups are discovered automatically. A changed machine never shares a primary
table with another machine; a changed protocol never shares a table with another protocol. Generated
data includes an evidence fingerprint that is independent of the build time and Git commit.
