# resume

## Current state (2026-10-26 evening)

The pipeline now lives in the personal_agent_eval_pipeline monorepo as its backend. Verified runs go to
`docs/benchmark/<family>/<quant>/backend/<run-id>/` at the repository root (the two runs below were moved
there); `modeleval publish` writes their results-page rows, `scripts/build_docs.py` builds the page. The
harness epoch snapshots `bench/` and agent_ability's `smoke/` + `shims/` from the monorepo (same trees, same
epoch `7582825160ad`); localmelo is the backend's own clone `vendor/localmelo`.

## State at 2026-10-26 17:15

**Done since 07:10**
- Reference capability finished: gate set 63/66, ranking set 61/66 (`docs/benchmark/qwen3.8_27b/oq5/backend/20261026-062952`).
- Nanbeige4.3-33B: speed leg complete; capability stopped by the owner after 40 of 66 gate runs; the
  thinking-on check was stopped too (too slow). Write-up: `docs/benchmark/nanbeige4.3_33b/oq5/backend/20261026-104645/README.md`.
  Verdict: worse than Qwen3.8-27B as a local agent backend (31/40 vs 39/40 on the same runs; 2.5-4x slower).
- Pipeline changes: two-level disturbance policy, sustained/frequent major rule (30 s or 25 % of a unit),
  foreign-GPU floor 250 ms, memory watchdog for agent and checker processes (8 GiB per process, 16 GiB per
  tree; a runaway agent script had driven macOS out of memory at 12:5x), exploratory arms
  (`--thinking on`, `--thinking-budget`, `--only`, `--harnesses`, all with `--legs capability --unverified`).

**Waiting for the owner:** whether to run Qwen3.8-Flash-Next next (~30 min timed + ~3 h capability).
localmuse_main runs Flash-Next screenshots (60) tonight after that; ping it when the GPU is free.

**Then:** public GitHub repo + PR for the owner to merge (redact check runs in the pre-commit hook).

## Previous state (2026-10-26 07:10)

**What this repo is.** The unified per-model evaluation pipeline the owner asked for on 2026-10-26
(speed and agent capability of a local model under one setting, capability through Claude Code,
Hermes and localmelo). DESIGN.md is the spec, PROTOCOL.md the fixed setting, README.md the how-to.

**Done**
- Research and design: `DESIGN.md` revision 2 after a four-lens review; the old campaign's cost
  analysis (51.8 machine-hours, ~30 % lost to whole-suite re-runs) motivates every change.
- Build: `src/modeleval`, the modified agentbench copy in `harness/` (`harness/CHANGES.md`),
  scripts (quietlib per-window judge, gpu_lock update-purpose, redact checks, pre-commit hook).
  Offline tests: 470 + 130 pass.
- Owner's instruction after the first live run: small disturbances are kept with a flag, not
  re-run (two-level policy, commit 646cab0).
- Nanbeige4.3-33B: BF16 downloaded (rev c1093c5), converted to `models/oq/Nanbeige4.3-33B-oQ5`
  (22 GiB, 126 s). oMLX doubled its BOS; fixed in the converted copy's tokenizer (DESIGN §4).
- Flash-Next load check (old repo, unverified): loads on 0.7.1rc1, PLE mmap, MTP on.
- Mechanics run on the reference (05:26, unverified) and the reference speed leg
  (`results/qwen3-8-27b-oq5/20261026-062952`, all 9 timed units clean).

**Running:** the reference's capability leg (resumed 07:04 after a harness-epoch drift: the agent
bench moved its localmelo worktree; the pipeline now uses its own clone `vendor/localmelo` at
3b8f310, epoch `7582825160ad`). ETA ~10:00.

**Queue**
1. Reference capability (gate set, then ranking set) — running.
2. localmuse_main's ~25 min smoke on 8500 after our release.
3. Nanbeige4.3-33B-oQ5: full run (10-min notice to localmelo_main before the timed block).
4. Qwen3.8-Flash-Next-oQ4e-mtp: full run.
5. `modeleval report`, redact check, public GitHub repo + PR for the owner to merge.

**Defaults stated to the owner, not yet confirmed:** the reference is re-run once in this pipeline
(old rows were image-mode with an older localmelo); capability runs with thinking off.

**Coordination:** GPU lock via `scripts/gpu_lock.sh`; purpose "... (timed)" makes localmelo_main's
agents stop, "... accuracy only (untimed)" lets them continue. 10-min notice before every timed block;
take/release messages to localmuse_main. Ports: 8703 only.
## Current state (2026-10-26 21:00)

**Where things are now.** The pipeline lives at `eval/backend` of the public repository
personal_agent_eval_pipeline (moved from ~/Developer/mlsys/experiments/agent-model-eval with its history).
Runs are published under `docs/benchmark/<machine>/<family>/<quant>/{backend,agent_ability}/<run-id>/`
(`modeleval publish`); machine paths come from `config.defaults.toml` plus a local `config.toml`.

**Harness epoch.** `ce99c7b5d0bd` (snapshot of this repository at b999407) is declared equivalent to
`7582825160ad`: the only difference is one reworded comment in `eval/bench/manifest.py` (history rewrite
before the repository went public); selftest verifies the diff is comments only.

**Running:** Qwen3.8-Flash-Next oQ4e, started 20:19 from the old checkout (agent-model-eval, epoch
7582825160ad). Speed leg done (all units clean or minor): 64K cold TTFT 31.5 s, cached TTFT 1.27 s, decode
85.6 / 78.2 / 74.5 / 80.1 tok/s at 16K / 32K / 64K / 110K, M4 headline 61.8 s, server peak 97 GiB.
Capability leg running (~3-4 h).

**Next:** import the Flash-Next run into docs/benchmark (m5max-40c-128gb/qwen3.8_flash_next/oq4e) and
publish; send localmuse_main the 12-exercise per-harness counts; release the GPU for its screenshots;
then the cleanup list for the owner (old agent-model-eval checkout, the old campaign's 363 GB oMLX SSD
cache, Nanbeige BF16 and oQ5, the old repo's bench/flash-next branch).

## Previous state (2026-10-26 17:15)