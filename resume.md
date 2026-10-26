# resume

## Current state (2026-10-26 17:15)

**Done since 07:10**
- Reference capability finished: gate set 63/66, ranking set 61/66 (`results/qwen3-8-27b-oq5/20261026-062952`).
- Nanbeige4.3-33B: speed leg complete; capability stopped by the owner after 40 of 66 gate runs; the
  thinking-on check was stopped too (too slow). Write-up: `findings/2026-10-26-nanbeige4.3-33b.md`.
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
