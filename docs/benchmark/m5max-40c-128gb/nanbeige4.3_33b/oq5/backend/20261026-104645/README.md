# Nanbeige4.3-33B against Qwen3.8-27B as a local agent backend (2026-10-26)

Setting: protocol 1 (the backend's DESIGN.md and PROTOCOL.md), MacBook Pro M5 Max 128 GB, macOS 27.2
(26B5101f), Xcode 27.1 (27A9275), oMLX 0.7.1rc1. Both models oQ5; Qwen3.8-27B with MTP, Nanbeige without speculative decoding
(it ships none). Thinking off, each model's agent sampling. Runs:
[Qwen3.8-27B 20261026-062952](../../../../qwen3.8_27b/oq5/backend/20261026-062952/) and this directory
(Nanbeige4.3-33B 20261026-104645).

**Status of the Nanbeige run.** The speed leg is complete (every unit clean or minor). The capability leg
was stopped by the owner after 40 of the 66 gate-set runs (Claude Code 22/22, Hermes 18/22, localmelo 0/22)
because the gap was already clear and the remaining runs would have taken many more hours. The backend's
RESULTS.md therefore lists Nanbeige's speed rows but no capability row; the numbers below compare the same 40
(harness, task) pairs.

## Speed

| | Nanbeige4.3-33B | Qwen3.8-27B | Nanbeige / Qwen |
|---|---|---|---|
| Cold 64K prompt, time to first token | 210.6 s | 111.4 s | 1.9x slower |
| Cached-prefix time to first token (M1) | 5.05 s | 3.20 s | 1.6x slower |
| Decode at 16K / 32K / 64K / 110K | 15.9 / 13.5 / 10.5 / 9.0 tok/s | 55.1 / 51.0 / 41.9 / 36.9 tok/s | 0.24-0.29x |
| 20-step trajectory (M4 headline) | 328.5 s | 133.4 s | 2.5x slower |

Why: decode on this machine is bound by memory bandwidth, and per token Nanbeige reads its 23.6 GB of
weights plus a KV cache of 208 KiB per token in every one of its 52 full-attention layers (22 GiB at 110K);
Qwen3.8-27B reads 19 GB plus 64 KiB per token (16 of its 64 layers use full attention) and drafts several
tokens per step with MTP. Both models run at an effective ~420 GB/s; the difference is how much each token
costs.

## Capability (thinking off, the same 40 runs)

| | Nanbeige solved | Qwen solved | Nanbeige time | Qwen time |
|---|---|---|---|---|
| Claude Code, Aider (12) | 6 | 11 | 86 min | 13 min |
| Claude Code, MCPMark easy (10) | 10 | 10 | 17 min | 5 min |
| Hermes, Aider (8) | 5 | 8 | 108 min | 8 min |
| Hermes, MCPMark easy (10) | 10 | 10 | 20 min | 6 min |
| Total | 31/40 | 39/40 | 3.9 h | 0.5 h |

- Runs solved by only one model: Qwen 9, Nanbeige 1. Per task (summed over the harnesses run): Qwen ahead
  on 7 tasks, Nanbeige on 1, 14 tied; two-sided sign test p = 0.07.
- The gap is all in coding (Aider); both solve every MCPMark filesystem task.
- M7 tool-call cases (22, 5 trials each): Nanbeige 17 pass, Qwen 21.
- One Nanbeige run hit the 3600 s wall (Hermes, variable-length-quantity, 26 steps): at 60-70K tokens of
  context it decodes at about 10 tok/s.

## Thinking on (exploratory, not completed)

The model card's results use the highest thinking effort, so a thinking-on re-run of the failed Aider
exercises was started (`--thinking on`, results in the backend's gitignored `results/_unverified/`). Unbounded, one
Claude Code step on `transpose` reasoned for 30 minutes (about 27K tokens at 15 tok/s). With oMLX's thinking
budget at 4096 tokens per block, the same exercise had not finished after 15 minutes. The owner stopped it:
at this speed, thinking on is not usable for agent work on this machine, whatever it does for accuracy.

## Conclusion

On this Mac, as a local agent backend, Nanbeige4.3-33B is worse than Qwen3.8-27B: weaker at coding with
thinking off, unusably slow with thinking on, and 2.5-4x slower in every speed measure. Its published
scores (highest thinking effort, server GPUs) do not carry over to this setting.
