# Qwen3.8-Flash-Next against Qwen3.8-27B as a local agent backend (2026-10-26)

Setting: protocol 1 (the backend's DESIGN.md and PROTOCOL.md), MacBook Pro M5 Max 128 GB, macOS 27.2
(26B5101f), Xcode 27.1 (27A9275), served by oMLX 0.7.1rc1 (7a5a3d75). The checkpoint is the oQ4e build
published by the oMLX author (`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` at e878710), built with oMLX 0.7.3.dev1:
4.775 bits per weight, 100 GiB. MTP on; the PLE table (29.8 GiB) is mmapped from the SSD rather than resident.
Thinking off, the model's agent sampling (temperature 1.0, top_p 0.95, top_k 20). Harness epoch 7582825160ad,
the same as the reference run. Runs:
[Qwen3.8-27B oQ5 + MTP 20261026-062952](../../../../qwen3.8_27b/oq5/backend/20261026-062952/) (the reference)
and this directory (Qwen3.8-Flash-Next oQ4e + MTP 20261026-201922). Both legs are complete.

**Caveat: the quantization differs.** The reference is plain oQ5 (about 5.5 bits per weight). Flash-Next has
no published oQ5 build and its BF16 is about 250 GB, so this run uses the closest published build, oQ4e.
Every difference below mixes the model with the quantization.

## Speed

| | Flash-Next oQ4e | Qwen3.8-27B oQ5 | Flash-Next / 27B |
|---|---|---|---|
| Cold 64K prompt, time to first token | 31.5 s | 111.4 s | 0.28x (3.5x faster) |
| Cached-prefix time to first token (M1) | 1.27 s | 3.20 s | 0.40x (2.5x faster) |
| Decode at 16K / 32K / 64K / 110K | 85.6 / 78.2 / 74.5 / 80.1 tok/s | 55.1 / 51.0 / 41.9 / 36.9 tok/s | 1.55 / 1.53 / 1.78 / 2.17x |
| Decode at 16K, thinking on | 73.6 tok/s | 44.5 tok/s | 1.65x |
| 20-step trajectory (M4 headline) | 61.8 s | 133.4 s | 0.46x (2.2x faster) |
| Peak server memory | 97 GiB | 58 GiB | 1.67x |

The 64K cold prompt, all four decode cells and the thinking-on decode were kept with a minor disturbance flag
(DESIGN §5.4); no unit needed a repeat.

## Capability (thinking off)

| | Flash-Next oQ4e | Qwen3.8-27B oQ5 |
|---|---|---|
| Gate set: Claude Code / Hermes / localmelo (22 each) | 20 / 22 / 22 | 21 / 20 / 22 |
| Gate set, runs solved | 64/66 | 63/66 |
| Ranking set: Claude Code / Hermes / localmelo (22 each) | 21 / 21 / 21 | 18 / 21 / 22 |
| Ranking set, runs solved | 63/66 | 61/66 |
| M7 tool-call cases (5 trials each) | 20/22 | 21/22 |
| Wall time, summed over the runs both solved (gate 61 / ranking 58) | 18 / 23 min | 45 / 76 min |

- Per-task sign test (the backend's RESULTS.md; a task's score is the number of harnesses that solved it).
  Gate set: Flash-Next ahead on 3 tasks, behind on 2, 17 tied, p = 1.000; decision **not worse**. Ranking
  set: ahead on 5, behind on 3, 14 tied, p = 0.727; **no detectable difference**.
- Time per solved run is 0.40x the reference's on the gate set and 0.30x on the ranking set, with fewer steps
  per run (6.4 against 7.4 and 9.7).
- M7 failures: Flash-Next m7-05-array-strings and m7-16-commit-message; the reference m7-07-enum.

## Memory

The speed costs memory. The server's footprint peaks at 97 GiB against 58 GiB (74 GiB right after load,
against 24 GiB), and the 29.8 GiB PLE table is file-backed, outside that footprint. The run recorded 164
swap-outs during the timed units; the reference recorded none. On a 128 GB Mac this leaves much less room for the agents' own
processes or a second model.

## Screenshot reading (a separate measurement, not part of capability-1)

This protocol is text-only, so vision is not in the tables above. Another session measured screenshot
reading separately on 2026-10-26: 12 synthetic UI screenshots x 5 trials (seed 20261021, about 1,320 prompt
tokens each), served by oMLX 0.7.1rc1 with the tool-image patch (57620a3f). Flash-Next oQ4e read 47/60
correctly, Qwen3.8-27B oQ5 54/60 (Qwen3.8-27B oQ4 / oQ6 / oQ8, measured 2026-10-21/22: 54 / 56 / 54). All
misses are 0/O or upper/lower-case confusions in small or low-contrast monospace text (for example `a0SO`
read as `a0S0`).

## Conclusion

On this Mac, Qwen3.8-Flash-Next oQ4e is the faster agent backend: 1.5-2.2x the decode rate, 2.5-3.5x faster
to the first token, and less than half the time for the 20-step trajectory, with no detectable difference in
how many agent tasks it solves. The costs are 1.7x the server memory and weaker reading of small text in
screenshots, and the comparison is across quantizations (oQ4e against oQ5).
