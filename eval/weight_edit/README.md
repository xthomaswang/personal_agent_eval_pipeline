# weight_edit: editing an embedding model's weights directly (no training)

Edit a copy of Qwen3-Embedding-0.6B's weights directly, with no training, and measure what that did to retrieval. The
edits are closed-form (projections, PCA, means of activations), written either by hand or by a frontier LLM acting as the
editing agent. Each edit is scored on a **dev** suite (what the editor may see) and a **held-out test** suite.

Inspired by rikiyat.github.io/blog/weight-space-programming (logs: github.com/RikiyaT/lm_understand-logs, exp8): a
rank-one removal of one direction from output matrices raised a model's scores by removing an obstacle that hid an ability.

## Results

Metric: nDCG@10 (the main metric of every task), with paired-bootstrap 95% CIs (10000 resamples) of edited minus base.
Environment of the runs: macOS 27.2 (build 26B123), Xcode 27.2 (build 27B52), MPS fp16.

**Dev** (40 LongMemEval-S questions + 300 SciFact-train queries):

| Arm | What | LongMemEval dev (n=40) | SciFact train (n=300) |
|---|---|---|---|
| base | the original model | 0.646 | 0.751 |
| `v-e24c1a05` | closed-form edit: half of the centered top residual direction removed from the o_proj / down_proj outputs of blocks 24-27 (8 tensors) | 0.671, +0.025 [+0.003, +0.052] | flat (-0.001 [-0.007, +0.004]) |
| `base+pca1` | post-map (not a weight edit): remove the top principal component of the document vectors | +0.033 [+0.006, +0.063] | -0.005 [-0.013, +0.003] |
| `v-e20c0a05` | a wrong edit: uncentered direction, from block 20 | -0.185 [-0.262, -0.114] | -0.036 [-0.053, -0.020] |

Other arms of the same run (dev, LongMemEval delta): `v-e26c1a10` +0.001, `v-e28c1a10` +0.014, pca2 +0.025, pca4 +0.037,
pca8 +0.030, pca16 +0.019 (SciFact -0.013, significant), whiten shrink 0.1/0.3/0.6: +0.000 / +0.024 / +0.033; only
the pca1 and pca4 LongMemEval CIs exclude zero; the whiten ones all include it.

**Held-out test**, `v-e24c1a05` minus base:

| Suite | n | base | edited | Delta [95% CI] |
|---|---|---|---|---|
| LongMemEval test | 102 | 0.6754 | 0.6794 | +0.0040 [+0.0004, +0.0085] |
| LoCoMo | 1536 | 0.4968 | 0.4996 | +0.0028 [+0.0003, +0.0054] |
| SciFact | 300 | 0.7005 | 0.6973 | -0.0032 [-0.0109, +0.0042], not significant |
| NFCorpus | 323 | 0.3625 | 0.3614 | -0.0012 [-0.0042, +0.0019], not significant |
| ArguAna | 1406 | 0.6674 | 0.6674 | 0.0000 [-0.0024, +0.0024], not significant |

The held-out gain is about 1/6 of the dev gain (+0.004 vs +0.025). The dev gain is inflated by selection: the best of 12
comparisons on 40 dev questions. The edit is real but small, and it hurts nothing held out.

The `pca1` post-map shows no significant transfer on the test suite (LongMemEval +0.0075 [-0.0024, +0.0181], LoCoMo
+0.0009, SciFact +0.0024, NFCorpus -0.0029) and loses on ArguAna (-0.0042 [-0.0081, -0.0004]).

**Harness vs official scores** (nDCG@10 on the test split, `reference_scores.json`): SciFact 0.7005 vs 0.6972, NFCorpus
0.3625 vs 0.3671, ArguAna 0.667 vs 0.710. ArguAna is likely low because of the 512-token document cap (the official run
used 8192). That is fine for paired comparisons, but do not quote it as an absolute score.

The numbers above were checked against the run logs; the result JSONs themselves are not published (see Pipeline).

## Editor models

| Editor model | Passes used | Budget | Cost (USD) | Wall time | Best dev arm | Dev delta | Test delta |
|---|---|---|---|---|---|---|---|
| None (hand-written: 4 fixed settings of the blog-style edit) | 6 dev passes (base at N=20 and N=40, 1 acts, 4 arms) | not budgeted | 0 (no API) | 33 min on MPS | `v-e24c1a05` | +0.025 (nDCG@10) | +0.0040 LongMemEval test, +0.0028 LoCoMo |
| Claude Opus 5.5 (effort xhigh, 2026-11-03) | 3 of 15 (1 acts read of all 15,195 dev texts at depths 26-28, 2 evals) | 15 passes, $18 cap | 4.39 | 33 min, 44 turns | `opus1-b81a6` | +0.016 nDCG@10 (r@5 +0.050 [0.000, +0.125]) | LongMemEval -0.0023 [-0.0086, +0.0030], LoCoMo +0.0019 [-0.0009, +0.0047]; NFCorpus -0.0030* and ArguAna -0.0026* |
| GPT-6 Astra (effort xhigh, 2026-11-03) | 7 of 15 | 15 passes | 0 (subscription plan, no API bill; 6.02M input tokens, 5.96M cached, 45k output) | 33 min | `astra1` | r@5 0.85 to 0.90 (dev LongMemEval) | LongMemEval +0.0018 [-0.0074, +0.0115], LoCoMo +0.0001, SciFact +0.0018, NFCorpus -0.0022, ArguAna -0.0007; none significant |

**Claude Opus 5.5's edit** (`opus1-b81a6`) changes one column of `layers.27.mlp.down_proj` (1022 of 3.1M entries). At the pooled token, MLP neuron 81 of block 27 fires almost the same way for every input, so it works as a bias channel. The edit uses it to subtract 0.6 times the mean embedding direction before the L2 norm: "all-but-the-top" centering, done inside the weights, in closed form from activation means. Its predicted dev numbers matched the measured ones (r@5 0.90, nDCG 0.664 predicted vs 0.662 measured). The dev gain rested on 2 questions that flipped and did not transfer; held out, it lost slightly on NFCorpus and ArguAna (* = CI excludes zero). It is the same pattern as the `pca1` post-map, which also centers on the dev mean. The hand-written `v-e24c1a05` remains the only edit with a significant held-out gain. Lesson: 40 dev questions are too few to select an edit on; a 2-question flip passed the gate. The next editor model keeps the same protocol so that the rows compare. Cost is the API bill of the editing run; passes are model runs charged to the shared budget.

**GPT-6 Astra's edit** (`astra1`) zeroes 32 of the 1024 entries of the final `norm.weight`: the coordinates with the largest absolute mean over the dev documents. It is the same idea as Opus's edit and `pca1`, removing a shared offset, done more coarsely. Its sandbox had no GPU access, so its forward passes ran through a pass service outside the sandbox, with the same budget. Held out it is flat everywhere: no gain, but none of Opus's small losses either.

A post-hoc variant that was not the agent's pick, rank-64 whitening of the final MLP (3.1M entries), splits by domain: LongMemEval +0.0364*, LoCoMo -0.0210*, ArguAna -0.0114*. A CPU control fits the same post-map in closed form on cached base vectors. Fitted on dev it gives LongMemEval +0.0589* and LoCoMo -0.0534*. Fitted per LoCoMo conversation, it gives LoCoMo +0.0244*. Fitted per LongMemEval haystack, it gives -0.011, not significant. So this is a transform fitted to one domain, not a general gain: across the three editors, the only significant held-out gain is still the hand-written `v-e24c1a05`, and it is small.

The editing agent gets `examples/editor_brief.md` (the task and its rules), a pass budget shared by all of its arms, and a
USD cap. The test suite is run by the harness owner on the agent's chosen arm only if that arm beat base on dev
LongMemEval recall_any@5 without lowering SciFact-train nDCG@10 by more than 0.5 points.

## Pipeline and how to reproduce

Needs an Apple-silicon Mac for the GPU passes (CPU smoke mode works anywhere), [uv](https://docs.astral.sh/uv/), a local
copy of Qwen3-Embedding-0.6B, and the LongMemEval-S and LoCoMo json files.

Stages: (1) copy the original; (2) a read pass dumps activations of dev texts; (3) `edit apply` runs an edit file on the
original weights and writes the copy; (4) budgeted passes score the copy on dev; (5) `compare` gives paired CIs; (6)
closed-form post-maps are fit on cached dev document vectors (free); (7) the test suite is run once, for the chosen arm.

```sh
cd eval/weight_edit
uv sync
export WSP_ORIGINAL=/path/to/Qwen3-Embedding-0.6B
export WSP_LME=/path/to/longmemeval_s_cleaned.json
export WSP_LOCOMO=/path/to/locomo10.json
export WSP_ALLOW_MPS=1                                   # opt in to the GPU
uv run pytest                                            # CPU only; tests needing the tokenizer need WSP_ORIGINAL
uv run wsp eval --arm base --suite dev --dev-lme 40      # base, cached per weights hash
uv run wsp acts --arm base --suite dev --layers 20,24,26,27,28 --limit 512 --dev-lme 40   # read pass
uv run wsp copy v-e24c1a05
uv run wsp edit apply v-e24c1a05 edits/example_remove_direction.py --arg layer=24 --arg a=0.5 --arg center=1
uv run wsp eval --arm v-e24c1a05 --suite dev --dev-lme 40
uv run wsp compare base v-e24c1a05 --suite dev --dev-lme 40
uv run wsp postmap fit --kind pca-remove --k 1 --arm base --dev-lme 40 --out state/postmaps/pca1.lme40.npz
uv run wsp eval --arm base --suite dev --dev-lme 40 --post state/postmaps/pca1.lme40.npz --label base+pca1
uv run wsp eval --arm base --suite test ; uv run wsp eval --arm v-e24c1a05 --suite test   # held-out, once
uv run wsp compare base v-e24c1a05 --suite test
```

`examples/validity.zsh` runs the whole dev-side recipe above (base, post-maps, read pass, four edits, comparisons), and
`examples/night_o1.zsh` runs one editing-agent session through the `EDITOR_CMD` hook, then the test suite if the chosen
arm beat base on dev. Both write logs under `work/lead/`.

Build of the recorded runs: `sw_vers` gave ProductVersion 27.2, BuildVersion 26B123; `xcodebuild -version` gave Xcode
27.2, build 27B52. Every result JSON records both.

Per-query results (`results/`) and caches (`state/`) are not committed (gitignored). Point `WSP_RESULTS` and
`WSP_STATE` at a folder to keep them elsewhere. Pointing `WSP_STATE` at an existing state reuses its cached base
vectors, so a rerun of an arm only encodes the edited model.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `WSP_ORIGINAL` | Qwen3-Embedding-0.6B folder (read-only, never written) | required |
| `WSP_EDITS` | folder for edited copies, one subfolder per arm | `models/edits` in this folder |
| `WSP_LME` | `longmemeval_s_cleaned.json` | required for LongMemEval suites |
| `WSP_LOCOMO` | `locomo10.json` | required for the LoCoMo suite |
| `WSP_STATE` | caches, activations, budgets, post-maps | `state/` in this folder |
| `WSP_RESULTS` | result JSONs | `results/` in this folder |
| `WSP_STAGE_DIR` | folder with `stage_set_v1.txt` / `pilot_set_v1.txt` | `../memory/public` (sibling `eval/memory/public`) |
| `HF_HOME` | Hugging Face cache holding the MTEB SciFact / NFCorpus / ArguAna sets | the Hugging Face default |
| `WSP_ALLOW_MPS` | `1` opts in to `--device mps` | unset |
| `WSP_GPU_OWNER` | owner name to look for in the first line of a local `gpu` booking tool (`now: bench by <owner>`) | unset |
| `NIGHT_DEADLINE` | set by a night runner for its queued job; also allows MPS | unset |
| `WSP_BUDGET_ACCOUNT` | charge all arms' passes to one shared budget account | the arm itself |
| `WSP_AGENT_MODE` | `1` for the editing agent (see rules below) | unset |

## Locations

| What | Where |
|---|---|
| Original model (read-only, never written) | `$WSP_ORIGINAL` |
| Edited copies (arms) | `$WSP_EDITS/<arm>/` (full copy + `edit_manifest.json`) |
| Vector cache | `state/cache/<weights_key>/<suite>.<task>.<text hash>.<dtype>.npy` (+ `.json`) |
| Activations ("read" pass) | `state/acts/<arm>/dev.L<depth>.npy`, `dev.meta.json` |
| Budgets and pass logs | `state/budget/<arm>.json`, `<arm>.log` (`<account>` when `WSP_BUDGET_ACCOUNT` is set) |
| Post-maps | `state/postmaps/*.npz` |
| Results | `results/<arm>[+post]/<suite>[.limitN][.lmeN].json` |
| Original's recorded hashes | `state/original_sha256.json` |
| Official task instructions | `qwen3_task_prompts.json` (from https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json) |
| Official reference scores | `reference_scores.json` (from github.com/embeddings-benchmark/results, URLs inside) |

`weights_key` is `base` for the original (and for a copied but unedited arm), otherwise the sha256 of the arm's edit
manifest without timestamps. Cache files also carry a hash of the encoded texts (so `--limit` and `--dev-lme` never
mix) and the dtype. Datasets are read-only: LongMemEval-S (`WSP_LME`) and LoCoMo (`WSP_LOCOMO`), and the MTEB
SciFact/NFCorpus/ArguAna sets in the HF cache (`HF_HOME`, pinned to the revisions the reference scores used).

## Model use

Last-token pooling over left-padded batches, L2-normalized. fp16 on MPS, fp32 on CPU. Queries are
`Instruct: {task}\nQuery:{text}` (the model card form); documents get no instruction, except ArguAna, whose official
Qwen3 prompt file defines a `passage` instruction that the official evaluation also puts on documents. Max length 512
tokens for documents, 128 for queries (the official MTEB run used 8192, so scores differ slightly). Texts are sorted by
length and batched by token budget; identical strings are embedded once.

## Suites

* **dev**: LongMemEval-S, `--dev-lme N` non-abstention questions (default 100; a seed-0 permutation of the eligible
  ids, so a smaller N is a prefix) that are not in `stage_set_v1.txt` / `pilot_set_v1.txt` and whose id appears in no
  file of that folder; plus 300 SciFact **train** queries (seed 0) against the SciFact corpus.
* **test** (never given to an editing agent): the non-abstention LongMemEval-S questions of `stage_set_v1.txt`; LoCoMo
  (all 10 conversations, categories 1-4, evidence ids normalized); SciFact, NFCorpus, ArguAna test.
* Dev and test LongMemEval ids are asserted disjoint (`wsp check-split`, also a test).

LongMemEval: per question, the corpus is every round of its haystack (a user turn plus the assistant reply after it,
as `User: ...\nAssistant: ...`); gold = rounds with a `has_answer` turn. Metrics: recall_any@5, recall_any@10,
nDCG@10, session_recall@5 (fraction of `answer_session_ids` covered by the sessions of the top-5 rounds) and
session_any@5. LoCoMo: per conversation, the corpus is every dialogue turn (`Speaker: text`, with the image caption
appended when there is one); gold = `evidence` turns; metrics recall@5, recall@10, nDCG@10. MTEB tasks: nDCG@10,
recall@100, MRR@10 (title and text joined by a space; ArguAna drops the query's own document). The main metric of every
task (used by `compare`) is nDCG@10; `compare --metric` picks another.

`wsp stats --suite dev|test` prints sizes and token counts (no model); the numbers are saved in `results/suite_stats.*.json`.

## Commands

| Command | What it does |
|---|---|
| `wsp copy <arm>` | Copy the original to `edits/<arm>/`; refuses an existing arm. |
| `wsp verify-original [--record]` | Hash the original's files and compare with `state/original_sha256.json`. Exit 1 on a mismatch. |
| `wsp eval --arm base\|<arm> --suite dev\|test [--device mps\|cpu] [--limit N] [--dev-lme N] [--post pm.npz]` | Score the arm, write `results/...json`, print a table with the wall time (MTEB tasks show the official reference next to ours). |
| `wsp postmap fit --kind pca-remove --k K` or `--kind whiten --shrink S` `[--arm base]` | Fit on the cached dev **document** vectors only. `wsp eval --post` applies it to queries and documents, re-normalizes, and scores from cached vectors (no model run). |
| `wsp acts --arm A --suite dev --layers 12,16,20 --limit N` | The "read" pass: dump last-token residual activations of N sampled dev texts. Depth k = k blocks applied (0 = embeddings, 28 = last block output before the final norm). |
| `wsp edit apply <arm> <edit.py> [--arg K=V] [--replace]` | Run `edit(tensors, read_acts)` on the CPU from the ORIGINAL weights and write the result into the arm's `model.safetensors` in the original dtype (bf16); writes `edit_manifest.json`. Rejects unknown tensors, shape changes, non-finite values, and edits that change nothing after rounding. |
| `wsp budget set <arm> N` / `show <arm>` | Forward-pass budget of an arm. |
| `wsp pass eval\|acts --arm A ...` | The budgeted wrapper: like `eval --suite dev` / `acts`, but takes one pass off `state/budget/<arm>.json`, refuses at 0, never runs `--suite test`, logs every pass (with wall time) to `<arm>.log`. A refused call costs nothing; a pass that fails after starting still counts. |
| `wsp compare <A> <B> --suite dev\|test` | Paired per-query differences with a bootstrap 95% CI (10000 resamples) for the main metric of each task; A and B are result labels (`base`, `myarm`, `myarm+post`). |
| `wsp stats --suite dev\|test` | Sizes and token counts. |
| `wsp check-split` | Assert dev/test disjointness. |

**Devices.** `--device mps` is refused unless `WSP_ALLOW_MPS=1`, or `NIGHT_DEADLINE` is set (a night runner sets it for
its queued job), or `WSP_GPU_OWNER` is set and the first line of a local `gpu` booking tool shows `now: bench by <owner>`. `--device cpu` needs `--limit` of at most 64.
`--limit N` is smoke mode: N queries per task and, per corpus, only the gold documents plus a few distractors; its
metrics are not comparable with full runs (results go to `...limitN.json`). A pass on MPS shortens with
`--dev-lme N` (the SciFact part is fixed at about 1.65M tokens; LongMemEval dev is about 87k tokens per question).
Long encodes save partial progress every two minutes and resume when rerun with the same arguments.

**Edit files.** A python file with `edit(tensors, read_acts) -> dict`. `tensors` maps safetensors names (for example
`layers.20.mlp.down_proj.weight`) to float32 numpy copies of the ORIGINAL weights; return only the tensors you changed
(same shape). `read_acts(layer, arm="base")` returns the dumped activations `(n_texts, 1024)` of a depth. The module
global `ARGS` holds the `--arg KEY=VALUE` strings. See `edits/example_remove_direction.py` (top principal direction of
the residual stream at a depth, removed from the `o_proj` and `down_proj` outputs of later blocks:
`W' = W - a u (u^T W)`).

```
uv run wsp pass acts --arm base --layers 16 --limit 256
uv run wsp copy demo
uv run wsp edit apply demo edits/example_remove_direction.py --arg layer=16 --arg a=0.3
uv run wsp eval --arm base --suite dev --device mps ; uv run wsp eval --arm demo --suite dev --device mps
uv run wsp compare base demo --suite dev
```

## Rules for an editing agent

1. Set `WSP_AGENT_MODE=1`. Then `wsp eval` only accepts `--suite dev --post`, `wsp acts` is refused, and `wsp compare` is dev only.
2. Model runs go through `wsp pass ...` only, on the dev suite, and cost budget. The test suite is never run and its numbers are never shown.
3. Edit only arms under `edits/`. Never write to the original model folder, the dataset files, `state/budget`, or `results/` of other arms.
4. Closed-form work (post-maps, `edit apply`, reading the cached dev vectors) costs no pass.
5. Do not read stage/pilot id files or the LongMemEval stage set; do not tune on `lme_test`, `locomo`, `scifact`, `nfcorpus`, `arguana`.
6. SciFact dev (train queries) and SciFact test share one corpus, so dev-fitted maps see test documents (not their labels).
