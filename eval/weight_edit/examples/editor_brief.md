# Brief: edit a copy of an embedding model's weights to retrieve better (no training)

## The question
Can you make Qwen3-Embedding-0.6B better at retrieval for a personal assistant's memory by reading its weights and activations and editing the weights directly, with no gradient descent and no training?

Background: a recent post (rikiyat.github.io/blog/weight-space-programming) had a frontier model edit Qwen3.5-0.8B's weights to stop it looping, which raised math scores a lot.
- Its edits were a rank-one removal of one "repetition" direction from output matrices, `W' = W - a·u(uᵀW)`, or switching off one attention head.
- On a model with no such obstacle, editing gained nothing. The authors' reading: editing can remove an obstacle that hides an ability, but cannot add one.

So the useful first step is to find what, if anything, holds this model back on the dev retrieval tasks.

## What you have
The harness in `eval/weight_edit/`: read its `README.md` first and use `uv run wsp ...` from there.
- **Weights:** `wsp copy <arm>` makes a full copy of the original model under the edits folder (`WSP_EDITS`). Load and analyse tensors from any copy in Python on the CPU as much as you like (numpy/torch tensors, linear algebra).
- **Edits:** `wsp edit apply <arm> <edit.py>` writes an edit into that arm's copy and records a manifest. Use a new arm for each candidate, named `opus1-<short>`.
- **Forward passes: a budget of 15.** `wsp pass ...` is the only way to run the model. Each call costs one pass:
  - a dev-suite evaluation of an arm (the dev metrics of LongMemEval round retrieval and SciFact-train retrieval);
  - or an activation read (`acts`: last-token hidden states per layer for dev texts).
  - Base-model dev results are already cached and cost nothing; see the README for where they are.
  - `wsp budget` shows what is left.
- **Closed-form post-maps** (`wsp postmap ...`) on cached vectors are free. They are not weight edits, so they are not your result, but they can tell you what a weight edit should do.

## Exact commands (run from the harness folder; your environment already sets `WSP_AGENT_MODE=1` and `WSP_BUDGET_ACCOUNT=opus1`)
- `N=$(cat work/lead/N)` is the dev LongMemEval size used for the base results. Always pass `--dev-lme $N`, so that results compare with base.
- `uv run wsp copy opus1-<short>` and `uv run wsp edit apply opus1-<short> work/opus1/<edit>.py --arg K=V ...` (`--replace` to redo an arm's edit).
- `uv run wsp pass eval --arm opus1-<short> --dev-lme $N` costs one pass and prints the dev table; the result lands in `results/opus1-<short>/dev[.lmeN].json`.
- `uv run wsp pass acts --arm base --layers 20,24,28 --dev-lme $N --limit 512` costs one pass. Activations from the validity run are already in `state/acts/base/` (depths 20, 24, 26, 27, 28; 512 dev texts) and cost nothing to read.
- `uv run wsp compare base opus1-<short> --suite dev --dev-lme $N` is free: paired CI per task.
- `uv run wsp budget show opus1` shows the passes left (15 at the start, shared by all your arms).
- Already done and free to read:
  - base dev results: `results/base/`;
  - the validity run's post-maps (`results/base+pca*`, `results/base+wh*`);
  - four direction-removal arms `v-e*` (top direction removed from blocks >= 20/24/26/27 at a = 0.5-1.0), with comparisons in `work/lead/compare.log` and the log in `work/lead/validity.log`.
  - Start from what those show.
- A pass can take several minutes. Run it in the foreground; the Bash timeout is raised to 30 min.

## Goal
- Raise the dev **LongMemEval recall_any@5** over the base model.
- Do not lower dev **SciFact-train nDCG@10** by more than 0.5 points.
- A smaller, explainable edit beats a larger one at equal gain.
- "No edit helped" is a valid, useful answer if you show what you tried and what you saw.

## Rules
- **No training:**
  - no backward passes, autograd, optimizers, or iterative fitting of weights to a loss;
  - closed-form linear algebra is allowed (projections, PCA/SVD, least squares, means of activations).
- Run the model only through `wsp pass`. Do not load the model into torch yourself to run it.
- **Never look at the test suite:**
  - do not run `wsp eval --suite test`;
  - do not read the test questions or qrels: the LongMemEval stage-set questions, LoCoMo, NFCorpus, ArguAna, SciFact test.
  - The harness owner evaluates your final arm on the test suite afterwards.
- **Do not touch:**
  - the original model folder, the Hugging Face cache, the harness code (put your own scripts and notes in `work/opus1/`);
  - any network port;
  - any file outside the harness folder and the edits folder.
- **Keep printed output short:** summaries, top-k numbers, shapes. Your context costs money; never print whole tensors or long logs.
- Keep a running notebook in `work/opus1/notes.md` (what you tried, pass number, result, next idea), so the work survives a restart.

## Finish
When the passes are used, or you are confident you are done, write `work/opus1/final.md` with:
- the chosen arm (or "none");
- the edit in one paragraph and its manifest summary (tensors, entries changed);
- why you expected it to help;
- base vs edited dev metrics (all tasks);
- passes used;
- what you would try next.

Your final answer is that file's content.
