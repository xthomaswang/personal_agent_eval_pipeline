# personal_agent_eval_pipeline

Evaluates local models as the backend of a personal agent on a Mac with Apple silicon, in three suites:
serving speed and tool use (backend), agent ability through three harnesses (agent_ability), and memory
(memory; its session runner is not built yet, so it has no results).

**Results, architecture and method: https://xthomaswang.github.io/personal_agent_eval_pipeline/**

## Layout

```
eval/backend           per-model runs: speed leg + capability leg (start here)
eval/agent_ability     the harness task suite and its checkers
eval/memory            T4: memory consolidation over a synthetic 10-day timeline
eval/bench             shared: the recorder proxy and the harness adapters
eval/customize         your own private tasks (gitignored, never pushed)
docs/                  the site (index.html) and the results (benchmark/)
scripts/build_docs.py  rebuilds the site from docs/benchmark
```

## Quick start

Needs an Apple-silicon Mac, [uv](https://docs.astral.sh/uv/), and an [oMLX](https://github.com/jundot/omlx)
0.7.1rc1 checkout (commit 7a5a3d75) with its own .venv. The capability leg also needs the three harnesses, at
the versions in `eval/backend/pins.toml`, and the pinned task data
([eval/agent_ability/README.md](eval/agent_ability/README.md#data)).

```sh
cd eval/backend
# once per Mac: put the paths that differ from config.defaults.toml in config.toml
uv sync && uv sync --project envs/capability && uv sync --project envs/agent-tools
uv run modeleval selftest                # offline; a FAIL names what is missing
# once per model: add it to models.toml, then prepare its checkpoint
uv run modeleval download <model> && uv run modeleval convert <model>   # it has a prepare entry
uv run modeleval register <model>        # or: its checkpoint is already at its models.toml path
(cd harness && uv run agentbench fixtures build)   # once: the speed prompts (Qwen3.8 tokenizer)
uv run modeleval plan <model>
uv run modeleval run <model>             # speed leg + capability leg
uv run modeleval publish                 # rows under docs/benchmark/<machine>/...
python3 ../../scripts/build_docs.py      # rebuild the site
```

On a new machine, three steps are not in these commands (the site's
[Run it](https://xthomaswang.github.io/personal_agent_eval_pipeline/#run) section has them): preparing the
reference model by hand (it has no prepare entry), writing the harness epoch your Mac hashes to into
`eval/backend/pins.toml`, and adding protocol ids for your macOS/Xcode build and epoch to
`eval/backend/src/modeleval/publish.py`. [eval/backend/README.md](eval/backend/README.md#on-another-mac) has the
details; each suite's README has its own.

## Privacy

Only synthetic or public task data are committed: a backend run's manifest and speed records, and the
capability leg's allowlisted files (no request or response bodies). Agent transcripts, harness homes and server
logs stay in the gitignored `runs/` directories, and private tasks in the gitignored `eval/customize/`. Before
pushing, run
`/usr/bin/python3 -I -B eval/backend/scripts/redact.py --check <changed files>` from the repository root, or
install the hook that runs it on every commit: `ln -s ../../eval/backend/scripts/pre-commit .git/hooks/pre-commit`.
