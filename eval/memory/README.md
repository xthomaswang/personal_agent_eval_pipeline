# memory

Does an agent keep a correct, current and compact memory? Two parts:

- **T4, memory consolidation** (`t4/`, `pab/`): a synthetic 10-day timeline of messages, emails and
  calendar entries is fed one day at a time, one agent session per day. Afterwards a fixed reader answers
  60 probe questions from the memory store alone, so the score is about what the agent wrote, not about
  how well it retrieves. Spec: `t4/task.yaml`.
- **R3, offline retrieval** (planned here; built in personal-agent-bench): recall@k and latency of
  localmelo's retrievers on recorded agent runs. No agent loop.

## T4

```
t4/scenario.py         the story: people, records, the gold memory items, the 60 probes
t4/gen.py              python -m memory.t4.gen --seed 1  ->  t4/fixtures/v1/ (--check verifies)
t4/fixtures/v1/        days/day01..10.jsonl, facts.yaml, probes.yaml, drop_list.yaml
pab/                   the bench's MCP server (stdio): feed_day, memory_*, submit
t4/reader.py           the fixed reader (one call per probe, thinking off, temperature 0)
t4/oracle.py           accuracy per category, fabricated_source, source_precision, duplicate_ratio, memory_tokens
t4/values.py           answer normalization and key-value extraction
```

**The timeline.** 46 story records over 2026-04-06..15 and 40 distractors placed by the seed (86 in
all). The story has updates (a rescheduled offsite and dentist, a corrected rent, a delayed delivery),
relative dates ("this Saturday", "two weeks from today"), cross-source links (an email and the calendar
entry for the same event), facts that only combine across days ("Jamie's sister" on day 1, her flight on
days 4 and 7), and near misses (Harbor Hall and Harbor Room; another flight number; a second concert
date). Probes, 12 per category:

| Category | What it needs |
|---|---|
| extraction | one fact from one record |
| cross_session | facts from records of different days, linked by a person, place or event |
| temporal | a relative date resolved against the record's date |
| knowledge_update | the latest value after a change |
| abstention | an answer the timeline never gives (the reader must say null) |

**Running one trial** (the runner that drives a harness over the 10 sessions is not built yet): each
session starts the harness with pab as its MCP server, `python -m memory.pab --run-dir <run> --session <n>`,
and the prompt in `t4/task.yaml`. Then:

```sh
uv run python -m memory.t4.reader --run-dir <run>     # needs oMLX, BENCH_MODEL and the GPU lock
uv run python -m memory.t4.oracle --run-dir <run>     # writes <run>/t4/oracle.json
```

**Checks that hold offline** (`uv run pytest memory/tests`): the fixtures regenerate byte for byte; the
gold store cites only records it can see, with source_precision 1 and no duplicates, on every day; a
scripted agent that writes the gold items through pab ends with exactly the gold store; citing a record
before it was fed is flagged.

**Scoring rules.** Accuracy is counted per probe and reported per category. The empty-memory filter
drops answerable probes that the reader gets right from an empty store; abstention probes are never
dropped, since an empty store answers them right by construction, and are reported apart. The headline
is the accuracy over the four answerable categories.
