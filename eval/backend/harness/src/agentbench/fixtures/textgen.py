"""Deterministic generator of varied, non-repetitive technical text.

No external data: every word comes from the vocabularies below, combined by a small
grammar with random identifiers and numbers, so a 64K-token document has a high share of
unique token 8-grams (checked by the tests). The text is about fictional software
systems only (no people, no personal data).
"""

from __future__ import annotations

import random

_SYL = ["ka", "ren", "tor", "vel", "mi", "quo", "dra", "sen", "lu", "pha", "gri", "nox", "tal", "bry",
        "zen", "or", "ix", "ul", "mar", "ce", "fen", "dal", "ost", "ry", "ae", "vin", "kor", "sil", "tem",
        "hal", "bo", "nim", "ca", "pel", "tri", "sam", "ver", "gol", "ash", "um", "pri", "lex", "do", "fa"]

COMPONENTS = ["scheduler", "shard router", "write-ahead log", "compaction worker", "token bucket",
              "lease manager", "index builder", "snapshot service", "replication stream", "admission gate",
              "query planner", "cost model", "garbage collector", "block allocator", "page cache",
              "connection pool", "retry policy", "feature flag store", "metrics exporter", "trace sampler",
              "build cache", "artifact registry", "dependency resolver", "test runner", "linker pass",
              "register allocator", "kernel launcher", "prefetch queue", "tensor arena", "attention kernel",
              "tokenizer service", "rate limiter", "circuit breaker", "health checker", "config loader",
              "schema migrator", "bloom filter", "merge iterator", "commit pipeline", "event bus",
              "watchdog timer", "job queue", "load balancer", "DNS resolver", "TLS terminator",
              "object store gateway", "checkpoint writer", "state machine", "consensus module",
              "vector index", "embedding cache", "router table", "plugin host", "sandbox runner",
              "file watcher", "diff engine", "patch applier", "layout engine", "render thread",
              "input dispatcher", "font rasterizer", "compositor", "audit trail", "quota tracker"]

ADJ = ["incremental", "lock-free", "bounded", "lazy", "eager", "idempotent", "versioned", "sharded",
       "append-only", "copy-on-write", "speculative", "deterministic", "best-effort", "adaptive",
       "hierarchical", "pipelined", "vectorized", "tiered", "stateless", "content-addressed",
       "zero-copy", "batched", "streaming", "fault-tolerant", "monotonic", "pluggable", "coarse-grained",
       "fine-grained", "read-mostly", "write-heavy", "latency-sensitive", "memory-mapped", "persistent",
       "ephemeral", "asynchronous", "single-threaded", "NUMA-aware", "cache-friendly", "sparse", "dense"]

NOUNS = ["requests", "segments", "pages", "blocks", "records", "tokens", "partitions", "snapshots",
         "manifests", "checkpoints", "frames", "packets", "batches", "spans", "events", "commits",
         "leases", "watermarks", "offsets", "handles", "descriptors", "buffers", "rows", "columns",
         "keys", "chunks", "slots", "ranges", "workers", "replicas", "tasks", "jobs", "queries",
         "kernels", "tiles", "layers", "entries", "files", "symbols", "edges", "nodes", "deltas"]

VERBS_T = ["coalesces", "rewrites", "validates", "flushes", "evicts", "batches", "reorders", "pins",
           "shards", "compresses", "deduplicates", "rehydrates", "replays", "throttles", "prefetches",
           "serializes", "partitions", "merges", "tags", "checkpoints", "streams", "hashes", "indexes",
           "rebalances", "truncates", "materializes", "invalidates", "schedules", "drains", "probes",
           "annotates", "spills", "promotes", "demotes", "seals", "forwards", "mirrors", "audits"]

VERBS_I = ["stalls", "backs off", "converges", "degrades gracefully", "falls back", "warms up",
           "times out", "recovers", "saturates", "fragments", "drifts", "stabilizes", "retries",
           "yields", "spins", "blocks", "resumes", "diverges", "rolls back", "catches up"]

METRICS = [("p50 latency", "time"), ("p99 latency", "time"), ("tail latency", "time"),
           ("throughput", "rate"), ("hit rate", "pct"), ("miss rate", "pct"), ("queue depth", "count"),
           ("resident memory", "mem"), ("peak memory", "mem"), ("error budget", "pct"), ("CPU time", "time"),
           ("GPU occupancy", "pct"), ("write amplification", "ratio"), ("read amplification", "ratio"),
           ("space amplification", "ratio"), ("fan-out", "count"), ("jitter", "time"), ("startup time", "time"),
           ("compile time", "time"), ("cold-start latency", "time"), ("steady-state throughput", "rate"),
           ("lock contention", "pct"), ("decode rate", "rate"), ("index size", "mem")]

UNITS = {"time": [("ms", 1, 900), ("µs", 3, 950), ("s", 1, 120), ("ns", 40, 990)],
         "rate": [("requests/s", 50, 90000), ("tokens/s", 20, 4000), ("MiB/s", 10, 9000)],
         "pct": [("%", 1, 99)],
         "mem": [("MiB", 4, 900), ("GiB", 1, 96), ("KiB", 8, 960)],
         "count": [("entries", 16, 65536), ("connections", 8, 4096), ("items", 2, 900)],
         "ratio": [("x", 1, 40)]}

PREPS = ["before", "after", "while", "unless", "until", "whenever", "as soon as", "only if", "even when"]
CONNECT = ["As a result", "In practice", "By contrast", "For this reason", "In the common case",
           "Under sustained load", "On cold start", "During failover", "In the worst case",
           "Since the last release", "With the default settings", "On small inputs", "At larger scale",
           "In our measurements", "Historically", "After the refactor", "For compatibility",
           "To keep the hot path short", "When the cache is warm", "When the cache is cold"]

TOPICS = ["Design overview", "Failure handling", "Performance notes", "Configuration", "Migration plan",
          "Open questions", "Rollout strategy", "Benchmark results", "Known limitations", "Data layout",
          "Concurrency model", "Memory budget", "Observability", "Testing strategy", "Security review",
          "API surface", "Backpressure", "Recovery procedure", "Capacity planning", "Future work",
          "Compatibility matrix", "Error codes", "Upgrade path", "Cache policy", "Scheduling policy"]

LOG_MSGS = ["flushed {n} {noun} in {v}", "lease renewed for {id} (ttl {t}s)", "retrying {id} after {v}",
            "evicted {n} {noun} under pressure", "checkpoint {id} sealed at offset {n}",
            "slow path taken for {id}: {v}", "rebalanced {n} {noun} across {m} workers",
            "compaction of level {m} finished in {v}", "watermark advanced to {n}",
            "rejected {n} {noun}: queue full", "warm-up complete, {n} {noun} resident",
            "stale handle {id} dropped", "config reload applied ({m} keys changed)",
            "spilled {n} {noun} to disk ({mem})", "probe to {id} failed: connection reset"]

LANGS = ["python", "go", "rust", "sql", "yaml", "bash", "toml", "typescript"]


class TextGen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)

    # ------------------------------------------------------------------ atoms
    def pick(self, xs):
        return xs[self.r.randrange(len(xs))]

    def word(self, n_syl: int | None = None) -> str:
        return "".join(self.pick(_SYL) for _ in range(n_syl or self.r.randint(2, 3)))

    def ident(self, style: str = "snake") -> str:
        parts = [self.word(self.r.randint(1, 2)) for _ in range(self.r.randint(1, 3))]
        if style == "snake":
            return "_".join(parts)
        if style == "kebab":
            return "-".join(parts)
        if style == "camel":
            return parts[0] + "".join(p.capitalize() for p in parts[1:])
        return "".join(p.capitalize() for p in parts)

    def project(self) -> str:
        return self.word(self.r.randint(2, 3)).capitalize()

    def metric(self) -> tuple[str, str]:
        return self.pick(METRICS)

    def number(self, family: str | None = None) -> str:
        unit, lo, hi = self.pick(UNITS[family or self.pick(["time", "time", "mem", "rate", "count"])])
        v = self.r.randint(lo, hi)
        if unit == "x":
            return f"{v / 10 + 1:.1f}x"
        if unit in ("ms", "s", "GiB") and self.r.random() < 0.4:
            return f"{v + self.r.randint(0, 9) / 10:.1f} {unit}"
        return f"{v} {unit}"

    def comp(self) -> str:
        c = self.pick(COMPONENTS)
        roll = self.r.random()
        if roll < 0.25:
            return f"the {self.pick(ADJ)} {c}"
        if roll < 0.45:
            return f"the {c} in {self.project()}"
        if roll < 0.6:
            return f"{self.project()}'s {c}"
        if roll < 0.75:
            return f"the `{self.ident()}` {c}"
        return f"the {c}"

    def obj(self) -> str:
        roll = self.r.random()
        if roll < 0.35:
            return f"{self.pick(ADJ)} {self.pick(NOUNS)}"
        if roll < 0.55:
            return f"the {self.pick(NOUNS)} of {self.comp()}"
        if roll < 0.7:
            return f"up to {self.r.randint(2, 4096)} {self.pick(NOUNS)}"
        if roll < 0.85:
            return f"every `{self.ident('camel')}` {self.pick(NOUNS)[:-1]}"
        return f"{self.pick(NOUNS)} tagged `{self.ident('kebab')}`"

    # -------------------------------------------------------------- sentences
    def sentence(self) -> str:
        k = self.r.randrange(12)
        c, o = self.comp(), self.obj()
        if k == 0:
            s = f"{c} {self.pick(VERBS_T)} {o} {self.pick(PREPS)} {self.comp()} {self.pick(VERBS_I)}"
        elif k == 1:
            m, fam = self.metric()
            s = f"{self.pick(CONNECT)}, {c} {self.pick(VERBS_T)} {o} and keeps the {m} below {self.number(fam)}"
        elif k == 2:
            s = f"When {self.comp()} {self.pick(VERBS_I)}, {c} {self.pick(VERBS_T)} {o} instead"
        elif k == 3:
            m, fam = self.metric()
            unit = self.pick(UNITS[fam])
            a, b = self.r.randint(unit[1], unit[2]), self.r.randint(unit[1], unit[2])
            u = "x" if unit[0] == "x" else " " + unit[0]
            s = f"The {m} of {c} moved from {a}{u} to {b}{u} after `{self.ident()}` was enabled"
        elif k == 4:
            verb = self.pick(["flush", "evict", "retry", "block", "reorder", "drop", "resize", "split"])
            s = f"{c} must not {verb} {o} {self.pick(PREPS)} the {self.pick(NOUNS)} are durable"
        elif k == 5:
            s = f"Setting `{self.ident()}` to {self.r.randint(0, 512)} makes {c} {self.pick(['twice as aggressive', 'more conservative', 'roughly linear', 'strictly ordered', 'cheaper to restart', 'easier to profile'])}"
        elif k == 6:
            s = f"{c} {self.pick(VERBS_I)} {self.pick(PREPS)} {self.r.randint(2, 900)} {self.pick(NOUNS)} are pending, which {self.pick(['doubles', 'halves', 'hides', 'exposes', 'bounds', 'amortizes'])} the {self.metric()[0]}"
        elif k == 7:
            s = f"Unlike {self.comp()}, {c} {self.pick(VERBS_T)} {o} without taking the `{self.ident()}` lock"
        elif k == 8:
            m, fam = self.metric()
            s = (f"A {self.pick(ADJ)} variant of {c} was tried in {self.project()} {self.r.randint(1, 9)}."
                 f"{self.r.randint(0, 30)} and reverted because the {m} regressed by {self.number(fam)}")
        elif k == 9:
            s = f"Each {self.pick(NOUNS)[:-1]} carries a {self.r.choice([16, 32, 64, 128])}-bit `{self.ident()}` so that {c} can {self.pick(['detect reordering', 'skip duplicates', 'resume mid-stream', 'verify ownership', 'merge without copying', 'expire stale state'])}"
        elif k == 10:
            s = f"{self.pick(CONNECT)}, the {self.metric()[0]} is dominated by {o}, not by {self.comp()}"
        else:
            s = f"`{self.ident('camel')}()` {self.pick(VERBS_T)} {o} and returns a `{self.ident('pascal')}` handle"
        s = s[0].upper() + s[1:]
        return s + "."

    def paragraph(self) -> str:
        return " ".join(self.sentence() for _ in range(self.r.randint(3, 6)))

    # ----------------------------------------------------------------- blocks
    def code(self) -> str:
        lang = self.pick(LANGS)
        f, a, b, c = self.ident(), self.ident(), self.ident(), self.ident()
        n1, n2 = self.r.randint(2, 512), self.r.randint(1, 64)
        if lang == "python":
            body = (f"def {f}({a}, {b}={n2}):\n    \"\"\"{self.sentence()}\"\"\"\n"
                    f"    {c} = [x for x in {a} if x.{self.ident()} > {b}]\n"
                    f"    if len({c}) > {n1}:\n        raise {self.ident('pascal')}Error(\"{self.ident('kebab')} overflow\")\n"
                    f"    return {{k.{self.ident()}: k for k in {c}}}\n")
        elif lang == "go":
            body = (f"func {self.ident('pascal')}({a} []{self.ident('pascal')}, {b} int) (int, error) {{\n"
                    f"\tif len({a}) > {n1} {{\n\t\treturn 0, fmt.Errorf(\"{self.ident('kebab')}: %d > {n1}\", len({a}))\n\t}}\n"
                    f"\ttotal := 0\n\tfor _, v := range {a} {{\n\t\ttotal += v.{self.ident('pascal')} * {b}\n\t}}\n"
                    f"\treturn total + {n2}, nil\n}}\n")
        elif lang == "rust":
            body = (f"pub fn {f}({a}: &[{self.ident('pascal')}], {b}: usize) -> Result<Vec<u{self.r.choice([8, 16, 32, 64])}>, {self.ident('pascal')}Error> {{\n"
                    f"    let mut {c} = Vec::with_capacity({b} * {n2});\n"
                    f"    for item in {a}.iter().filter(|i| i.{self.ident()} < {n1}) {{\n        {c}.push(item.{self.ident()}());\n    }}\n"
                    f"    Ok({c})\n}}\n")
        elif lang == "sql":
            body = (f"SELECT {a}, COUNT(*) AS {b}, AVG({c}) AS avg_{c}\nFROM {self.ident()}\n"
                    f"WHERE {self.ident()} > {n1} AND {self.ident()} <> '{self.ident('kebab')}'\n"
                    f"GROUP BY {a}\nHAVING COUNT(*) >= {n2}\nORDER BY {b} DESC\nLIMIT {self.r.randint(5, 200)};\n")
        elif lang == "yaml":
            body = (f"{self.ident()}:\n  {a}: {n1}\n  {b}: \"{self.ident('kebab')}\"\n  {c}:\n"
                    f"    - name: {self.ident('kebab')}\n      weight: {self.r.randint(1, 100) / 100}\n"
                    f"    - name: {self.ident('kebab')}\n      weight: {self.r.randint(1, 100) / 100}\n"
                    f"  {self.ident()}: {self.r.choice(['true', 'false'])}\n")
        elif lang == "toml":
            body = (f"[{self.ident()}]\n{a} = {n1}\n{b} = \"{self.ident('kebab')}\"\n{c} = [{n2}, {n1 // 2}, {n2 * 3}]\n\n"
                    f"[{self.ident()}.{self.ident()}]\nenabled = {self.r.choice(['true', 'false'])}\nratio = {self.r.randint(1, 99) / 100}\n")
        elif lang == "typescript":
            body = (f"export async function {self.ident('camel')}({a}: {self.ident('pascal')}[], {b} = {n2}): Promise<number> {{\n"
                    f"  const {c} = {a}.filter((x) => x.{self.ident('camel')} > {b});\n"
                    f"  if ({c}.length > {n1}) throw new Error(\"{self.ident('kebab')}\");\n"
                    f"  return {c}.reduce((acc, x) => acc + x.{self.ident('camel')}, 0);\n}}\n")
        else:
            body = (f"$ {self.ident('kebab')} --{self.ident('kebab')} {n1} --{self.ident('kebab')}={self.ident()}\n"
                    f"{self.ident('kebab')}: {self.pick(LOG_MSGS).format(n=n1, noun=self.pick(NOUNS), v=self.number('time'), mem=self.number('mem'), id=self.ident('kebab'), t=n2, m=n2)}\n"
                    f"$ echo $?\n0\n")
        return f"```{lang}\n{body}```"

    def log(self) -> str:
        h, m, s = self.r.randint(0, 23), self.r.randint(0, 59), self.r.randint(0, 59)
        lines = []
        for _ in range(self.r.randint(5, 10)):
            s += self.r.randint(0, 9)
            m, s = m + s // 60, s % 60
            msg = self.pick(LOG_MSGS).format(n=self.r.randint(1, 99999), noun=self.pick(NOUNS), v=self.number("time"),
                                             mem=self.number("mem"), id=self.ident("kebab"), t=self.r.randint(1, 120),
                                             m=self.r.randint(2, 64))
            lines.append(f"{h:02d}:{m % 60:02d}:{s:02d}.{self.r.randint(0, 999):03d} "
                         f"{self.pick(['INFO', 'INFO', 'DEBUG', 'WARN', 'ERROR'])} [{self.ident()}] {msg}")
        return "```text\n" + "\n".join(lines) + "\n```"

    def table(self) -> str:
        (m1, f1), (m2, f2) = self.metric(), self.metric()
        cols = ["case", m1, m2, "notes"]
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _ in range(self.r.randint(3, 7)):
            rows.append(f"| `{self.ident('kebab')}` | {self.number(f1)} | {self.number(f2)} | {self.pick(ADJ)} {self.pick(NOUNS)} |")
        return "\n".join(rows)

    def bullets(self) -> str:
        items = []
        for _ in range(self.r.randint(3, 6)):
            s = self.sentence()
            items.append("- " + s)
        return "\n".join(items)

    def api(self) -> str:
        name = self.ident("camel")
        params = [(self.ident(), self.pick(["int", "str", "bool", "float", "bytes", "list[str]", "dict[str, int]"]))
                  for _ in range(self.r.randint(2, 4))]
        sig = ", ".join(f"{p}: {t}" for p, t in params)
        lines = [f"### `{name}({sig}) -> {self.ident('pascal')}`", "", self.sentence(), ""]
        for p, t in params:
            lines.append(f"- `{p}` ({t}): {self.sentence()}")
        lines.append(f"\nRaises `{self.ident('pascal')}Error` when {self.comp()} {self.pick(VERBS_I)}.")
        return "\n".join(lines)

    def qa(self) -> str:
        return (f"**Q:** Why does {self.comp()} {self.pick(VERBS_T).rstrip('s')} {self.obj()}?\n\n"
                f"**A:** {self.paragraph()}")

    def block(self) -> str:
        roll = self.r.random()
        if roll < 0.42:
            return self.paragraph()
        if roll < 0.58:
            return self.code()
        if roll < 0.66:
            return self.log()
        if roll < 0.74:
            return self.table()
        if roll < 0.84:
            return self.bullets()
        if roll < 0.93:
            return self.api()
        return self.qa()

    def section(self) -> str:
        title = f"## {self.pick(TOPICS)}: {self.project()} {self.pick(COMPONENTS)}"
        return "\n\n".join([title] + [self.block() for _ in range(self.r.randint(3, 6))])

    def document(self, min_chars: int) -> str:
        parts = [f"# {self.project()} engineering notes, revision {self.r.randint(10, 99)}"]
        size = len(parts[0])
        while size < min_chars:
            s = self.section()
            parts.append(s)
            size += len(s) + 2
        return "\n\n".join(parts)
