"""Retrieval suites: dev (an editing agent may see it) and test (held out).

dev  = LongMemEval-S (N non-abstention questions, seed 0, not in the stage/pilot sets, id not mentioned in the stage
       folder) + SciFact train queries (300, seed 0) against the SciFact corpus.
test = LongMemEval-S stage-set questions + LoCoMo (categories 1-4) + SciFact / NFCorpus / ArguAna test.

A Task holds Groups. A Group is one corpus with its queries (a LongMemEval question haystack, a LoCoMo conversation, or
a whole MTEB corpus). Texts are encoded once per task: identical strings are embedded once.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import util

LME_INSTRUCTION = "Given a question about the user's past conversations, retrieve the conversation turns that answer it"
SEED = 0
DEV_SCIFACT_QUERIES = 300

MTEB_REPOS = {
    "SciFact": ("mteb/scifact", "0228b52cf27578f30900b9e5271d331663a030d7"),
    "NFCorpus": ("mteb/nfcorpus", "ec0fa4fe99da2ff19ca1214b7966684033a58814"),
    "ArguAna": ("mteb/arguana", "c22ab2a51041ffd869aaddef7af8d8215647e41a"),
}

PROMPTS_FILE = util.PKG_ROOT / "qwen3_task_prompts.json"
PROMPTS_URL = "https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json"


@dataclass
class Group:
    qids: list[str]
    qtexts: list[str]
    dids: list[str]
    dtexts: list[str]
    qrels: list[dict[int, float]]  # per query: doc index (in this group) -> grade > 0
    doc_session: list[str] | None = None  # LongMemEval: session id of each doc
    gold_sessions: list[set[str]] | None = None  # LongMemEval: answer sessions per query
    ignore_identical: bool = False  # drop doc == query id from the ranking (ArguAna)


@dataclass
class Task:
    name: str
    kind: str  # mteb | lme | locomo
    q_instruction: str | None
    d_instruction: str | None
    groups: list[Group]
    meta: dict[str, Any] = field(default_factory=dict)
    limited: bool = False

    def format_query(self, text: str) -> str:
        return format_instruct(self.q_instruction, text)

    def format_doc(self, text: str) -> str:
        return format_instruct(self.d_instruction, text) if self.d_instruction else text

    def n_queries(self) -> int:
        return sum(len(g.qids) for g in self.groups)

    def n_docs(self) -> int:
        return sum(len(g.dids) for g in self.groups)

    def strings(self) -> tuple[list[str], list[str]]:
        """(unique query strings, unique doc strings) as the model sees them, in first-seen order."""
        qs: dict[str, None] = {}
        ds: dict[str, None] = {}
        for g in self.groups:
            for t in g.qtexts:
                qs.setdefault(self.format_query(t), None)
            for t in g.dtexts:
                ds.setdefault(self.format_doc(t), None)
        return list(qs), list(ds)

    def text_hash(self) -> str:
        qs, ds = self.strings()
        return util.sha256_texts(qs + ["\x00docs\x00"] + ds)


def format_instruct(instruction: str | None, text: str) -> str:
    return f"Instruct: {instruction}\nQuery:{text}" if instruction else text


# --- LongMemEval-S ---


def rounds_of_session(turns: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """Split a session into rounds (a user turn plus the assistant reply after it). Returns (text, has_answer).

    A user turn starts a round; an assistant turn joins the round before it (several in a row all join); an assistant
    turn with no round before it is a round of its own. Blank turns are skipped."""
    rounds: list[dict[str, Any]] = []
    for t in turns:
        content = str(t.get("content") or "").strip()
        if not content:
            continue
        has = bool(t.get("has_answer"))
        if t.get("role") == "user":
            rounds.append({"parts": [f"User: {content}"], "has": has})
        elif rounds:
            rounds[-1]["parts"].append(f"Assistant: {content}")
            rounds[-1]["has"] = rounds[-1]["has"] or has
        else:
            rounds.append({"parts": [f"Assistant: {content}"], "has": has})
    return [("\n".join(r["parts"]), r["has"]) for r in rounds]


def lme_group(q: dict[str, Any]) -> Group | None:
    dids: list[str] = []
    dtexts: list[str] = []
    sessions: list[str] = []
    gold: dict[int, float] = {}
    for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"], strict=True):
        for i, (text, has) in enumerate(rounds_of_session(turns)):
            if has:
                gold[len(dids)] = 1.0
            dids.append(f"{sid}#{i}")
            dtexts.append(text)
            sessions.append(str(sid))
    if not gold:
        return None
    return Group(
        qids=[str(q["question_id"])],
        qtexts=[str(q["question"])],
        dids=dids,
        dtexts=dtexts,
        qrels=[gold],
        doc_session=sessions,
        gold_sessions=[{str(s) for s in q["answer_session_ids"]}],
    )


def read_id_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def stage_ids() -> list[str]:
    return read_id_file(util.stage_dir() / "stage_set_v1.txt")


def pilot_ids() -> list[str]:
    return read_id_file(util.stage_dir() / "pilot_set_v1.txt")


def mentioned_ids(ids: list[str], folder: Path) -> set[str]:
    """Ids whose string appears in any file of the folder (text or compiled), searched recursively."""
    blobs = [p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()]
    return {i for i in ids if any(i.encode() in b for b in blobs)}


def is_abstention(qid: str) -> bool:
    return qid.endswith("_abs")


def lme_dev_ids(questions: list[dict[str, Any]], n: int) -> list[str]:
    """Dev question ids: a seed-0 permutation of the eligible non-abstention ids, first n (so a smaller n is a prefix)."""
    all_ids = [str(q["question_id"]) for q in questions if not is_abstention(str(q["question_id"]))]
    banned = set(stage_ids()) | set(pilot_ids()) | mentioned_ids(all_ids, util.stage_dir())
    eligible = sorted(i for i in all_ids if i not in banned)
    order = random.Random(SEED).sample(eligible, len(eligible))
    return order[:n]


def lme_test_ids() -> list[str]:
    return sorted(i for i in stage_ids() if not is_abstention(i))


def load_lme(ids: list[str], name: str) -> Task:
    questions = json.loads(util.lme_path().read_text())
    by_id = {str(q["question_id"]): q for q in questions}
    groups = []
    skipped = []
    for i in ids:
        g = lme_group(by_id[i])
        if g is None:
            skipped.append(i)
        else:
            groups.append(g)
    return Task(name, "lme", LME_INSTRUCTION, None, groups, meta={"question_ids": [g.qids[0] for g in groups], "skipped_no_gold": skipped})


# --- LoCoMo ---

_DIA_RE = re.compile(r"D:?(\d+):(\d+)")


def parse_evidence(raw: str) -> list[str]:
    """Normalize a LoCoMo evidence string to dia_ids ('D3:7'). Handles several ids in one string and odd spellings."""
    return [f"D{int(a)}:{int(b)}" for a, b in _DIA_RE.findall(raw)]


def load_locomo() -> Task:
    data = json.loads(util.locomo_path().read_text())
    groups = []
    for conv in data:
        c = conv["conversation"]
        sessions = sorted((k for k, v in c.items() if re.fullmatch(r"session_\d+", k) and isinstance(v, list)), key=lambda k: int(k.split("_")[1]))
        dids, dtexts = [], []
        for s in sessions:
            for t in c[s]:
                text = f"{t['speaker']}: {t['text']}"
                if t.get("blip_caption"):
                    text += f" [shared a photo: {t['blip_caption']}]"
                dids.append(str(t["dia_id"]))
                dtexts.append(text)
        index = {d: i for i, d in enumerate(dids)}
        qids, qtexts, qrels = [], [], []
        for n, qa in enumerate(conv["qa"]):
            if int(qa["category"]) not in (1, 2, 3, 4):
                continue
            gold = {index[d]: 1.0 for e in qa.get("evidence", []) for d in parse_evidence(e) if d in index}
            if not gold:
                continue
            qids.append(f"{conv['sample_id']}#{n}")
            qtexts.append(str(qa["question"]))
            qrels.append(gold)
        groups.append(Group(qids, qtexts, dids, dtexts, qrels))
    return Task("locomo", "locomo", LME_INSTRUCTION, None, groups)


# --- MTEB retrieval sets ---


def _mteb_file(task: str, rel: str) -> Path:
    from huggingface_hub import hf_hub_download

    repo, rev = MTEB_REPOS[task]
    try:
        return Path(hf_hub_download(repo, rel, repo_type="dataset", revision=rev, local_files_only=True))
    except Exception:  # not cached yet: download into the HF cache
        return Path(hf_hub_download(repo, rel, repo_type="dataset", revision=rev))


def mteb_instructions(task: str) -> tuple[str, str | None]:
    """(query instruction, doc instruction) from the official Qwen3-Embedding file. Symmetric tasks (ArguAna) carry a
    'passage' instruction that the official evaluation also puts on documents; asymmetric tasks give documents none."""
    spec = json.loads(PROMPTS_FILE.read_text())[task]
    if isinstance(spec, dict):
        return spec["query"], spec.get("passage") or None
    return spec, None


def read_qrels(task: str, split: str) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for ln in _mteb_file(task, f"qrels/{split}.tsv").read_text().splitlines()[1:]:
        if not ln.strip():
            continue
        q, d, s = ln.split("\t")
        if float(s) > 0:
            qrels.setdefault(q, {})[d] = float(s)
    return qrels


def load_mteb(task: str, split: str, name: str, query_sample: int | None = None) -> Task:
    corpus = [json.loads(ln) for ln in _mteb_file(task, "corpus.jsonl").read_text().splitlines() if ln.strip()]
    dids = [str(d["_id"]) for d in corpus]
    dtexts = [(str(d.get("title") or "") + " " + str(d["text"])).strip() for d in corpus]
    index = {d: i for i, d in enumerate(dids)}
    qrels_all = read_qrels(task, split)
    queries = [json.loads(ln) for ln in _mteb_file(task, "queries.jsonl").read_text().splitlines() if ln.strip()]
    queries = [q for q in queries if str(q["_id"]) in qrels_all]
    if query_sample is not None and len(queries) > query_sample:
        ids = sorted(str(q["_id"]) for q in queries)
        keep = set(random.Random(SEED).sample(ids, query_sample))
        queries = [q for q in queries if str(q["_id"]) in keep]
    qrels = [{index[d]: s for d, s in qrels_all[str(q["_id"])].items() if d in index} for q in queries]
    group = Group(
        qids=[str(q["_id"]) for q in queries],
        qtexts=[str(q["text"]) for q in queries],
        dids=dids,
        dtexts=dtexts,
        qrels=qrels,
        ignore_identical=(task == "ArguAna"),
    )
    qi, di = mteb_instructions(task)
    return Task(name, "mteb", qi, di, [group], meta={"mteb_task": task, "split": split, "dataset": MTEB_REPOS[task][0], "revision": MTEB_REPOS[task][1]})


# --- suites ---


def load_suite(name: str, dev_lme: int = util.DEFAULT_DEV_LME) -> list[Task]:
    if name == "dev":
        questions = json.loads(util.lme_path().read_text())
        ids = lme_dev_ids(questions, dev_lme)
        del questions
        tasks = [load_lme(ids, "lme_dev"), load_mteb("SciFact", "train", "scifact_train", DEV_SCIFACT_QUERIES)]
    elif name == "test":
        tasks = [
            load_lme(lme_test_ids(), "lme_test"),
            load_locomo(),
            load_mteb("SciFact", "test", "scifact"),
            load_mteb("NFCorpus", "test", "nfcorpus"),
            load_mteb("ArguAna", "test", "arguana"),
        ]
    else:
        raise ValueError(f"unknown suite {name!r} (dev or test)")
    return tasks


def assert_split_disjoint(dev_lme: int = util.DEFAULT_DEV_LME) -> None:
    """LongMemEval ids of dev and test never overlap, dev ids are outside stage/pilot, SciFact dev queries are not test queries."""
    questions = json.loads(util.lme_path().read_text())
    dev = set(lme_dev_ids(questions, dev_lme))
    test = set(lme_test_ids())
    assert not dev & test, f"dev/test overlap: {sorted(dev & test)}"
    assert not dev & (set(stage_ids()) | set(pilot_ids())), "dev overlaps stage/pilot"
    sf_test = set(read_qrels("SciFact", "test"))
    sf_dev = set(read_qrels("SciFact", "train"))
    assert not sf_dev & sf_test, "SciFact train/test queries overlap"


# --- limit mode (smoke runs) ---


def limited(task: Task, limit: int, base_distractors: int = 6) -> Task:
    """A small version of a task: `limit` queries (seeded), and per group only the gold docs of those queries plus a
    seeded handful of distractors. For smoke runs only; metrics are not comparable with the full task."""
    rng = random.Random(f"limit:{task.name}:{SEED}")
    flat = [(gi, qi) for gi, g in enumerate(task.groups) for qi in range(len(g.qids))]
    chosen = rng.sample(flat, min(limit, len(flat)))
    by_group: dict[int, list[int]] = {}
    for gi, qi in chosen:
        by_group.setdefault(gi, []).append(qi)
    groups = []
    for gi in sorted(by_group):
        g = task.groups[gi]
        qis = sorted(by_group[gi])
        gold = sorted({d for qi in qis for d in g.qrels[qi]})
        rest = [d for d in range(len(g.dids)) if d not in set(gold)]
        n_dis = min(len(rest), base_distractors + len(qis))
        keep = sorted(set(gold) | set(rng.sample(rest, n_dis)))
        remap = {old: new for new, old in enumerate(keep)}
        groups.append(
            Group(
                qids=[g.qids[qi] for qi in qis],
                qtexts=[g.qtexts[qi] for qi in qis],
                dids=[g.dids[d] for d in keep],
                dtexts=[g.dtexts[d] for d in keep],
                qrels=[{remap[d]: s for d, s in g.qrels[qi].items()} for qi in qis],
                doc_session=[g.doc_session[d] for d in keep] if g.doc_session else None,
                gold_sessions=[g.gold_sessions[qi] for qi in qis] if g.gold_sessions else None,
                ignore_identical=g.ignore_identical,
            )
        )
    return Task(task.name, task.kind, task.q_instruction, task.d_instruction, groups, dict(task.meta), limited=True)
