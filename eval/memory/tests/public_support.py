"""Helpers the public-benchmark tests share: a tiny synthetic LongMemEval file, a launcher that keeps its specs, a study.

The data here is made up for the tests; it is not a record of LongMemEval.
"""
import json
from pathlib import Path

from memory.public import data, judge, run
from memory.tests.across_fake import FakeModelServer
from memory.tests.across_support import Launcher, git, make_repo

GOLD = "Business Administration"


def turns(*pairs):
    """[(user, assistant), ...] as LongMemEval turns; the user turns carry a has_answer mark, as some of the real ones do."""
    out = []
    for user, assistant in pairs:
        out += [{"role": "user", "content": user, "has_answer": False}, {"role": "assistant", "content": assistant}]
    return out


def mini_questions():
    """Two synthetic questions: a single-session-user one answered by session `sess_a`, and an abstention one."""
    return [
        {
            "question_id": "q1aa", "question_type": "single-session-user", "question": "Which degree do I hold?", "answer": GOLD,
            "question_date": "2023/05/30 (Tue) 23:40",
            "haystack_session_ids": ["sess_a", "sess_b", "sess_c"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 10:00", "2023/05/22 (Mon) 09:30"],
            "haystack_sessions": [
                turns(("I hold a degree in Business Administration.", "Congratulations on the degree!")),
                turns(("My favourite hobby is bouldering at the gym.", "That sounds like fun.")),
                [*turns(("Give me a pancake recipe.", "Mix flour, eggs and milk.")), {"role": "user", "content": "  "}],
            ],
            "answer_session_ids": ["sess_a"],
        },
        {
            "question_id": "q2bb_abs", "question_type": "temporal-reasoning", "question": "Which instrument do I play?",
            "answer": "You never said which instrument you play.", "question_date": "2023/06/02 (Fri) 08:05",
            "haystack_session_ids": ["sess_x", "sess_y"],
            "haystack_dates": ["2023/06/01 (Thu) 18:00", "2023/05/31 (Wed) 07:15"],
            "haystack_sessions": [
                turns(("I play tennis on weekends.", "Nice, how long have you played?")),
                turns(("I need a packing list for a trip.", "Passport, charger, socks.")),
            ],
            "answer_session_ids": ["sess_x"],
        },
    ]


def synthetic_questions(per_type=5, abstaining=("knowledge-update", "multi-session", "temporal-reasoning"), abs_each=3):
    """Many tiny questions for the selection tests: ``per_type`` plain ones of every type, ``abs_each`` `_abs` ones for some types."""
    out = []

    def make(qid, type_):
        return {
            "question_id": qid, "question_type": type_, "question": f"question {qid}?", "answer": f"answer {qid}",
            "question_date": "2023/05/30 (Tue) 23:40", "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [turns((f"hello {qid}", "hi"))], "answer_session_ids": ["s1"],
        }

    for type_ in data.TYPES:
        out += [make(f"{type_}-{n:02d}", type_) for n in range(per_type)]
    for type_ in abstaining:
        out += [make(f"{type_}-{n:02d}_abs", type_) for n in range(abs_each)]
    return out


def write_dataset(path, questions=None):
    Path(path).write_text(json.dumps(questions or mini_questions()), encoding="utf-8")
    return path


def fake_judge(body):
    """The judge model of the tests: yes for a reply that holds the gold answer (or, for an unanswerable question, says it does not know)."""
    prompt = body["messages"][0]["content"]
    response = prompt.split("Model Response:", 1)[1]
    if "unanswerable" in prompt:
        return "yes" if "know" in response else "no"
    return "yes" if GOLD in response else "no"


def judge_server(replies=fake_judge, ids=("chat-model", "embed-model")):
    return FakeModelServer(list(ids), replies=replies)


class PublicLauncher(Launcher):
    """The across launcher around the in-process FakeGateway, keeping every LaunchSpec (its env and its -X pairs)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.specs = []

    def __call__(self, spec):
        self.specs.append(spec)
        return super().__call__(spec)


def make_public_study(tmp_path, repo, models, launcher=None, questions=None, arms=None, arm_names=None, **kw):
    questions = questions or mini_questions()
    arms = arms or {"A": run.Arm("A", "embedding retriever")}
    if "expect_sha" not in kw:
        kw["expect_sha"] = git(repo, "rev-parse", "HEAD")[:8]
    options = dict(
        run_dir=tmp_path / "run", questions=questions, ids=tuple(sorted(q["question_id"] for q in questions)), arms=arms,
        arm_names=tuple(arm_names or arms), repeats=1, localmelo=repo, base_url=models.base_url, model="chat-model",
        embedding_model="embed-model", judge_model="chat-model", data_path=Path("mini.json"), data_sha256="0" * 64,
        poll_s=0.01, launcher=launcher or PublicLauncher(), snapshot=lambda study: {"sw_vers": "ProductName: test", "xcodebuild_version": "Xcode test"},
        cold_start=False, do_warmup=False, sleep=lambda s: None, scratch=tmp_path, work_dir=tmp_path / "work",
    )
    options.update(kw)
    return run.Study(**options)


__all__ = ["GOLD", "PublicLauncher", "fake_judge", "judge", "judge_server", "make_public_study", "make_repo", "mini_questions", "synthetic_questions", "write_dataset"]


def result_row(qid, qtype, arm, repeat, correct, *, abstention=False, auto=None, cover=None, searches=0, search_hit=None, via=None,
               answer_sessions=("s1",), judge_flags=(), prompt_tokens=None, ttft_ms=None, state="finished"):
    """A results.jsonl row with the fields the report reads; ``judge_flags`` is any of "retry", "unparsed", "error"."""
    flags = set(judge_flags)
    judged = state == "finished"
    return {
        "question_id": qid, "question_type": qtype, "abstention": abstention, "arm": arm, "repeat": repeat, "state": state,
        "correct": correct, "prompt_tokens": prompt_tokens, "ttft_ms": ttft_ms,
        "judge_retry": int("retry" in flags or "unparsed" in flags), "judge_unparsed": int("unparsed" in flags),
        "judge": {"label": correct, "retry": "retry" in flags or "unparsed" in flags, "unparsed": "unparsed" in flags,
                  "error": "OSError: down" if "error" in flags else ""} if judged else None,
        "retrieval": {"answer_sessions": list(answer_sessions), "auto_hit": auto, "auto_cover": cover, "searches": searches,
                      "search_hit": search_hit, "search_via": via, "search_hint": None, "used": []},
    }


def write_run(run_dir, rows, *, arms, questions, repeats=2, controls=(), ingests=None, extra_bodies=None):
    """The files of a finished run, written by hand: public.json, questions.json, results.jsonl, controls.jsonl, ingest.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "public-lme-v1", "data": {"name": "mini.json", "sha256": "ab" * 32}, "stage": {"per_type": 17, "abstention": 18, "seed": 0},
        "questions": [q["question_id"] for q in questions], "arms": {a: {"x": [], "config": {}, "extra_body": (extra_bodies or {}).get(a, {}),
        "embedding": True} for a in arms}, "repeats": repeats, "localmelo_sha": "0123456789abcdef", "model": "chat-model",
        "embedding_model": "embed-model", "judge_model": "chat-model",
    }
    (run_dir / "public.json").write_text(json.dumps({"config": config, "complete": True}))
    (run_dir / "questions.json").write_text(json.dumps(questions))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "controls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in controls))
    (run_dir / "environment.json").write_text(json.dumps({"start": {"sw_vers": "ProductName: macOS\nBuildVersion: TEST", "bench_sha": "f" * 40}}))
    for qid, report in (ingests or {}).items():
        (run_dir / "questions" / qid).mkdir(parents=True, exist_ok=True)
        (run_dir / "questions" / qid / "ingest.json").write_text(json.dumps(report))
