"""The answer check: LongMemEval's prompts as published, the verdict parse and the second ask."""
import hashlib

import pytest
from memory.public import data, judge
from memory.tests.public_support import judge_server


def digest(task, abstention=False):
    return hashlib.sha256(judge.get_anscheck_prompt(task, "Q?", "A!", "R.", abstention).encode()).hexdigest()[:16]


def test_prompts_are_the_benchmarks_own():
    """The first 16 hex digits of the SHA-256 of each prompt for Q?/A!/R., taken from upstream's get_anscheck_prompt."""
    assert digest("single-session-user") == digest("single-session-assistant") == digest("multi-session") == "b304e51aae67582c"
    assert digest("temporal-reasoning") == "530874d9ff032528"
    assert digest("knowledge-update") == "87a5ae5e7a02d595"
    assert digest("single-session-preference") == "7b139d99ffbd8f62"
    assert digest("multi-session", abstention=True) == "9cbd0c9477210ba9"
    assert digest("anything", abstention=True) == "9cbd0c9477210ba9"  # an abstention question ignores its type
    with pytest.raises(NotImplementedError):
        judge.get_anscheck_prompt("other", "Q?", "A!", "R.")
    assert set(data.TYPES) >= {"temporal-reasoning", "knowledge-update", "single-session-preference"}
    prompt = judge.get_anscheck_prompt("temporal-reasoning", "How long?", "18 days", "19 days")
    assert "do not penalize off-by-one errors" in prompt and prompt.endswith("Is the model response correct? Answer yes or no only.")
    assert "Question: How long?\n\nCorrect Answer: 18 days\n\nModel Response: 19 days\n\n" in prompt


def test_the_source_and_license_are_kept():
    text = judge.__doc__
    assert "https://github.com/xiaowu0162/LongMemEval (src/evaluation/evaluate_qa.py)" in text
    assert "Copyright (c) 2024 Di Wu" in text and "MIT License" in text


@pytest.mark.parametrize("reply,label", [
    ("yes", True), ("Yes", True), ("YES.", True), (" yes, the response is right", True), ("\nYes\n", True),
    ("no", False), ("No.", False), ("NO, it is not", False), ("**No**", False),
    ("", None), ("maybe", None), ("not sure", None), ("The answer is yes", None), ("1", None),
])
def test_parse_verdict_reads_the_first_word(reply, label):
    assert judge.parse_verdict(reply) is label


def test_a_clear_reply_is_asked_once_with_production_fields():
    server = judge_server(["Yes"])
    try:
        v = judge.judge(server.base_url, "judge-model", "single-session-user", "Which degree?", "BA", "A BA")
        assert (v.label, v.correct, v.retry, v.unparsed, v.error, v.reply, v.model) == (True, True, False, False, "", "Yes", "judge-model")
        assert len(server.bodies) == 1
        body = server.bodies[0]
        assert body["model"] == "judge-model" and body["temperature"] == 0 and body["max_tokens"] == 10
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"] == [{"role": "user", "content": judge.get_anscheck_prompt("single-session-user", "Which degree?", "BA", "A BA")}]
    finally:
        server.stop()


def test_a_reply_that_is_neither_is_asked_again_without_sampling_fields():
    server = judge_server(["hmm", "No"])
    try:
        v = judge.judge(server.base_url, "m", "multi-session", "Q?", "A", "R")
        assert (v.label, v.retry, v.unparsed, v.reply, v.retry_reply) == (False, True, False, "hmm", "No")
        first, second = server.bodies
        assert first["temperature"] == 0 and "temperature" not in second  # the server's defaults on the second ask
        assert second["max_tokens"] == 10 and second["chat_template_kwargs"] == {"enable_thinking": False}
        assert second["messages"] == first["messages"]
    finally:
        server.stop()


def test_two_unclear_replies_grade_the_answer_wrong_and_are_counted():
    server = judge_server(["", "who knows"])
    try:
        v = judge.judge(server.base_url, "m", "knowledge-update", "Q?", "A", "R")
        assert (v.label, v.correct, v.retry, v.unparsed, len(server.bodies)) == (None, False, True, True, 2)
        assert v.to_json()["unparsed"] is True and v.to_json()["retry"] is True
    finally:
        server.stop()


def test_an_abstention_question_uses_its_own_prompt():
    server = judge_server(["yes"])
    try:
        judge.judge(server.base_url, "m", "temporal-reasoning", "Q?", "explanation", "I do not know", abstention=True)
        assert "unanswerable question" in server.bodies[0]["messages"][0]["content"]
    finally:
        server.stop()


def test_an_unreachable_judge_leaves_the_verdict_open():
    server = judge_server()
    url = server.base_url
    server.stop()
    v = judge.judge(url, "m", "multi-session", "Q?", "A", "R")
    assert v.label is None and v.correct is None and v.error and not v.unparsed
    broken = judge.judge("http://127.0.0.1:1/v1", "m", "multi-session", "Q?", 42, "R", post=lambda *a, **k: (1.0, {"nope": 1}))
    assert broken.error.startswith("ValueError") and broken.correct is None  # not a chat completion
