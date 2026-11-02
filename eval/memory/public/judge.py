"""The answer check of LongMemEval, run by the model server the question was asked on.

LongMemEval grades a free-text reply with a model: one yes/no question per reply, with a prompt per question type. The
prompts below are the benchmark's own, copied without a change from its `get_anscheck_prompt` (the metric model and the
call around it are replaced by `judge`):

    Source:    https://github.com/xiaowu0162/LongMemEval (src/evaluation/evaluate_qa.py)
    Copyright (c) 2024 Di Wu
    MIT License (the license text is the LICENSE file of that repository)

The upstream script calls its hosted judge model at temperature 0 with `max_tokens` 10 and takes any reply that contains "yes" as correct.
`judge` sends the same request to an OpenAI-compatible server (`/v1/chat/completions`) with the thinking of a reasoning model
switched off (`chat_template_kwargs.enable_thinking=false`), and reads the first word of the reply: yes or no. A reply that
is neither is asked again once, with no sampling fields (the server's own defaults, in case temperature 0 made a model
loop), and the result says that it was: `retry` for a question asked twice, `unparsed` for one whose second reply is
no better; an unparsed reply grades the answer wrong. A judge that cannot be reached leaves `label` and `correct`
None with the error, and the driver asks again on the next run.
"""

from __future__ import annotations

import http.client
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from memory.across import server

MAX_TOKENS = 10
TIMEOUT_S = 300.0
_FIRST_WORD = re.compile(r"[^A-Za-z]*([A-Za-z]+)")


# --- the benchmark's prompts, verbatim (the trailing space of one line is dropped) ---

# fmt: off
# (the long lines are the benchmark's, not wrapped)
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt
# fmt: on


# --- the call ---


def parse_verdict(reply: str) -> bool | None:
    """True for a reply whose first word is "yes", False for "no" (any letter case, punctuation around it ignored), else None."""
    found = _FIRST_WORD.match(reply or "")
    word = found.group(1).lower() if found else ""
    return True if word == "yes" else False if word == "no" else None


@dataclass
class Verdict:
    label: bool | None  # True: the judge said yes; False: no; None: neither (unparsed) or no answer (error)
    reply: str = ""  # the first reply, stripped
    retry: bool = False  # the first reply was neither yes nor no, so the question was asked again
    retry_reply: str = ""  # the second reply
    unparsed: bool = False  # the second reply was neither either
    error: str = ""  # the judge could not be reached or answered with something that is not a chat completion
    ms: float = 0.0
    model: str = ""

    @property
    def correct(self) -> bool | None:
        """Whether the answer counts as right: the judge's yes; an unparsed reply is wrong; None when there was no answer."""
        return None if self.error else self.label is True

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def request_body(model: str, prompt: str, *, sampling: bool = True) -> dict[str, Any]:
    """The chat completion that asks one yes/no question; without ``sampling`` the temperature is left to the server."""
    body: dict[str, Any] = {
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if sampling:
        body["temperature"] = 0
    return body


def ask(
    base_url: str, body: dict[str, Any], post: Callable[..., tuple[float, Any]] = server.http_json, timeout: float = TIMEOUT_S,
) -> tuple[str, float]:
    """(the reply's text, milliseconds) of one chat completion; OSError, HTTPException or ValueError when it fails."""
    ms, parsed = post(base_url.rstrip("/") + "/chat/completions", body, timeout)
    try:
        content = parsed["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(f"not a chat completion: {str(parsed)[:200]}") from exc
    return (content or "").strip(), ms


def judge_prompt(
    base_url: str, model: str, prompt: str, *, post: Callable[..., tuple[float, Any]] = server.http_json, timeout: float = TIMEOUT_S,
) -> Verdict:
    """Ask the model server one yes/no ``prompt`` the way ``judge`` asks LongMemEval's: a reply that is neither is asked again once."""
    verdict = Verdict(label=None, model=model)
    try:
        verdict.reply, verdict.ms = ask(base_url, request_body(model, prompt), post, timeout)
        verdict.label = parse_verdict(verdict.reply)
        if verdict.label is None:
            verdict.retry = True
            verdict.retry_reply, ms = ask(base_url, request_body(model, prompt, sampling=False), post, timeout)
            verdict.ms += ms
            verdict.label = parse_verdict(verdict.retry_reply)
            verdict.unparsed = verdict.label is None
    except (OSError, http.client.HTTPException, ValueError) as exc:
        verdict.error = f"{type(exc).__name__}: {exc}"
        verdict.label = None
        verdict.unparsed = False
    return verdict


def judge(
    base_url: str, model: str, question_type: str, question: str, answer: Any, response: str, *, abstention: bool = False,
    post: Callable[..., tuple[float, Any]] = server.http_json, timeout: float = TIMEOUT_S,
) -> Verdict:
    """Ask the model server whether ``response`` answers ``question`` the way LongMemEval's prompt for the type wants."""
    return judge_prompt(base_url, model, get_anscheck_prompt(question_type, question, answer, response, abstention), post=post, timeout=timeout)
