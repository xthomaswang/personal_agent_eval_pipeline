"""Extract the answer from an agent's reply (DESIGN.md, section 7, step 1).

The prompt asks the agent to end its reply with one line, ``FINAL ANSWER: <answer>``. This module finds that line
after Markdown and a few inline HTML tags are stripped, so ``**FINAL ANSWER:** Delhi``, ``> FINAL ANSWER: Delhi`` and
``<b>Final answer:</b> Delhi`` all work. The marker is case-insensitive, and the last line that has it wins, so a
reply that quotes the format earlier in its text is read from its closing line.

Outcome of ``extract``:

- ``no_answer``: no line has the marker.
- ``abstain``: the answer is empty or ``unknown``, the agent's way to say it could not find the answer.
- ``answer``: anything else; scoring decides whether it is right (score.py).

The marker follows GAIA's answer format (https://huggingface.co/spaces/gaia-benchmark/leaderboard).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ANSWER = "answer"
ABSTAIN = "abstain"
NO_ANSWER = "no_answer"

_MARKER = re.compile(r"final\s+answer\s*:", re.IGNORECASE)
_FENCE_LINE = re.compile(r"^[ \t]*(?:```|~~~).*$", re.MULTILINE)
_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")
_TAG = re.compile(r"</?(?:a|b|br|code|div|em|i|mark|p|span|strong|u)\b[^>]*>", re.IGNORECASE)
_EMPHASIS = re.compile(r"[*`~]|__")
_UNDERSCORE_PAIR = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("«", "»"))


@dataclass(frozen=True)
class Extraction:
    """``status`` is ANSWER, ABSTAIN or NO_ANSWER; ``answer`` is the text after the marker ("" for NO_ANSWER)."""

    status: str
    answer: str


def strip_markup(text: str) -> str:
    """Remove Markdown emphasis, code, links, fences and a few inline HTML tags; the words stay."""
    text = _FENCE_LINE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    text = _EMPHASIS.sub("", text)
    return _UNDERSCORE_PAIR.sub(r"\1", text)


def _unquote(answer: str) -> str:
    """Drop one layer of quotes that wrap the whole answer (``"Delhi"``), but not ``"A" and "B"``."""
    for opening, closing in _QUOTE_PAIRS:
        if (
            len(answer) >= 2
            and answer[0] == opening
            and answer[-1] == closing
            and opening not in answer[1:-1]
        ):
            return answer[1:-1].strip()
    return answer


def extract(text: str | None) -> Extraction:
    """The answer on the last ``FINAL ANSWER:`` line of ``text``, or why there is none."""
    if not text:
        return Extraction(NO_ANSWER, "")
    for line in reversed(strip_markup(text).splitlines()):
        found = list(_MARKER.finditer(line))
        if found:
            answer = _unquote(line[found[-1].end() :].strip())
            if answer.rstrip(" .!").lower() in ("", "unknown"):
                return Extraction(ABSTAIN, answer)
            return Extraction(ANSWER, answer)
    return Extraction(NO_ANSWER, "")
