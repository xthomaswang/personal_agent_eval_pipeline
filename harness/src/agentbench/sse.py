"""Minimal incremental Server-Sent Events parser (WHATWG event-stream rules)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    event: str = "message"
    data: str = ""
    id: str | None = None
    comments: list[str] = field(default_factory=list)


class SSEParser:
    """Feed decoded lines (without trailing newline); yields events at blank lines.

    Comment lines (starting with ':') are keep-alives; they are counted but never
    produce an event on their own.
    """

    def __init__(self) -> None:
        self._event = "message"
        self._data: list[str] = []
        self._id: str | None = None
        self.n_comments = 0

    def feed_line(self, line: str) -> SSEEvent | None:
        line = line.rstrip("\r")
        if line == "":
            if not self._data:
                self._event = "message"
                return None
            ev = SSEEvent(event=self._event, data="\n".join(self._data), id=self._id)
            self._event, self._data = "message", []
            return ev
        if line.startswith(":"):
            self.n_comments += 1
            return None
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value or "message"
        elif name == "id":
            self._id = value
        return None

    def flush(self) -> SSEEvent | None:
        """Dispatch a trailing event when the stream ends without a blank line."""
        return self.feed_line("")
