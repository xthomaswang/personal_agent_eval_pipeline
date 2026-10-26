"""Incremental parser for text/event-stream bodies.

The proxy forwards raw bytes to the client as they arrive and feeds the same
bytes to this parser on the side, so chunk boundaries can fall anywhere,
including inside a line or a multi-byte character.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    event: str | None  # value of the "event:" field, if any
    data: str  # "data:" lines joined with "\n"
    comment: bool = False  # a ":" comment line, used by servers as keepalive


class SSEParser:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._event: str | None = None
        self._data: list[str] = []
        self._pending = False

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buf += chunk
        events: list[SSEEvent] = []
        start = 0
        while True:
            end = self._buf.find(b"\n", start)
            if end < 0:
                break
            line = bytes(self._buf[start:end])
            start = end + 1
            self._line(line[:-1] if line.endswith(b"\r") else line, events)
        del self._buf[:start]
        return events

    def close(self) -> list[SSEEvent]:
        """Flush a final event that was not terminated by a blank line."""
        events: list[SSEEvent] = []
        if self._buf:
            line = bytes(self._buf)
            self._buf.clear()
            self._line(line.rstrip(b"\r"), events)
        self._dispatch(events)
        return events

    def _line(self, line: bytes, events: list[SSEEvent]) -> None:
        if not line:
            self._dispatch(events)
            return
        if line.startswith(b":"):
            events.append(SSEEvent(None, line[1:].decode("utf-8", "replace").strip(), comment=True))
            return
        name, _, value = line.partition(b":")
        if value.startswith(b" "):
            value = value[1:]
        if name == b"event":
            self._event = value.decode("utf-8", "replace")
            self._pending = True
        elif name == b"data":
            self._data.append(value.decode("utf-8", "replace"))
            self._pending = True
        # "id", "retry" and unknown fields do not matter for observation.

    def _dispatch(self, events: list[SSEEvent]) -> None:
        if self._pending:
            events.append(SSEEvent(self._event, "\n".join(self._data)))
        self._event = None
        self._data = []
        self._pending = False
