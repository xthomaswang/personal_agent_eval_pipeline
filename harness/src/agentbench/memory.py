"""Memory sampler: physical footprint of a server PID tree (footprint(1), RSS fallback)
plus system memory used from vm_stat. Run as a background asyncio task."""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from typing import Any


async def _sh(*cmd: str) -> str:
    p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await p.communicate()
    return out.decode("utf-8", "replace")


def descendants(root: int, ps_text: str) -> list[int]:
    children: dict[int, list[int]] = {}
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(children.get(p, []))
    return out


_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_FP_RE = re.compile(r"\[(\d+)\].*?Footprint:\s*([\d.]+)\s*([KMGT]?B)")


def parse_footprint(text: str) -> dict[int, int]:
    return {int(m.group(1)): int(float(m.group(2)) * _UNITS[m.group(3)]) for m in _FP_RE.finditer(text)}


def parse_vm_stat(text: str) -> dict[str, Any]:
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 16384
    vals = {k.strip().lower(): int(v) for k, v in re.findall(r"^([^:\n]+):\s+(\d+)\.?$", text, re.M)}
    anon = vals.get("anonymous pages")
    purge = vals.get("pages purgeable", 0)
    wired = vals.get("pages wired down", 0)
    comp = vals.get("pages occupied by compressor", 0)
    app = (anon - purge) if anon is not None else vals.get("pages active", 0)
    return {"page_size": page, "used_bytes": (app + wired + comp) * page,
            "wired_bytes": wired * page, "compressed_bytes": comp * page,
            "free_bytes": vals.get("pages free", 0) * page}


async def sample_once(root_pid: int) -> dict[str, Any]:
    pids = descendants(root_pid, await _sh("ps", "-axo", "pid=,ppid="))
    fp: dict[int, int] = {}
    if shutil.which("footprint"):
        args = ["footprint", "--noCategories"]
        for p in pids:
            args += ["-p", str(p)]
        fp = parse_footprint(await _sh(*args))
    source = "footprint" if fp else "rss"
    if not fp:
        for line in (await _sh("ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids)))).splitlines():
            a = line.split()
            if len(a) == 2:
                fp[int(a[0])] = int(a[1]) * 1024
    return {"type": "memory", "t_wall": time.time(), "root_pid": root_pid, "pids": pids, "source": source,
            "per_pid_bytes": fp, "tree_bytes": sum(fp.values()), "system": parse_vm_stat(await _sh("vm_stat"))}


class MemorySampler:
    """`async with MemorySampler(pid, sink=writer.write)`: samples every `interval` s."""

    def __init__(self, pid: int, sink, interval: float = 2.0) -> None:
        self.pid, self.sink, self.interval = pid, sink, interval
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            s = await sample_once(self.pid)
            self.samples.append(s)
            self.sink(s)
            await asyncio.sleep(self.interval)

    async def __aenter__(self) -> "MemorySampler":
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        try:  # always end with one sample, so short segments are never empty
            s = await sample_once(self.pid)
            self.samples.append(s)
            self.sink(s)
        except Exception:  # noqa: BLE001 - sampling must never break a measurement
            pass

    def peak_tree_bytes(self) -> int | None:
        return max((s["tree_bytes"] for s in self.samples), default=None)
