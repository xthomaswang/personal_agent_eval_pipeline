"""Suite options. `None` means "use the suite default" (smaller defaults in --quick)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SuiteOpts:
    trials: int | None = None
    # m1 / m5 main step
    cold_trials: int | None = None
    rewarm: bool = False
    m1_max_tokens: int = 192
    # m2
    sizes: list[int] | None = None
    # m3
    spec_label: str | None = None
    contexts: list[int] | None = None
    caps: list[int] | None = None
    think_contexts: list[int] | None = None
    think_caps: list[int] | None = None
    no_thinking_bucket: bool = False
    think_effort: str = "xhigh"
    bust_cache: bool = False
    # m4 / m6 trajectory
    steps: int | None = None
    runs: int | None = None
    m4_max_tokens: int = 150
    no_bust: bool = False
    # m5
    aux: list[int] | None = None
    aux_lead: float = 0.5
    aux_max_tokens: int = 256
    no_rewarm: bool = False
    # m6
    minutes: float | None = None
    mem_interval: float = 5.0
    # m7
    cases: list[str] | None = None
    m7_max_tokens: int = 1024


def pick(value, quick: bool, quick_default, default):
    if value is not None:
        return value
    return quick_default if quick else default
