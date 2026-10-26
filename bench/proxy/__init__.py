"""Model-call recording proxy.

Harnesses (Hermes, Claude Code, localmelo) talk to this proxy instead of the
model server. The proxy forwards every request to the upstream (oMLX), relays
the response chunk by chunk, can rewrite the thinking settings so that every
harness runs at the same thinking level, and appends one JSONL record per call
to ``<run_dir>/calls.jsonl``.

Usage:
    uv run python -m bench.proxy --port 8600 --harness hermes --thinking off

See docs/proxy.md for the record fields and how to point each harness here.
"""

from .app import create_app
from .config import ProxyConfig
from .sampling import SamplingPolicy
from .thinking import ThinkingMode

__all__ = ["ProxyConfig", "SamplingPolicy", "ThinkingMode", "create_app"]
