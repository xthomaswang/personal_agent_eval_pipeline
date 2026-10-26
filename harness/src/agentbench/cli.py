"""agentbench CLI: fixtures, gate, m1-m7, run-all, summarize, m8-report, mock-server, info, mem.

See harness/README.md for usage of every subcommand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path


def _ints(s: str) -> list[int]:
    return [int(x.replace("_", "")) for x in s.split(",") if x.strip()]


def _strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def target_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("server under test")
    g.add_argument("--profile", choices=["generic", "llamacpp", "omlx", "vllm", "lithos", "mlxlm", "mlc"],
                   default="generic", help="framework profile that shapes requests (see profiles.py)")
    g.add_argument("--structured-format", choices=["mlc", "openai", "none"],
                   help="mlc profile: response_format shape (default mlc = {type: json_object, schema: <string>})")
    g.add_argument("--structured-union", choices=["anyOf", "oneOf"],
                   help="mlc profile: union keyword of the step schema (default anyOf; MLC's xgrammar ignores oneOf)")
    g.add_argument("--base-url", help="server URL; default http://127.0.0.1:<profile port>")
    g.add_argument("--model", default="", help="model id to send (mlxlm omits it)")
    g.add_argument("--framework", help="output directory under the repo root, e.g. omlx, mlx-lm, llama.cpp")
    g.add_argument("--framework-version", default="", help="recorded in the provenance header")
    g.add_argument("--launch-flags", default="", help="server launch flags, recorded in the provenance header")
    g.add_argument("--tag", default="default", help="row label, e.g. 8bit-spec-off")
    g.add_argument("--server-pid", type=int, help="server PID for memory sampling (G5, M6)")
    g.add_argument("--server-log", type=Path, help="server log to read cached tokens from (lithos)")
    g.add_argument("--context-config", type=int, help="configured context length, recorded for G5")
    g.add_argument("--api-key", default=os.environ.get("AGENTBENCH_API_KEY", ""), help="bearer key (or AGENTBENCH_API_KEY)")
    g.add_argument("--extra-body", default="{}", help="JSON object deep-merged into every OpenAI request body")
    g.add_argument("--thinking-fields", choices=["hermes", "kwargs", "effort", "none"], default="hermes",
                   help="how metric suites express thinking on/off (default: Hermes-style, both fields)")
    g.add_argument("--screenshot-placement", choices=["user", "tool"], default="user",
                   help="screenshots in a user message after the tool result (default) or inside the tool message")
    g.add_argument("--text-only", action="store_true", help="replace screenshots with equal-token text equivalents")
    g.add_argument("--quick", action="store_true", help="smoke mode: minimal trials and sizes")
    g.add_argument("--trials", type=int, help="trials per cell (suite defaults otherwise)")
    g.add_argument("--read-timeout", type=float, default=1800.0)
    g.add_argument("--total-timeout", type=float, default=3600.0)
    g.add_argument("--preflight", action="store_true",
                   help="run scripts/quiet_check.sh before each suite; a failing check routes results to raw/_unverified/")
    g.add_argument("--preflight-cmd", help="quiet-check command to run instead of scripts/quiet_check.sh")
    g.add_argument("--unverified", action="store_true",
                   help="write results to <framework>/raw/_unverified/ (gitignored, never summarized)")
    g.add_argument("--interference", help="M8: JSON object stored as the header's `interference` record "
                   "(scenario, parameters, allowances, load PID, block; scripts/run_m8.sh)")
    g.add_argument("--out-root", type=Path, help="repo root that holds <framework>/raw (default: this repo)")
    g.add_argument("--fixtures", type=Path, help="fixtures directory (default: <repo>/fixtures)")
    return p


def suite_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("suite options (each applies to the suites named in its help)")
    g.add_argument("--cold-trials", type=int, help="m1: cold-comparison trials (default 3)")
    g.add_argument("--rewarm", action="store_true", help="m1: resend the warm-up before every trial")
    g.add_argument("--m1-max-tokens", type=int, default=192, help="m1/m5: output cap of the main step")
    g.add_argument("--sizes", type=_ints, help="m2: prompt sizes, e.g. 16000,32000,64000")
    g.add_argument("--spec-label", choices=["on", "off"], help="m3: how the server was launched (required)")
    g.add_argument("--contexts", type=_ints, help="m3: thinking-off contexts (default 16000,32000,64000)")
    g.add_argument("--caps", type=_ints, help="m3: thinking-off output caps (default 50,150,300)")
    g.add_argument("--think-contexts", type=_ints, help="m3: thinking-on contexts (default 16000,32000)")
    g.add_argument("--think-caps", type=_ints, help="m3: thinking-on output caps (default 1000,2000)")
    g.add_argument("--no-thinking-bucket", action="store_true", help="m3: skip the thinking-on bucket")
    g.add_argument("--think-effort", default="xhigh", help="m3: reasoning effort for the thinking bucket")
    g.add_argument("--bust-cache", action="store_true", help="m3: fresh nonce per trial (default: cache allowed)")
    g.add_argument("--steps", type=int, help="m4/m6: trajectory steps (default 20)")
    g.add_argument("--runs", type=int, help="m4: trajectory runs (default 3)")
    g.add_argument("--m4-max-tokens", type=int, default=150, help="m4/m6: output cap per step")
    g.add_argument("--no-bust", action="store_true", help="m4: no per-run nonce (only meaningful for one run)")
    g.add_argument("--aux", type=_ints, help="m5: auxiliary request counts (default 1,2; 0 = solo is always run)")
    g.add_argument("--aux-lead", type=float, default=0.5, help="m5: seconds between helper start and main step")
    g.add_argument("--aux-max-tokens", type=int, default=256, help="m5: helper output cap")
    g.add_argument("--no-rewarm", action="store_true", help="m5: do not re-warm the prefix before each trial")
    g.add_argument("--minutes", type=float, help="m6: sustained-loop duration (required for m6)")
    g.add_argument("--mem-interval", type=float, default=5.0, help="m6: memory sampling interval in seconds")
    g.add_argument("--cases", type=_strs, help="m7: case ids or id prefixes, comma-separated")
    g.add_argument("--m7-max-tokens", type=int, default=1024)
    return p


def make_config(a: argparse.Namespace):
    from .fixtures.store import default_root
    from .profiles import get_profile
    from .provenance import repo_root
    from .suites.common import RunConfig

    prof = get_profile(a.profile)
    if prof.tool_call_mode == "structured_output":
        from dataclasses import replace
        prof = replace(prof, structured_format=a.structured_format or prof.structured_format,
                       structured_union=a.structured_union or prof.structured_union)
    elif a.structured_format or a.structured_union:
        raise SystemExit("--structured-format/--structured-union apply to structured-output profiles (mlc) only")
    base = a.base_url or (f"http://127.0.0.1:{prof.port}" if prof.port else None)
    if not base:
        raise SystemExit("--base-url is required with the generic profile")
    from .suites.common import check_port
    check_port(base)
    framework = a.framework or (prof.framework if prof.name != "generic" else None)
    if not framework:
        raise SystemExit("--framework is required with the generic profile")
    try:
        extra = json.loads(a.extra_body)
        assert isinstance(extra, dict)
    except (ValueError, AssertionError):
        raise SystemExit("--extra-body must be a JSON object")
    interference = None
    if a.interference:
        try:
            interference = json.loads(a.interference)
            assert isinstance(interference, dict)
        except (ValueError, AssertionError):
            raise SystemExit("--interference must be a JSON object")
    args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k not in ("func", "api_key")}
    preflight = None
    if a.preflight_cmd:
        import shlex
        preflight = shlex.split(a.preflight_cmd)
    elif a.preflight:
        preflight = [str(repo_root() / "scripts" / "quiet_check.sh")]
    return RunConfig(profile=prof, base_url=base.rstrip("/"), model=a.model, framework=framework,
                     framework_version=a.framework_version, launch_flags=a.launch_flags, tag=a.tag,
                     out_root=a.out_root or repo_root(), fixtures_dir=a.fixtures or default_root(),
                     text_only=a.text_only or bool(prof.force_text_only),
                     text_only_reason="" if a.text_only else (prof.force_text_only or ""),
                     quick=a.quick, server_pid=a.server_pid, server_log=a.server_log,
                     api_key=a.api_key, extra_body=extra, thinking_fields=a.thinking_fields,
                     read_timeout=a.read_timeout, total_timeout=a.total_timeout,
                     screenshot_placement=a.screenshot_placement, context_config=a.context_config, args=args,
                     preflight_cmd=preflight, unverified=a.unverified, interference=interference)


def make_opts(a: argparse.Namespace):
    from .suites.opts import SuiteOpts

    return SuiteOpts(trials=a.trials, cold_trials=a.cold_trials, rewarm=a.rewarm, m1_max_tokens=a.m1_max_tokens,
                     sizes=a.sizes, spec_label=a.spec_label, contexts=a.contexts, caps=a.caps,
                     think_contexts=a.think_contexts, think_caps=a.think_caps, no_thinking_bucket=a.no_thinking_bucket,
                     think_effort=a.think_effort, bust_cache=a.bust_cache, steps=a.steps, runs=a.runs,
                     m4_max_tokens=a.m4_max_tokens, no_bust=a.no_bust, aux=a.aux, aux_lead=a.aux_lead,
                     aux_max_tokens=a.aux_max_tokens, no_rewarm=a.no_rewarm, minutes=a.minutes,
                     mem_interval=a.mem_interval, cases=a.cases, m7_max_tokens=a.m7_max_tokens)


def cmd_suite(a: argparse.Namespace) -> int:
    from .fixtures.store import FixturesMissing
    from .suites import run_suite, validate

    opts = make_opts(a)
    errs = validate([a.cmd], opts)
    if errs:
        print("; ".join(errs), file=sys.stderr)
        return 2
    cfg = make_config(a)
    try:
        path, _ = asyncio.run(run_suite(cfg, opts, a.cmd))
    except FixturesMissing as e:
        print(str(e), file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"{a.cmd}: {e}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def cmd_run_all(a: argparse.Namespace) -> int:
    from .fixtures.store import FixturesMissing
    from .suites import run_all

    cfg, opts = make_config(a), make_opts(a)
    try:
        res = asyncio.run(run_all(cfg, opts, a.suites, auto_text_only=not a.no_auto_text_only,
                                  write_summary=not a.no_summary))
    except FixturesMissing as e:
        print(str(e), file=sys.stderr)
        return 2
    failed = [k for k, v in res["results"].items() if not v["ok"]]
    print(f"run-all: {len(res['results']) - len(failed)} suites ok, failed: {failed or 'none'}; "
          f"summary in {res['summary_md']}", file=sys.stderr)
    return 1 if failed else 0


def cmd_summarize(a: argparse.Namespace) -> int:
    from .profiles import get_profile
    from .provenance import repo_root
    from .summarize import summarize

    fw = a.framework or (get_profile(a.profile).framework if a.profile else None)
    if not fw:
        print("summarize needs --framework (or --profile)", file=sys.stderr)
        return 2
    root = a.out_root or repo_root()
    print(summarize(root / fw, a.tag, files=a.files, all_runs=a.all_runs, write=a.write))
    return 0


def cmd_fixtures(a: argparse.Namespace) -> int:
    from .fixtures.build import DEFAULT_SEED, build, check, format_sizes
    from .fixtures.store import default_root

    root = a.out or default_root()
    if a.action == "check":
        problems = check(root)
        for p in problems:
            print("MISMATCH", p)
        print("fixtures check: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
        return 0 if not problems else 1
    t0 = time.time()
    m = build(root, a.seed if a.seed is not None else DEFAULT_SEED)
    print(format_sizes(m))
    print(f"wrote {len(m['files'])} files under {root} in {time.time() - t0:.1f} s; G2 code {m['g2_code']}")
    return 0


def cmd_mock(a: argparse.Namespace) -> int:
    from .mock_server import MockServer

    if a.port != 0 and not 8790 <= a.port <= 8799:
        print("mock servers bind port 0 (ephemeral) or 8790-8799 only", file=sys.stderr)
        return 2
    responder = None
    if a.behavior:
        from .fixtures.store import default_root
        from .mock_smart import SmartResponder

        responder = SmartResponder.from_fixtures(a.fixtures or default_root(), a.behavior,
                                                 log_path=str(a.log) if a.log else None)
    s = MockServer(port=a.port, responder=responder).start()
    print(f"mock server ({a.behavior or 'legacy scenarios'}) on {s.url}", flush=True)
    while True:
        time.sleep(3600)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentbench", description="Agent-shaped benchmark client for local LLM servers.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp, sp = target_parser(), suite_parser()

    sub.add_parser("info", help="print tokenizer snapshot and image-token arithmetic")
    fx = sub.add_parser("fixtures", help="build or check the deterministic fixtures")
    fx.add_argument("action", choices=["build", "check"])
    fx.add_argument("--out", type=Path, help="fixtures directory (default: <repo>/fixtures)")
    fx.add_argument("--seed", type=int)
    descr = {"gate": "capability gate G1-G5 (pass/fail table + capabilities.md fragment)",
             "m1": "cached-prefix TTFT", "m2": "cold prefill throughput", "m3": "short structured decode (--spec-label)",
             "m4": "20-step trajectory, per-step TTFT / e2e / cached tokens", "m5": "concurrency with 1-2 helpers",
             "m6": "memory and stability (--minutes, --server-pid)", "m7": "tool-call correctness"}
    for name, h in descr.items():
        sub.add_parser(name, parents=[tp, sp], help=h)
    ra = sub.add_parser("run-all", parents=[tp, sp], help="gate plus a subset of M1-M7 in order, then summary.md")
    ra.add_argument("--suites", type=_strs, default=["gate", "m1", "m2", "m3", "m4", "m5", "m7"])
    ra.add_argument("--no-auto-text-only", action="store_true",
                    help="keep sending images after the gate's G2 failed")
    ra.add_argument("--no-summary", action="store_true",
                    help="print the tables but leave summary.md alone (M8 blocks; scripts/run_m8.sh)")
    sm = sub.add_parser("summarize", help="markdown tables from <framework>/raw/*.jsonl")
    sm.add_argument("--framework")
    sm.add_argument("--profile")
    sm.add_argument("--tag")
    sm.add_argument("--files", type=Path, nargs="+")
    sm.add_argument("--all-runs", action="store_true", help="pool all files instead of the latest per suite and tag")
    sm.add_argument("--write", action="store_true", help="upsert the tables into <framework>/summary.md")
    sm.add_argument("--out-root", type=Path)
    m8 = sub.add_parser("m8-report", help="M8 interference: slowdown per framework and scenario (markdown)")
    m8.add_argument("--framework", action="append", help="framework directory (repeatable; default: every one with M8 files)")
    m8.add_argument("--tag", help="the run_m8.sh tag, without -m8-<block>")
    m8.add_argument("--details", action="store_true", help="add one row per scenario block and suite")
    m8.add_argument("--unverified", action="store_true", help="read raw/_unverified/ instead (mechanics checks only)")
    m8.add_argument("--out-root", type=Path)
    ms = sub.add_parser("mock-server", help="run the mock OpenAI/Anthropic SSE server")
    ms.add_argument("--port", type=int, default=8790)
    ms.add_argument("--behavior", choices=["generic", "llamacpp", "omlx", "vllm", "lithos", "mlxlm", "mlc"],
                    help="smart responder imitating a framework (default: legacy scenario mode)")
    ms.add_argument("--fixtures", type=Path)
    ms.add_argument("--log", type=Path, help="lithos behavior: write cached-token log lines here")
    mem = sub.add_parser("mem", help="sample memory of a PID tree every --interval s")
    mem.add_argument("--pid", type=int, required=True)
    mem.add_argument("--interval", type=float, default=2.0)
    mem.add_argument("--duration", type=float, default=10.0)
    a = ap.parse_args(argv)

    if a.cmd == "info":
        from .tokenizer import QwenPrompt, image_tokens
        p = QwenPrompt()
        print(json.dumps({"snapshot": p.snap.name, "preprocessor": p.pre,
                          "image_tokens_1280x1024": image_tokens(1280, 1024, p.pre)}, indent=2))
        return 0
    if a.cmd == "fixtures":
        return cmd_fixtures(a)
    if a.cmd in descr:
        return cmd_suite(a)
    if a.cmd == "run-all":
        return cmd_run_all(a)
    if a.cmd == "summarize":
        return cmd_summarize(a)
    if a.cmd == "m8-report":
        from .m8report import report
        from .provenance import repo_root

        print(report(a.out_root or repo_root(), frameworks=a.framework, tag=a.tag, details=a.details,
                     unverified=a.unverified))
        return 0
    if a.cmd == "mock-server":
        return cmd_mock(a)
    if a.cmd == "mem":
        from .memory import sample_once

        async def run() -> None:
            end = time.time() + a.duration
            while time.time() < end:
                print(json.dumps(await sample_once(a.pid)))
                await asyncio.sleep(a.interval)

        asyncio.run(run())
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
