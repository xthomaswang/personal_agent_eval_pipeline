"""Suites: gate (G1-G5) and metrics M1-M7, plus `run-all`."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from ..fixtures.store import Fixtures
from . import gate, m1, m2, m3, m4, m5, m6, m7
from .common import RunConfig, Target, Writer, eprint, run_preflight
from .opts import SuiteOpts

SUITES = {"gate": gate.run, "m1": m1.run, "m2": m2.run, "m3": m3.run, "m4": m4.run, "m5": m5.run, "m6": m6.run,
          "m7": m7.run}
ORDER = list(SUITES)


def validate(suites: list[str], opts: SuiteOpts) -> list[str]:
    errs = [f"unknown suite {s!r}" for s in suites if s not in SUITES]
    if "m3" in suites and opts.spec_label not in ("on", "off"):
        errs.append("m3 needs --spec-label on|off")
    if "m6" in suites and opts.minutes is None:
        errs.append("m6 needs --minutes N")
    if "m3" in suites and opts.reuse_nonce is not None:
        if opts.bust_cache:
            errs.append("m3: --reuse-nonce and --bust-cache exclude each other")
        if opts.contexts is None or len(opts.contexts) != 1:
            errs.append("m3: --reuse-nonce needs exactly one thinking-off context (--contexts N)")
        if not (opts.reuse_nonce.isdigit() and len(opts.reuse_nonce) == 12):
            errs.append("m3: --reuse-nonce must be 12 digits (a nonce never changes a prompt's token count)")
    if opts.warmup_max_tokens is not None and opts.warmup_max_tokens < 1:
        errs.append("--warmup-max-tokens must be at least 1")
    return errs


async def run_suite(cfg: RunConfig, opts: SuiteOpts, name: str, fx: Fixtures | None = None) -> tuple[Path, dict[str, Any]]:
    """Run one suite in-process; one raw file `<name>-<tag>-<timestamp>.jsonl` under cfg.fw_dir/raw
    (raw/_unverified with cfg.unverified). The returned summary carries the unit's window `t_start`
    and `t_end` (epoch seconds, stamped by the Writer), so one call can be one judged unit: e.g.
    one M3 cell (`contexts=[c], caps=[150], no_thinking_bucket=True`, or `contexts=[],
    think_contexts=[c], think_caps=[1000]`) or one M4 run (`runs=1`)."""
    errs = validate([name], opts)
    if errs:
        raise ValueError("; ".join(errs))
    fx = fx or Fixtures(cfg.fixtures_dir)
    target = Target(cfg, fx)  # checks the port before any file is written
    run_preflight(cfg)  # quiet-machine check per timed segment (opt-in)
    writer = Writer(cfg, name)
    try:
        summary = await SUITES[name](cfg, target, writer, opts)
        return writer.path, summary
    except BaseException as e:
        writer.write({"type": "error", "suite": name, "error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        writer.close()
        await target.aclose()


async def run_all(cfg: RunConfig, opts: SuiteOpts, suites: list[str], auto_text_only: bool = True,
                  write_summary: bool = True) -> dict[str, Any]:
    from ..summarize import summarize

    if "m3" in suites and opts.spec_label is None:
        opts.spec_label = "off"
        cfg.args["spec_label"] = "off"
        cfg.args["spec_label_defaulted_by_run_all"] = True
        eprint("run-all: no --spec-label given; m3 rows are labelled spec=off (pass --spec-label on for a spec-on server)")
    errs = validate(suites, opts)
    if errs:
        raise SystemExit("run-all: " + "; ".join(errs))
    fx = Fixtures(cfg.fixtures_dir)
    files: list[Path] = []
    results: dict[str, Any] = {}
    for name in [s for s in ORDER if s in suites]:
        eprint(f"== {name} ==")
        try:
            path, summary = await run_suite(cfg, opts, name, fx)
            files.append(path)
            results[name] = {"file": str(path.name), "ok": True}
        except (Exception, SystemExit) as e:  # one failing suite must not stop the others
            eprint(f"{name} failed: {type(e).__name__}: {e}")
            if not isinstance(e, SystemExit):
                traceback.print_exc()
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            raw = sorted((cfg.fw_dir / "raw" / ("_unverified" if cfg.unverified else "")).glob(f"{name}-{cfg.tag}-*.jsonl"))
            if raw:
                files.append(raw[-1])
            continue
        if name == "gate" and auto_text_only and not cfg.text_only:
            g2 = summary["items"].get("G2", {})
            if g2.get("result") == "FAIL" and g2.get("rejected"):  # accepted-but-misread images are still measured
                cfg.text_only = True
                cfg.text_only_reason = f"auto: G2 failed ({g2.get('detail', '')[:120]})"
                eprint(f"G2 failed; the remaining suites run in --text-only mode ({cfg.text_only_reason})")
    verified = [f for f in files if "_unverified" not in f.parts]
    if len(verified) < len(files):
        eprint(f"{len(files) - len(verified)} result file(s) are in raw/_unverified/ and are left out of summary.md")
    md = (summarize(cfg.fw_dir, cfg.tag, files=verified, write=write_summary and bool(verified))
          if verified else "_no verified results_\n")
    print(md)
    return {"results": results, "files": [str(f) for f in files], "summary_md": str(cfg.fw_dir / "summary.md")}
