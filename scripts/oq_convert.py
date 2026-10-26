#!/usr/bin/env python3
"""Convert a BF16 HF snapshot to an oMLX oQ checkpoint, apply registry tokenizer fixes, register it.

Generalizes ../qwen3.8-27b_20261017-2258_benchmark/scripts/oq_convert.py (Qwen3.8-27B only):
  - the source snapshot, output name and oQ level come from the command line;
  - the MTP head is kept only when the source has mtp.* tensors (--preserve-mtp auto);
  - tokenizer fixes run on the output copy only and are recorded in <out>/agent-model-eval.json;
  - the sensitivity pass is the full-precision one, as for the Qwen3.8 oQ levels: the script refuses
    to start when oQ would fall back to its 4-bit proxy (too little free memory), unless --allow-proxy.

Run with the bench oMLX interpreter while no model server runs and the GPU lock is held:
  ~/Developer/mlsys/serving/omlx_0.7/.venv/bin/python -I scripts/oq_convert.py \
      --src <HF snapshot dir> --name Nanbeige4.3-33B-oQ5 --level 5 --fix drop-bos-post-processor

Writes ~/Developer/mlsys/models/oq/<name>, links ~/Developer/mlsys/models/hf-bench/<name> to it, and sets
max_context_window 131072 (and mtp_enabled when the head is kept) in the bench oMLX's model_settings.json.
--register-only skips the conversion; --fix-only re-applies the tokenizer fixes to an existing output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
MODELS = HOME / "Developer/mlsys/models"
SETTINGS = HOME / "Developer/mlsys/serving/omlx_0.7/.bench-home/model_settings.json"
CTX = 131072
FIXES = ("drop-bos-post-processor",)


def tilde(p: Path | str) -> str:
    s = str(p)
    return "~" + s[len(str(HOME)):] if s.startswith(str(HOME)) else s


def has_mtp(src: Path) -> bool:
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        keys = json.loads(idx.read_text())["weight_map"]
        return any(".mtp." in k or k.startswith("mtp.") for k in keys)
    return False


def port_listening(port: int) -> bool:
    r = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def drop_bos_post_processor(out: Path) -> dict:
    """Remove a BOS that the tokenizer's post-processor prepends on top of the chat template's own.

    oMLX renders the template to text and encodes it with special tokens on, so a TemplateProcessing
    post-processor whose single template starts with the BOS doubles it when the template already
    begins with that token (Nanbeige: <|im_start|> is both). The reference stacks tokenize chat prompts
    with add_special_tokens=False, which this reproduces.
    """
    tj, tc = out / "tokenizer.json", out / "tokenizer_config.json"
    for p in (tj, tc):
        # oQ copies tokenizer files with the HF cache blob's read-only mode. Only a private copy may be
        # made writable: a symlink or hard link would edit the shared cache blob.
        if p.is_symlink() or p.stat().st_nlink > 1:
            sys.exit(f"{tilde(p)} is linked to another file; refusing to edit it")
        p.chmod(p.stat().st_mode | 0o200)
    before = {"tokenizer.json": sha256(tj), "tokenizer_config.json": sha256(tc)}
    tok = json.loads(tj.read_text())
    pp = tok.get("post_processor") or {}
    if pp.get("type") != "TemplateProcessing":
        return {"fix": "drop-bos-post-processor", "changed": False, "reason": f"post_processor is {pp.get('type')}"}
    cfg = json.loads(tc.read_text())
    bos = cfg.get("bos_token")
    bos = bos.get("content") if isinstance(bos, dict) else bos
    keep = lambda seq: [e for e in seq if not ("SpecialToken" in e and e["SpecialToken"]["id"] == bos)]
    pp["single"], pp["pair"] = keep(pp.get("single", [])), keep(pp.get("pair", []))
    pp["special_tokens"] = {k: v for k, v in (pp.get("special_tokens") or {}).items() if k != bos}
    tok["post_processor"] = pp
    tj.write_text(json.dumps(tok, ensure_ascii=False))
    cfg["add_bos_token"] = False
    tc.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    after = {"tokenizer.json": sha256(tj), "tokenizer_config.json": sha256(tc)}
    return {"fix": "drop-bos-post-processor", "changed": True, "bos": bos, "sha256_before": before, "sha256_after": after}


def check_single_bos(out: Path) -> dict:
    """Encode a rendered chat prompt the way oMLX does and compare with apply_chat_template(tokenize=True)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out))
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "List the files."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    served = tok.encode(text)
    ref = tok.encode(text, add_special_tokens=False)
    return {"served_len": len(served), "reference_len": len(ref), "served_head": served[:3], "equal": served == ref}


def record(out: Path, entry: dict) -> None:
    p = out / "agent-model-eval.json"
    data = json.loads(p.read_text()) if p.exists() else {"events": []}
    data["events"].append({"time": dt.datetime.now().astimezone().isoformat(timespec="seconds"), **entry})
    p.write_text(json.dumps(data, indent=2) + "\n")


def register(out: Path, mtp: bool) -> None:
    if port_listening(8703):
        sys.exit("port 8703 is in use: stop the bench oMLX first (a running server rewrites model_settings.json)")
    link = MODELS / "hf-bench" / out.name
    if not link.is_symlink():
        link.symlink_to(Path("..") / "oq" / out.name)
        print(f"linked {tilde(link)} -> ../oq/{out.name}")
    data = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {"version": 1, "models": {}}
    entry = data.setdefault("models", {}).setdefault(out.name, {})
    want = {"max_context_window": CTX, **({"mtp_enabled": True} if mtp else {})}
    if any(entry.get(k) != v for k, v in want.items()):
        entry.update(want)
        tmp = SETTINGS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(SETTINGS)
        print(f"{want} set for {out.name} in {tilde(SETTINGS)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, help="BF16 HF snapshot directory")
    ap.add_argument("--name", required=True, help="output directory name = oMLX model id")
    ap.add_argument("--level", type=float, default=5)
    ap.add_argument("--preserve-mtp", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--fix", action="append", default=[], choices=FIXES)
    ap.add_argument("--allow-proxy", action="store_true", help="accept oQ's 4-bit proxy sensitivity pass")
    ap.add_argument("--register-only", action="store_true")
    ap.add_argument("--fix-only", action="store_true")
    a = ap.parse_args()
    out = MODELS / "oq" / a.name

    if not (a.register_only or a.fix_only):
        if a.src is None or not (a.src / "config.json").exists():
            sys.exit("--src must be an HF snapshot directory with config.json")
        if out.exists():
            sys.exit(f"{tilde(out)} exists; delete it first, or use --register-only / --fix-only")
        for port in (8500, 8703):
            if port_listening(port):
                sys.exit(f"port {port} is in use: no model server may run during a conversion")
        mtp = has_mtp(a.src) if a.preserve_mtp == "auto" else a.preserve_mtp == "yes"
        from omlx import oq

        ckpt = sum(f.stat().st_size for f in a.src.glob("*.safetensors"))
        # oQ's own pre-check (omlx/oq.py _calibration_memory_budget); it decides the proxy fallback the
        # same way inside quantize_oq_streaming, from the calibration footprint (>= the checkpoint size).
        from omlx.settings import get_system_memory

        cap = oq._calibration_memory_budget(ckpt, fallback_system_bytes=get_system_memory())
        print(f"{time.strftime('%T')} checkpoint {ckpt / 1e9:.1f} GB; memory: {cap}", flush=True)
        if cap and cap.get("requires_proxy") and not a.allow_proxy:
            sys.exit("oQ would use its 4-bit proxy for the sensitivity pass (too little free memory); "
                     "free memory or pass --allow-proxy")
        last = [0.0]

        def progress(phase: str, pct: float, detail: str = "", meta: dict | None = None) -> None:
            now = time.time()
            if now - last[0] >= 30 or pct >= 100:
                print(f"{time.strftime('%T')} {pct:5.1f}% {phase.split('|', 1)[0]} {detail}", flush=True)
                last[0] = now

        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"{time.strftime('%T')} oQ{a.level:g} (preserve_mtp={mtp}): {tilde(a.src)} -> {tilde(out)}", flush=True)
        t0 = time.time()
        oq.quantize_oq_streaming(str(a.src), str(out), a.level, 64, progress, False, None, None, "", "bfloat16",
                                 mtp, True)
        secs = time.time() - t0
        print(f"{time.strftime('%T')} done in {secs:.0f} s", flush=True)
        record(out, {"step": "convert", "source": tilde(a.src), "level": a.level, "group_size": 64,
                     "preserve_mtp": mtp, "seconds": round(secs), "memory": cap})
    else:
        mtp = (out / "config.json").exists() and has_mtp(out)

    for fix in a.fix:
        res = drop_bos_post_processor(out)
        res["check"] = check_single_bos(out)
        print(json.dumps(res["check"]))
        if not res["check"]["equal"]:
            sys.exit(f"{fix}: the served encoding still differs from the reference")
        record(out, {"step": "tokenizer-fix", **res})
    if not a.fix_only:
        register(out, mtp)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.exit(main())
