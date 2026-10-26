"""speed.py on synthetic agentbench raw lines: unit metrics, the M4 headline, M7 per-case gate, G0
checks, units.json merging, memory peaks, and the independent-render subprocess plumbing."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from speedkit import INCOMPLETE, NANBEIGE, header, m3_lines, m4_lines, make_model, req

from modeleval import paths, speed

# ------------------------------------------------------------------ units and plumbing
def test_timed_units_skip_the_thinking_cell_without_speculative_decoding():
    assert speed.timed_units(make_model()) == list(speed.UNIT_ORDER)
    units = speed.timed_units(make_model(**NANBEIGE))
    assert "m3-think-16k" not in units and units[0] == "g5+m3-64k" and units[-3:] == ["m4-1", "m4-2", "m4-3"]
    assert speed.required_prompts(make_model(**NANBEIGE)) == ["m3-off-110000", "m3-off-16000", "m3-off-32000",
                                                             "m3-off-64000"]
    assert "m3-on-16000" in speed.required_prompts(make_model())


def test_server_root_and_sampling():
    assert speed.server_root("http://127.0.0.1:8703/v1") == "http://127.0.0.1:8703"
    assert speed.server_root("http://127.0.0.1:8703/v1/") == "http://127.0.0.1:8703"
    assert speed.server_root("http://127.0.0.1:8791") == "http://127.0.0.1:8791"
    assert speed.sampling_body(make_model(**NANBEIGE)) == {"temperature": 1.0, "top_p": 0.95, "top_k": 0}


# ------------------------------------------------------------------ M4
def test_m4_headline_uses_ttft_sum_and_a_fixed_output_length():
    lines = m4_lines(over={5: dict(ok=False, error=INCOMPLETE, completion_tokens=150, decode_s=3.0,
                                   completion_tokens_source="tokenizer", cached_tokens=None, e2e_s=9.0),
                           7: dict(finish_reason="length", completion_tokens=150, decode_s=2.0)})
    m, notes = speed.m4_metrics(lines)
    # R over the 20 steps with server usage: 19 x 100 tokens in 19 x 2 s, plus 150 tokens in 2 s
    rate = (19 * 100 + 150) / (19 * 2.0 + 2.0)
    assert m["decode_rate_tps"] == pytest.approx(rate) and m["rate_steps"] == 20
    assert m["ttft_sum_s"] == pytest.approx(21.0)  # the cut step is a valid latency sample
    assert m["headline_s"] == pytest.approx(21.0 + 21 * 100 / rate)
    assert m["raw_total_s"] == pytest.approx(20 * 3.0 + 9.0)
    assert m["cap_hits"] == 2 and m["cap_hits_by_kind"] == {"length": 1, "incomplete_tool_call": 1}
    assert m["steps"] == 21 and not notes
    assert m["mean_completion_tokens"] == pytest.approx((19 * 100 + 150 + 150) / 21)
    metrics, valid, _ = speed.unit_metrics("m4-1", lines)
    assert valid and metrics["m4"]["headline_s"] == m["headline_s"]


def test_m4_cold_check_and_hard_errors_invalidate():
    _, valid, notes = speed.unit_metrics("m4-2", m4_lines(over={0: dict(cached_tokens=5698)}))
    assert not valid and any("step 0 reused 5698" in n for n in notes)
    hard = m4_lines(over={3: dict(ok=False, error="HTTP 500: boom", ttft_s=None, e2e_s=0.1)})
    m, valid, _ = speed.unit_metrics("m4-3", hard)
    assert not valid and m["m4"]["ttft_sum_s"] is None and m["m4"]["headline_s"] is None
    _, valid, notes = speed.unit_metrics("m4-1", m4_lines(over={0: dict(ok=False, error=INCOMPLETE, cached_tokens=None)}))
    assert valid and any("no usage" in n for n in notes)
    _, valid, _ = speed.unit_metrics("m4-1", m4_lines(n=20))
    assert not valid  # 21 steps expected


# ------------------------------------------------------------------ M3, G5, M1
def test_m3_cell_decode_median_and_seconds_per_100():
    m, valid, notes = speed.unit_metrics("m3-32k", m3_lines(ctx=32000))
    c = m["m3"]
    assert valid and not notes
    assert c["decode_tps"]["median"] == 31.0 and c["s_per_100"] == pytest.approx(100 / 31)
    assert (c["ctx"], c["cap"], c["trials"], c["hit_cap"], c["warmup"]) == (32000, 150, 3, 3, "normal")
    bad = m3_lines()
    bad[2]["metrics"].update(ok=False, error="HTTP 500: x", decode_tps=None)
    _, valid, notes = speed.unit_metrics("m3-16k", bad)
    assert not valid and "2/3 trials ok" in notes[0]


def test_g5_and_64k_cell_share_a_unit():
    g5 = req("gate", "", probe="G5.m3-off-64000", t_send=2000.0,
             m=dict(ttft_s=60.0, e2e_s=61.0, prompt_tokens=64012, cached_tokens=0))
    cell = [ln for ln in m3_lines(ctx=64000) if ln["phase"] == "trial"]
    cell[0]["warmup"] = "reused"
    m, valid, notes = speed.unit_metrics("g5+m3-64k", [g5] + cell)
    assert valid and not notes
    assert m["g5"]["ttft_s"] == 60.0 and m["g5"]["t_start"] == 2000.0 and m["g5"]["t_end"] == 2061.0
    assert m["m3"]["ctx"] == 64000 and m["m3"]["warmup"] == "reused" and m["m3"]["trial0_cached_tokens"] == 63000
    fallback = req("m3", "warmup", warmup="fallback", cell={"ctx": 64000, "phase": "warmup"}, m=dict(cached_tokens=12))
    cell[0].pop("warmup")
    m, valid, notes = speed.unit_metrics("g5+m3-64k", [g5, fallback] + cell)
    assert valid and m["m3"]["warmup"] == "fallback" and m["m3"]["trial0_cached_tokens"] == 12
    assert "not cached" in notes[0]
    g5["metrics"]["cached_tokens"] = 60000
    _, valid, notes = speed.unit_metrics("g5+m3-64k", [g5] + cell)
    assert not valid and "not cold" in notes[0]


def test_m1_warm_and_cold():
    lines = [req("m1", "warmup", m=dict(cached_tokens=0))]
    lines += [req("m1", "warm", trial=i, m=dict(ttft_s=0.3 + i / 100, cached_tokens=14900)) for i in range(5)]
    lines += [req("m1", "cold", trial=0, m=dict(ttft_s=9.0, cached_tokens=0))]
    m, valid, _ = speed.unit_metrics("m1", lines)
    assert valid and m["m1"]["warm_ttft_s"]["median"] == pytest.approx(0.32)
    assert m["m1"]["cold_ttft_s"]["median"] == 9.0 and m["m1"]["warm_cached_tokens"] == 14900
    lines[-1]["metrics"]["cached_tokens"] = 5698  # a system prompt rendered first was a cache hit
    _, valid, notes = speed.unit_metrics("m1", lines)
    assert not valid and "cold trial reused 5698" in notes[0]
    lines[-1]["metrics"].update(ok=False, error=INCOMPLETE, cached_tokens=None)
    _, valid, notes = speed.unit_metrics("m1", lines)
    assert valid and "no usage" in notes[0]


# ------------------------------------------------------------------ M7 per-case gate
def test_m7_case_gate_fails_at_three_of_five():
    lines = []
    for case, misses in (("m7-01-single", 2), ("m7-18-parallel", 3), ("m7-21-no-call", 0)):
        for i in range(5):
            ok = i >= misses
            lines.append(req("m7", "trial", trial=i, cell={"case": case, "category": "x"},
                             judgement={"pass": ok, "reasons": [] if ok else ["wrong argument"]}))
    lines.append({"type": "skip", "suite": "m7", "cell": {"case": "m7-22-image-read", "category": "image"},
                  "reason": "N/A"})
    g = speed.m7_case_gate(lines)
    assert g["failing"] == ["m7-18-parallel"] and g["passed_cases"] == 2 and g["scored_cases"] == 3
    assert g["na"] == ["m7-22-image-read"] and g["trials"] == 15 and g["misses"] == 5
    assert g["cases"]["m7-01-single"]["pass"] and g["cases"]["m7-18-parallel"]["reasons"] == ["wrong argument"]


# ------------------------------------------------------------------ G0 checks
class FX:
    system_prompt = "You are an agent.\nUse the tools."


def history() -> list[dict]:
    return [{"role": "system", "content": FX.system_prompt}, {"role": "user", "content": "Turn on format on save."},
            {"role": "assistant", "content": "Clicking.", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "computer_click", "arguments": json.dumps({"x": 287, "y": 361})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Clicked at (287, 361). request_id: 123456789012"},
            {"role": "user", "content": "Screenshot after the click."}]


QWEN_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rendered(system: str = FX.system_prompt, tail: str = QWEN_TAIL, call: bool = True, result: bool = True) -> str:
    s = f"<|im_start|>system\n# Tools ...\n\n{system}<|im_end|>\n<|im_start|>user\nTurn on format on save.<|im_end|>\n"
    if call:
        s += ("<|im_start|>assistant\nClicking.\n\n<tool_call>\n<function=computer_click>\n<parameter=x>\n287\n"
              "</parameter>\n<parameter=y>\n361\n</parameter>\n</function>\n</tool_call><|im_end|>\n")
    if result:
        s += "<|im_start|>user\n<tool_response>\nClicked at (287, 361). request_id: 123456789012\n</tool_response>"
    return s + "\nScreenshot after the click.<|im_end|>\n" + tail


def checks(kind: str, ids: list[int], text: str) -> dict[str, bool]:
    return {c["check"]: c["ok"] for c in speed.g0_properties(kind, ids, text, FX, history())}


def test_g0_properties_pass_on_a_clean_render():
    got = checks("history", [10, 11, 12], rendered())
    assert all(got.values()) and len(got) == 5


def test_g0_properties_catch_each_failure():
    assert not checks("tools", [7, 7, 9], rendered())["single BOS"]
    assert not checks("tools", [1, 2], rendered(system="Some default prompt."))["system prompt verbatim"]
    loose = checks("tools", [1, 2], rendered(system="You are an agent.  Use the tools."))
    assert loose["system prompt verbatim"]  # whitespace differences only
    assert not checks("tools", [1, 2], rendered(tail="<|im_start|>assistant\n<think>\n"))["thinking-off suffix"]
    assert not checks("history", [1, 2], rendered(call=False))["tool-call turn rendered"]
    assert not checks("history", [1, 2], rendered(result=False))["tool-response turn rendered"]


def test_thinking_suffix_and_first_difference():
    assert speed.thinking_off_suffix("x<|im_start|>assistant\n")[0]  # a template without think markup
    assert not speed.thinking_off_suffix("<think>a</think> answer <|im_start|>assistant\n")[0]
    assert speed.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert speed.first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert speed.first_difference([5, 1, 2], [1, 2]) == 0
    assert speed.first_difference([1, 2], [1, 2, 3]) == 2


def test_g0_usage_check():
    g0 = {"tools_count": 14949, "requests": {}, "checks": []}
    assert speed.g0_usage(g0, req("gate", "", m=dict(prompt_tokens=14949)))["ok"]
    bad = speed.g0_usage(g0, req("gate", "", m=dict(prompt_tokens=14950)))
    assert not bad["ok"] and "14950" in bad["detail"]
    assert not speed.g0_usage(g0, None)["ok"]


# ------------------------------------------------------------------ independent render subprocess
def fake_python(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "fakepy"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_independent_render_plumbing(tmp_path, monkeypatch):
    log = tmp_path / "args.txt"
    py = fake_python(tmp_path, f'printf "%s\\n" "$@" > {log}\ncat > {tmp_path}/stdin.json\n'
                               'echo \'{"ids": [151644, 8948, 198], "transformers": "5.17.0"}\'\n')
    monkeypatch.setattr(speed, "RENDER_PYTHON", py)
    out = speed.independent_render(tmp_path / "tok", [{"role": "user", "content": "hi"}], None,
                                   {"enable_thinking": False})
    assert out["ids"] == [151644, 8948, 198]
    args = log.read_text().splitlines()
    assert args[:3] == ["-I", "-B", "-c"] and args[-1] == str(tmp_path / "tok")
    sent = json.loads((tmp_path / "stdin.json").read_text())
    assert sent["chat_template_kwargs"] == {"enable_thinking": False} and sent["tools"] is None


def test_independent_render_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(speed, "RENDER_PYTHON", fake_python(tmp_path, 'echo "no transformers" >&2\nexit 3\n'))
    with pytest.raises(RuntimeError, match="exit 3"):
        speed.independent_render(tmp_path, [{"role": "user", "content": "hi"}], None, {})


@pytest.mark.skipif(os.environ.get("MODELEVAL_REAL_RENDER") != "1", reason="opt-in: runs the real oMLX venv")
def test_independent_render_real_omlx_venv():
    tok = paths.MLSYS / "models/oq/Qwen3.8-27B-oQ5-mtp"
    if not (paths.OMLX_PY.exists() and (tok / "tokenizer.json").exists()):
        pytest.skip("oMLX venv or the oQ5 tokenizer is absent")
    out = speed.independent_render(tok, [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}], None,
                                   {"enable_thinking": False})
    assert len(out["ids"]) > 5 and out["ids"][0] != out["ids"][1]


# ------------------------------------------------------------------ units.json and memory
def ctx_for(tmp_path: Path, **model_kw):
    return speed.SpeedContext(make_model(**model_kw), "http://127.0.0.1:8790/v1", 1, tmp_path / "run",
                              tmp_path / "tok", tmp_path / "unverified")


def unit_result(ctx, unit: str, attempt: int, valid: bool, t0: float, t1: float, metrics: dict) -> speed.UnitResult:
    raw = speed.speed_dir(ctx.run_dir) / f"{unit}-{attempt}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("{}\n")
    return speed.UnitResult(unit, attempt, t0, t1, raw, metrics, valid, "")


def test_units_json_merges_attempts_and_verdicts(tmp_path):
    ctx = ctx_for(tmp_path)
    speed.record_unit(ctx, unit_result(ctx, "m1", 1, True, 10, 20, {"m1": {"x": 1}}), {"macos_build": "26B5101f"})
    speed.record_unit(ctx, unit_result(ctx, "m1", 2, True, 30, 40, {"m1": {"x": 2}}))
    speed.record_unit(ctx, unit_result(ctx, "m1", 2, True, 31, 41, {"m1": {"x": 3}}))  # same attempt: replaced
    doc = speed.load_units(ctx.run_dir)
    rec = doc["units"]["m1"]
    assert [a["attempt"] for a in rec["attempts"]] == [1, 2] and rec["verdict"] is None
    assert rec["attempts"][1]["metrics"]["m1"]["x"] == 3 and rec["attempts"][0]["raw_file"] == "speed/m1-1.jsonl"
    assert doc["model"] == "qwen-ref" and doc["model_info"]["display"] and doc["system"]["macos_build"] == "26B5101f"
    assert doc["protocol_version"] == paths.PROTOCOL_VERSION and doc["run_id"] == "run"
    # attempt 1 disturbed and moved; attempt 2 clean (the way run.py records them)
    moved = tmp_path / "unverified" / "m1-1.jsonl"
    speed.record_verdict(ctx.run_dir, "m1", 1, {"clean": False, "reasons": ["mds 80 %"], "peak_foreign_cpu": 80.0},
                         moved)
    assert speed.load_units(ctx.run_dir)["units"]["m1"]["verdict"] is None
    speed.record_verdict(ctx.run_dir, "m1", 2, {"clean": True, "reasons": []})
    rec = speed.load_units(ctx.run_dir)["units"]["m1"]
    assert rec["attempts"][0]["raw_file"] == "../unverified/m1-1.jsonl"
    assert rec["attempts"][0]["verdict"] == "disturbed" and rec["attempts"][0]["reasons"] == ["mds 80 %"]
    assert rec["verdict"] == "clean" and speed.kept_attempt(rec, ctx.run_dir)["attempt"] == 2
    with pytest.raises(KeyError):
        speed.record_verdict(ctx.run_dir, "m1", 9, {"clean": True})
    assert "/Users/" not in speed.units_path(ctx.run_dir).read_text()


def test_verdicts_mark_invalid_and_disturbed_units(tmp_path):
    ctx = ctx_for(tmp_path)
    speed.record_unit(ctx, unit_result(ctx, "m4-1", 1, False, 1, 2, {}))
    speed.record_verdict(ctx.run_dir, "m4-1", 1, {"clean": True})  # quiet, but the cold request hit the cache
    assert speed.load_units(ctx.run_dir)["units"]["m4-1"]["attempts"][0]["verdict"] == "invalid"
    for i in (1, 2, 3):
        speed.record_unit(ctx, unit_result(ctx, "m1", i, True, i, i + 1, {}))
        speed.record_verdict(ctx.run_dir, "m1", i, "disturbed")
    rec = speed.load_units(ctx.run_dir)["units"]["m1"]
    assert rec["verdict"] == "disturbed x3" and speed.kept_attempt(rec, ctx.run_dir) is None


def test_minor_verdicts_keep_the_attempt(tmp_path):
    """DESIGN §5.4: a minor verdict keeps the attempt with its flags; a major one is retried."""
    ctx = ctx_for(tmp_path)
    speed.record_unit(ctx, unit_result(ctx, "m1", 1, True, 1, 2, {}))
    speed.record_verdict(ctx.run_dir, "m1", 1, {"clean": False, "severity": "major", "reasons": ["lock"],
                                                "major_reasons": ["lock"]})
    speed.record_unit(ctx, unit_result(ctx, "m1", 2, True, 3, 4, {}))
    speed.record_verdict(ctx.run_dir, "m1", 2, {"clean": False, "severity": "minor", "reasons": ["busy"],
                                                "minor_reasons": ["busy"], "minor_samples": 1})
    rec = speed.load_units(ctx.run_dir)["units"]["m1"]
    assert [a["verdict"] for a in rec["attempts"]] == ["disturbed", "minor"]
    assert [a["kept"] for a in rec["attempts"]] == [False, True]
    assert rec["verdict"] == "minor" and rec["attempts"][1]["minor_reasons"] == ["busy"]
    assert speed.kept_attempt(rec, ctx.run_dir)["attempt"] == 2
    assert speed.kept_attempt({"attempts": [{"attempt": 1, "verdict": "minor", "valid": True}]})["attempt"] == 1
    assert speed.kept_attempt({"attempts": [{"attempt": 1, "valid": True,
                                             "verdict": {"clean": False, "severity": "minor"}}]})["attempt"] == 1


def test_kept_attempt_reads_run_py_records(tmp_path):
    """run.py's own attempt records: verdict strings, `kept`, raw_file as a ~ path."""
    rec = {"verdict": "clean", "attempts": [
        {"attempt": 1, "verdict": "disturbed", "kept": False, "valid": True, "raw_file": "~/x/_unverified/m1-1.jsonl"},
        {"attempt": 2, "verdict": "clean", "kept": True, "valid": True, "raw_file": "~/x/results/m/r/speed/m1-2.jsonl"}]}
    assert speed.kept_attempt(rec)["attempt"] == 2
    assert speed.kept_attempt({"verdict": "invalid", "attempts": [
        {"attempt": 1, "verdict": "invalid", "kept": False, "valid": False, "raw_file": "x"}]}) is None


def test_kept_attempt_without_verdicts_uses_the_files_left_in_speed(tmp_path):
    ctx = ctx_for(tmp_path)
    a = [{"attempt": 1, "valid": True, "raw_file": "speed/m1-1.jsonl", "verdict": None},
         {"attempt": 2, "valid": False, "raw_file": "speed/m1-2.jsonl", "verdict": None}]
    assert speed.kept_attempt(a, ctx.run_dir) is None  # file not there: run.py moved it
    (speed.speed_dir(ctx.run_dir)).mkdir(parents=True)
    (speed.speed_dir(ctx.run_dir) / "m1-1.jsonl").write_text("{}")
    (speed.speed_dir(ctx.run_dir) / "m1-2.jsonl").write_text("{}")
    assert speed.kept_attempt(a, ctx.run_dir)["attempt"] == 1  # attempt 2 is invalid


def test_memory_peaks_by_window(tmp_path):
    ctx = ctx_for(tmp_path)
    sd = speed.speed_dir(ctx.run_dir)
    sd.mkdir(parents=True)
    rows = ["t\ttree_bytes\tsys_delta_bytes\tpressure\tswapouts_delta\tlabel"]
    gib = 2**30
    for t, fp, sysd, pr, sw in [(100, 1, 2, 1, 0), (110, 20, 22, 1, 0), (199, 21, 23, 1, 0), (205, 22, 24, 1, 0),
                                (210, 30, 40, 1, 0), (220, 25, 30, 1, 0), (400, 40, 70, 2, 5), (500, 26, 31, 1, 0)]:
        rows.append(f"{t}\t{fp * gib}\t{sysd * gib}\t{pr}\t{sw}\t")
    (sd / "memory.tsv").write_text("\n".join(rows) + "\n")
    speed.record_functional(ctx, speed.FunctionalResult(True, None, {}, None, [], 200.0, 250.0))
    speed.record_unit(ctx, unit_result(ctx, "g5+m3-64k", 1, True, 205, 260,
                                       {"g5": {"t_start": 205.0, "t_end": 212.0}, "m3": {}}))
    speed.record_unit(ctx, unit_result(ctx, "m3-110k", 1, True, 390, 410, {"m3": {}}))
    mem = speed.finalize(ctx.run_dir)
    assert mem["after_load"] == 21 * gib  # last sample before the functional block
    assert mem["g5_peak"] == 30 * gib and mem["g5_peak_system"] == 40 * gib
    assert mem["m3_110k_peak"] == 40 * gib and mem["peak"] == 40 * gib and mem["peak_system"] == 70 * gib
    assert mem["max_pressure"] == 2 and mem["swapouts"] == 5
    assert speed.load_units(ctx.run_dir)["memory"]["peak"] == 40 * gib


def test_memory_tsv_units_and_labels(tmp_path):
    p = tmp_path / "memory.tsv"
    p.write_text("# t\tfootprint_mb\tlabel\n# a comment\n1.0\t1024\tloaded\n2.0\t2048\t\nbad\t1\t\n")
    s = speed.read_memory_tsv(p)
    assert [x["footprint"] for x in s] == [2**30, 2 * 2**30] and s[0]["label"] == "loaded"
    assert speed.read_memory_tsv(tmp_path / "absent.tsv") == []


def test_system_from_header():
    s = speed._system_from_header([header()])
    assert s["macos_build"] == "26B5101f" and s["xcode_build"] == "27A9275" and s["macos"] == "27.2"


def test_memory_tsv_from_core_sampler_columns(tmp_path):
    """core's MemorySampler layout: system delta = wired + compressor + file-backed; swap-outs are
    cumulative per launch; the "after-load" label marks the loaded server."""
    try:
        from modeleval.server import MEMORY_COLUMNS
    except ImportError:
        MEMORY_COLUMNS = ("iso", "epoch", "launch", "label", "source", "tree_bytes", "wired_delta",
                          "compressor_delta", "file_backed_delta", "pressure", "swapouts_delta")
    gib = 2**30
    rows = ["\t".join(MEMORY_COLUMNS)]
    data = [(100, 1, "load", 1, 0, 0, 0, 1, 0), (102, 1, "after-load", 20, 18, 1, 2, 1, 0),
            (300, 1, "m3-110k", 45, 40, 3, 10, 2, 4), (310, 1, "idle", 30, 25, 1, 3, 1, 6),
            (400, 2, "after-load", 21, 19, 0, 1, 1, 0), (500, 2, "capability", 25, 20, 0, 2, 1, 3)]
    for t, launch, label, tree, wired, comp, fileb, pr, sw in data:
        row = {"iso": "2026-10-27T01:00:00", "epoch": t, "launch": launch, "label": label, "source": "footprint",
               "tree_bytes": tree * gib, "wired_delta": wired * gib, "compressor_delta": comp * gib,
               "file_backed_delta": fileb * gib, "pressure": pr, "swapouts_delta": sw}
        rows.append("\t".join(str(row[c]) for c in MEMORY_COLUMNS))
    ctx = ctx_for(tmp_path)
    sd = speed.speed_dir(ctx.run_dir)
    sd.mkdir(parents=True)
    (sd / "memory.tsv").write_text("\n".join(rows) + "\n")
    speed.record_unit(ctx, unit_result(ctx, "m3-110k", 1, True, 290, 305, {"m3": {}}))
    speed.record_verdict(ctx.run_dir, "m3-110k", 1, "clean")
    mem = speed.memory_summary(ctx.run_dir)
    assert mem["after_load"] == 20 * gib and mem["after_load_system"] == 21 * gib
    assert mem["m3_110k_peak"] == 45 * gib and mem["m3_110k_peak_system"] == 53 * gib
    assert mem["peak"] == 45 * gib and mem["max_pressure"] == 2
    assert mem["swapouts"] == 6 + 3  # the last value of each launch


# ------------------------------------------------------------------ direct tool calls, invalid kinds, M7 incomplete
def test_m1_uses_the_servers_ttft_for_trials_that_open_with_a_tool_call():
    """oMLX streams a tool call as one delta after generating it: the client's TTFT then includes the
    call's decode. The server's time_to_first_token is used instead; without it the row is flagged."""
    lines = [req("m1", "warmup", m=dict(cached_tokens=0))]
    lines += [req("m1", "warm", trial=i, m=dict(ttft_s=1.85, first_kind="tool_call", n_token_chunks=1, decode_s=0.0,
                                               server_ttft_s=0.31, server_generation_s=1.5, cached_tokens=14900))
              for i in range(5)]
    lines += [req("m1", "cold", trial=0, m=dict(ttft_s=9.0, cached_tokens=0))]
    m, valid, notes = speed.unit_metrics("m1", lines)
    assert valid and m["m1"]["warm_ttft_s"]["median"] == pytest.approx(0.31) and m["m1"]["cold_ttft_s"]["median"] == 9.0
    assert m["m1"]["ttft_sources"] == {"client": 1, "server": 5} and m["m1"]["ttft_includes_decode"] == 0
    for ln in lines[1:6]:
        ln["metrics"]["server_ttft_s"] = None
    m, valid, notes = speed.unit_metrics("m1", lines)
    assert m["m1"]["warm_ttft_s"]["median"] == pytest.approx(1.85) and m["m1"]["ttft_includes_decode"] == 5
    assert any("includes the decode" in n for n in notes)


def test_m4_direct_tool_call_steps_count_in_r_with_the_servers_generation_time():
    direct = dict(first_kind="tool_call", n_token_chunks=1, decode_s=0.0, ttft_s=3.0, server_ttft_s=1.0,
                  server_generation_s=2.0)
    lines = m4_lines(over={k: direct for k in range(21)})
    m, valid, _ = speed.unit_metrics("m4-1", lines)
    m4 = m["m4"]
    assert valid and m4["ttft_sum_s"] == pytest.approx(21.0) and m4["direct_calls"] == 21
    assert m4["decode_rate_tps"] == pytest.approx(50.0) and m4["headline_s"] == pytest.approx(21 + 21 * 100 / 50)
    no_timing = {k: {**direct, "server_ttft_s": None, "server_generation_s": None} for k in range(21)}
    m, valid, notes = speed.unit_metrics("m4-1", m4_lines(over=no_timing))
    assert m["m4"]["headline_s"] is None and m["m4"]["ttft_includes_decode"] == 21
    assert any("no step has a decode time" in n for n in notes)


def test_invalid_attempts_say_whether_a_repeat_can_fix_them():
    lines = m4_lines(over={0: dict(cached_tokens=5698)})   # a cold step 0 that hit the cache
    m, valid, _ = speed.unit_metrics("m4-1", lines)
    assert not valid and m["invalid_kind"] == "cache-hit"
    lines = m4_lines(over={7: dict(ok=False, ttft_s=None, error=INCOMPLETE, cached_tokens=None)})  # nothing streamed
    m, valid, _ = speed.unit_metrics("m4-1", lines)
    assert not valid and m["invalid_kind"] == "incomplete"
    m, valid, _ = speed.unit_metrics("m4-1", m4_lines())
    assert valid and "invalid_kind" not in m


def test_m7_case_cut_short_by_an_error_is_incomplete_never_a_pass():
    lines = []
    for case, n in (("m7-01-single", 5), ("m7-02-cut", 2)):
        for i in range(n):
            lines.append(req("m7", "trial", trial=i, cell={"case": case, "category": "x"},
                             judgement={"pass": i != 0, "reasons": [] if i else ["wrong argument"]}))
    g = speed.m7_case_gate(lines)
    assert g["incomplete"] == ["m7-02-cut"] and not g["cases"]["m7-02-cut"]["pass"]
    assert g["passed_cases"] == 1 and g["scored_cases"] == 1 and g["failing"] == []


def test_server_side_gate_errors_are_told_apart_from_cap_cuts():
    def line(err):
        return {"type": "request", "suite": "gate", "probe": "G1.hermes", "record": {"error": err}}

    assert speed.server_error([line(None)]) is None
    assert speed.server_error([line({"type": "transport", "detail": "ConnectError: refused"})]).startswith("transport")
    assert speed.server_error([line({"type": "http", "status": 503, "body": "busy"})]).startswith("http 503")
    assert speed.server_error([line({"type": "http", "status": 400, "body": "bad"})]) is None
    cut = {"type": "stream_error", "detail": {"code": "incomplete_tool_call"}}
    assert speed.server_error([line(cut)]) is None
    assert speed.server_error([line({"type": "stream_error", "detail": {"message": "Metal OOM"}})]).startswith("stream")
