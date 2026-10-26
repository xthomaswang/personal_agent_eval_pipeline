import json

import pytest

from agentbench.cli import main
from agentbench.summarize import select, summarize
from agentbench.suites.common import upsert_section

from conftest import MY_PID, cli_args, start_mock


@pytest.fixture(scope="module")
def run_dir(built, tmp_path_factory):
    root, _ = built
    out = tmp_path_factory.mktemp("out")
    srv, _ = start_mock(root, "generic")
    try:
        rc = main(["run-all"] + cli_args(srv.url, root, out, "omlx", "omlx", "--suites", "gate,m1,m2,m3,m4,m5,m6,m7",
                                         "--spec-label", "off", "--minutes", "0.02", "--mem-interval", "0.5",
                                         "--server-pid", MY_PID))
    finally:
        srv.stop()
    assert rc == 0
    return out


def test_run_all_writes_summary_with_every_suite(run_dir):
    md = (run_dir / "omlx" / "summary.md").read_text()
    for title in ("Gate G1-G5", "M1 cached-prefix TTFT", "M2 cold prefill", "M3 short structured decode",
                  "M4 20-step trajectory", "M5 concurrency", "M6 memory and stability", "M7 tool-call correctness"):
        assert f"### {title}" in md, title
    assert md.count("<!-- agentbench:begin summary t -->") == 1


def test_tables_have_median_min_max_and_m4_per_step(run_dir):
    md = summarize(run_dir / "omlx", "t")
    assert "| warm | 2 | " in md and "| cold | 1 | " in md
    warm_row = next(line for line in md.splitlines() if line.startswith("| warm |"))
    assert "(" in warm_row and "-" in warm_row  # median (min-max)
    m4 = md[md.index("### M4"):md.index("### M5")]
    rows = [line for line in m4.splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert [r.split("|")[1].strip().split()[0] for r in rows] == ["0", "1", "2", "3"]
    assert "TTFT slope" in m4
    assert "| 2,000 |" in md and "| off | off | 2,000 | 50 |" in md and "Passed 6/6." in md


def test_summarize_write_is_idempotent_and_keeps_other_text(run_dir, tmp_path):
    path = run_dir / "omlx" / "summary.md"
    path.write_text("# omlx: results\n\nHand-written notes stay.\n\n" + path.read_text().split("\n", 1)[1])
    assert main(["summarize", "--framework", "omlx", "--tag", "t", "--write", "--out-root", str(run_dir)]) == 0
    assert main(["summarize", "--framework", "omlx", "--write", "--out-root", str(run_dir)]) == 0
    text = path.read_text()
    assert "Hand-written notes stay." in text and text.count("<!-- agentbench:begin summary t -->") == 1


def test_upsert_section(tmp_path):
    p = tmp_path / "x.md"
    upsert_section(p, "gate a", "one", "Title")
    upsert_section(p, "gate b", "two", "Title")
    upsert_section(p, "gate a", "three", "Title")
    t = p.read_text()
    assert t.startswith("# Title") and "three" in t and "one" not in t and t.index("three") < t.index("two")


def test_summarize_all_runs_pools_files(run_dir):
    md = summarize(run_dir / "omlx", "t", all_runs=True)
    assert "### M1 cached-prefix TTFT" in md


def test_select_skips_files_without_a_tag(run_dir, tmp_path):
    """Core AI's ab-* direct-call files have a provenance header with a suite but no tag."""
    ab = tmp_path / "ab-coreai-p512-g1024-n5-20261020-092624.jsonl"
    ab.write_text(json.dumps({"type": "header", "suite": "ab", "created": "2026-10-20T09:26:24"}) + "\n")
    files = sorted((run_dir / "omlx" / "raw").glob("*.jsonl")) + [ab]
    assert set(select(files)) == {"t"} and set(select(files, "t")) == {"t"}


def test_cap_truncated_streams_count_as_latency_samples():
    """oMLX ends a stream with `incomplete_tool_call` when the token cap cuts a tool call; its TTFT
    and e2e are still complete measurements (llama.cpp reports the same case as finish=length)."""
    from agentbench.summarize import cap_truncated, s_m4

    def line(step, ok, err=None, ttft=1.0, e2e=2.0):
        return {"type": "request", "phase": "start" if step == 0 else "step", "step": step, "run": 0,
                "metrics": {"ok": ok, "error": err, "ttft_s": ttft, "e2e_s": e2e, "cached_tokens": 90 if ok else None,
                            "prompt_tokens": 100 if ok else None, "prompt_tokens_local": 100}}

    trunc = ("stream_error: {'message': 'Model output contains an unrecoverable tool call.', "
             "'code': 'incomplete_tool_call'}")
    lines = [line(0, True), line(1, False, trunc, ttft=1.5), line(2, True, ttft=2.0),
             line(3, False, "HTTP 500: boom", ttft=None, e2e=None)]
    assert cap_truncated(lines[1]["metrics"]) and not cap_truncated(lines[3]["metrics"])
    md = s_m4([{"lines": lines + [{"type": "summary", "run_total_e2e_s": [6.0]}]}])
    rows = {r.split("|")[1].strip().split()[0]: r.split("|") for r in md.splitlines() if r.startswith("| ") and r[2].isdigit()}
    assert rows["1"][2].strip() == "1" and rows["1"][-2].strip() == "0"  # counted, not an error
    assert rows["3"][2].strip() == "0" and rows["3"][-2].strip() == "1"  # a real error stays an error
    assert "slope over steps 1-3: 0.5000 s/step" in md
    assert "(steps 2-3: - s/step)" in md  # one point left: no slope
    assert "1 of 4 requests ended at the token cap" in md


def test_lithos_cap_truncation_counts_like_omlx():
    """lithos-metal reports the cap cutting a tool call by message ("incomplete tool call; increase
    the output budget", code invalid_tool_call). It is the same case as oMLX's code; a genuinely
    invalid tool call ("retry the request") stays an error."""
    from agentbench.summarize import cap_truncated, truncation_note

    def m(err):
        return {"ok": False, "error": err, "ttft_s": 5.5, "e2e_s": 8.3}

    lithos = ("stream_error: {'message': 'The model generated an incomplete tool call; increase the output "
              "budget', 'type': 'server_error', 'code': 'invalid_tool_call'}")
    invalid = ("stream_error: {'message': 'The model generated an invalid tool call; retry the request', "
               "'type': 'server_error', 'code': 'invalid_tool_call'}")
    assert cap_truncated(m(lithos)) and not cap_truncated(m(invalid))
    note = truncation_note([{"metrics": m(lithos)}, {"metrics": m(invalid)}])
    assert "1 of 2 requests ended at the token cap" in note and '("incomplete tool call")' in note
    omlx = "stream_error: {'message': 'Model output contains an unrecoverable tool call.', 'code': 'incomplete_tool_call'}"
    assert "`incomplete_tool_call` stream error" in truncation_note([{"metrics": m(omlx)}])


def test_m4_slope_with_and_without_step_1():
    """oMLX's prefix cache does not reuse the step-0 prompt at step 1, so step 1 is a near-full
    re-prefill (~17.9 s against ~3.7 s from step 2 on) that skews the slope over steps 1-N. The
    summary also gives the slope over steps 2-N."""
    from agentbench.summarize import s_m4

    def line(step, ttft, err=None):
        ok = err is None
        return {"type": "request", "phase": "start" if step == 0 else "step", "step": step, "run": 0,
                "metrics": {"ok": ok, "error": err, "ttft_s": ttft, "e2e_s": None if ttft is None else ttft + 5.0,
                            "cached_tokens": 90 if ok else None, "prompt_tokens": 100 if ok else None,
                            "prompt_tokens_local": 100}}

    trunc = "stream_error: {'code': 'incomplete_tool_call'}"
    # steps 2, 3, 5, 6 lie on 3 + 0.5 k; step 0 (cold start) and step 4 (no latency sample) are left out
    lines = [line(0, 25.0), line(1, 18.0), line(2, 4.0), line(3, 4.5, trunc), line(4, None, "HTTP 500: boom"),
             line(5, 5.5), line(6, 6.0)]
    md = s_m4([{"lines": lines}])
    assert "TTFT slope over steps 1-6: -1.5233 s/step (steps 2-6: 0.5000 s/step); total e2e" in md
    md = s_m4([{"lines": lines[:2]}])  # no step 2: no second slope
    assert "TTFT slope over steps 1-1: - s/step; total e2e" in md


def test_capabilities_md_follows_verified_gate_runs(run_dir, tmp_path):
    """A full `summarize --write` mirrors the latest verified gate fragment per tag, and drops the
    section when the tag's gate run was moved to raw/_unverified/."""
    import shutil

    from agentbench.summarize import summarize as summ

    fw = tmp_path / "omlx"
    shutil.copytree(run_dir / "omlx", fw)
    cap = fw / "capabilities.md"
    summ(fw, "t", write=True)
    assert "<!-- agentbench:begin gate t -->" in cap.read_text()
    unv = fw / "raw" / "_unverified"
    unv.mkdir(exist_ok=True)
    for f in (fw / "raw").glob("gate-t-*"):
        f.rename(unv / f.name)
    summ(fw, "t", files=sorted((fw / "raw").glob("m1-*.jsonl")), write=True)  # partial: leaves capabilities alone
    assert "<!-- agentbench:begin gate t -->" in cap.read_text()
    summ(fw, "t", write=True)
    assert "<!-- agentbench:begin gate t -->" not in cap.read_text()
