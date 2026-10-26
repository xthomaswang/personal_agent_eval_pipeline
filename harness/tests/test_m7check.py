import json

from agentbench.fixtures.agent_tools import TOOL_INDEX
from agentbench.m7check import check_case, match_value, point_check

SCHEMAS = {n: t["function"]["parameters"] for n, t in TOOL_INDEX.items()}


def calls(*pairs):
    return [{"name": n, "arguments": json.dumps(a), "arguments_parsed": a} for n, a in pairs]


READ = {"id": "x", "expect": {"calls": [{"name": "fs_read_file", "args": {"path": {"eq": "README.md", "norm": "path"}}}]}}


def test_single_call_pass_and_path_normalization():
    for p in ("README.md", "./README.md", "/workspace/demo-app/README.md"):
        assert check_case(READ, calls(("fs_read_file", {"path": p})), "", [], SCHEMAS)["pass"]


def test_wrong_tool_schema_error_extra_call_and_leak():
    r = check_case(READ, calls(("fs_grep", {"pattern": "x"})), "", [], SCHEMAS)
    assert not r["pass"] and "instead of 'fs_read_file'" in r["reasons"][0]
    r = check_case(READ, calls(("fs_read_file", {"path": "README.md", "bogus": 1})), "", [], SCHEMAS)
    assert not r["pass"] and any("schema" in x for x in r["reasons"])
    r = check_case(READ, calls(("fs_read_file", {"path": "README.md"}), ("fs_read_file", {"path": "a"})), "", [], SCHEMAS)
    assert not r["pass"] and any("spurious extra" in x for x in r["reasons"])
    r = check_case(READ, calls(("fs_read_file", {"path": "README.md"})), "<tool_call>", ["<tool_call>"], SCHEMAS)
    assert not r["pass"] and "leaked" in r["reasons"][0]
    bad = [{"name": "fs_read_file", "arguments": "{oops", "arguments_parsed": None, "arguments_error": "Expecting"}]
    assert "not a JSON object" in check_case(READ, bad, "", [], SCHEMAS)["reasons"][0]
    assert check_case(READ, [], "I cannot", [], SCHEMAS)["reasons"] == ["no tool call"]


def test_parallel_calls_any_order_and_no_call():
    case = {"expect": {"calls": [{"name": "fs_read_file", "args": {"path": {"eq": "A.md"}}},
                                 {"name": "fs_read_file", "args": {"path": {"eq": "B.md"}}}]}}
    assert check_case(case, calls(("fs_read_file", {"path": "B.md"}), ("fs_read_file", {"path": "A.md"})), "", [], SCHEMAS)["pass"]
    assert not check_case(case, calls(("fs_read_file", {"path": "A.md"})), "", [], SCHEMAS)["pass"]
    none = {"expect": {"no_call": True}}
    assert check_case(none, [], "It means not found.", [], SCHEMAS)["pass"]
    assert not check_case(none, calls(("web_search", {"query": "404"})), "", [], SCHEMAS)["pass"]


def test_matchers():
    assert not match_value(["CMD", "Shift", "t"], {"set_eq": ["cmd", "shift", "t"], "norm": "key"})
    assert not match_value(["command", "shift", "t"], {"set_eq": ["cmd", "shift", "t"], "norm": "key"})
    assert match_value(1, {"eq": True})  # booleans are not integers
    assert not match_value(0.35, {"range": [0.345, 0.355]}) and match_value(35, {"range": [0.345, 0.355]})
    assert not match_value({"a": 1, "b": 2}, {"subset": {"a": 1}})
    edits = [{"old_text": "PORT = 8000", "new_text": "PORT = 8080"}, {"old_text": "DEBUG = True", "new_text": "DEBUG = False"}]
    spec = {"match_each": [{"old_text": {"contains": "DEBUG"}, "new_text": {"contains": "False"}},
                           {"old_text": {"contains": "PORT"}, "new_text": {"contains": "8080"}}]}
    assert not match_value(edits, spec)
    assert match_value(edits[:1], spec)


def test_point_in_box_pixels_and_normalized():
    spec = {"x": "x", "y": "y", "box": [800, 560, 900, 600]}
    assert point_check({"x": 850, "y": 580}, spec) == ([], "pixels")
    assert point_check({"x": 664, "y": 566}, spec) == ([], "normalized_1000")  # 850/1280*1000, 580/1024*1000
    errs, conv = point_check({"x": 10, "y": 10}, spec)
    assert errs and conv is None
