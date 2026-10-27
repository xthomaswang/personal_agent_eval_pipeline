"""Ask a served model the ui_vision items and score its answers (README.md).

    python3 run.py --models <served id>[,<id>] --url http://127.0.0.1:8510/v1/chat/completions \
        --out ../../runs/ui_vision/<run-id> [--conditions vision,vision+ocr,ax] [--samples 3] [--scale 1]
        [--coords pixel|norm1000] [--pages a,b] [--kinds click,fill,read]
    python3 run.py --selftest

Renders every page (render.swift) and OCRs it (ocr.swift) into <out>/render/, then asks each item
under each condition, `samples` times per model at the model's agent sampling (temperature 1.0,
top_p 0.95, top_k 20) with thinking off; writes <out>/records.jsonl and <out>/summary.json. Needs a
running OpenAI-compatible server (oMLX) and, for rendering, macOS with Swift. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
MAX_TOKENS = {"click": 64, "fill": 1500, "read": 128}


# ---- rendering ---------------------------------------------------------------------------------------

def render_all(out: Path, pages: list[str], scale: int) -> dict[str, dict]:
    rdir = out / "render"
    rdir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        prefix = rdir / f"{page}@{scale}x"
        subprocess.run(["swift", str(HERE / "render.swift"), str(PAGES / f"{page}.html"), str(prefix), str(scale)],
                       check=True, cwd=HERE)
    assets = {}
    for page in pages:
        name = f"{page}@{scale}x"
        tasks = json.loads((PAGES / f"{page}.tasks.json").read_text(encoding="utf-8"))
        langs = ["zh-Hans,en-US"] if tasks.get("language") == "zh" else []   # Vision reads English by default
        ocr = subprocess.run(["swift", str(HERE / "ocr.swift"), str(rdir / f"{name}.png"), "0", *langs],
                             check=True, cwd=HERE, capture_output=True, text=True).stdout
        assets[page] = {
            "png": base64.b64encode((rdir / f"{name}.png").read_bytes()).decode(),
            "elements": json.loads((rdir / f"{name}.elements.json").read_text(encoding="utf-8")),
            "ocr": json.loads(ocr.splitlines()[0])["lines"],
            "tasks": tasks,
        }
    return assets


# ---- prompts -----------------------------------------------------------------------------------------

def fmt_box(box: list[int], el: dict, coords: str) -> list[int]:
    """A box in the coordinates the answer is asked in (pixels, or 0-1000 of the screenshot)."""
    if coords == "norm1000":
        return [round(box[0] * 1000 / el["width"]), round(box[1] * 1000 / el["height"]),
                round(box[2] * 1000 / el["width"]), round(box[3] * 1000 / el["height"])]
    return box


def unit(coords: str) -> str:
    return "in coordinates normalized to 0-1000" if coords == "norm1000" else "in pixels"


def ocr_text(lines: list[dict], el: dict, coords: str) -> str:
    rows = [f'"{ln["candidates"][0]["text"]}" box={fmt_box(ln["box"], el, coords)}' for ln in lines if ln.get("candidates")]
    return f"Text found on the screen by OCR, each line with its box [x0, y0, x1, y1] {unit(coords)}:\n" + "\n".join(rows)


def ax_text(el: dict, coords: str) -> str:
    rows = []
    for i, a in enumerate(el["ax"]):
        extra = "".join([f' placeholder="{a["placeholder"]}"' if a.get("placeholder") else "",
                         f' value="{a["value"]}"' if a.get("value") else "",
                         f' checked={str(a["checked"]).lower()}' if "checked" in a else ""])
        rows.append(f'[{i}] {a["role"]} "{a["name"]}"{extra} box={fmt_box(a["box"], el, coords)}')
    return ("The elements on the screen (accessibility tree), each with an index, role, name, value and box "
            f"[x0, y0, x1, y1] {unit(coords)}:\n" + "\n".join(rows))


def where(cond: str, coords: str, w: int, h: int) -> str:
    if cond == "ax":
        return '"element": <index of the element>'
    if coords == "norm1000":
        return ('"x": <int>, "y": <int>, coordinates normalized to 0-1000 (0,0 is the top left corner of the '
                'screenshot, 1000,1000 the bottom right)')
    return f'"x": <int>, "y": <int>, pixel coordinates in this {w}x{h} screenshot (0,0 is the top left corner)'


def build(item: dict, cond: str, coords: str, asset: dict) -> tuple[str, bool]:
    """The user text of one request, and whether the screenshot goes with it."""
    el = asset["elements"]
    w, h = el["width"], el["height"]
    zh = asset["tasks"].get("language") == "zh"
    context = {"vision": "", "vision+ocr": ocr_text(asset["ocr"], el, coords) + "\n\n",
               "ax": ax_text(el, coords) + "\n\n"}[cond]
    screen = (f"This is a screenshot of a computer screen ({w}x{h} pixels).\n\n" if cond != "ax"
              else "You are looking at a computer screen through its accessibility tree.\n\n")
    if item["type"] == "click":
        ask = f'{item["instruction"]}\n\nReply with JSON only: {{{where(cond, coords, w, h)}}}, where you would click.'
    elif item["type"] == "fill":
        loc = where(cond, coords, w, h)
        ask = (f'{item["instruction"]}\n\nProfile:\n{json.dumps(item["profile"], ensure_ascii=False, indent=1)}\n\n'
               "Reply with JSON only: the list of actions, in order. Each action is one of\n"
               f'  {{"action": "type", {loc}, "text": "<text>"}} to click a text field and type into it,\n'
               f'  {{"action": "click", {loc}}} to click a checkbox, radio button, switch, tile or button,\n'
               f'  {{"action": "select", {loc}, "option": "<option text>"}} to choose an option of a dropdown.\n'
               "Only act on what the profile and the instruction call for.")
    else:
        ask = f'{item["question"]}\n\n' + ("只回答答案本身。" if zh else "Reply with the answer only.")
    return screen + context + ask, cond != "ax"


def ask(url: str, model: str, text: str, png_b64: str | None, max_tokens: int, timeout: float) -> dict:
    content: Any = text if png_b64 is None else [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
    ]
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, **SAMPLING}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.loads(resp.read())
    return {"reply": (r["choices"][0]["message"].get("content") or "").strip(), "seconds": round(time.time() - t0, 2),
            "finish_reason": r["choices"][0].get("finish_reason"), "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens")}


# ---- scoring -----------------------------------------------------------------------------------------

def json_values(s: str) -> tuple[list, bool]:
    """The JSON values in s from left to right, skipping the text between them, and whether the first
    bracket opens one (False when the reply starts a value it does not close properly)."""
    dec, out, i, first_ok = json.JSONDecoder(), [], 0, None
    while i < len(s):
        if s[i] in "[{":
            try:
                v, i = dec.raw_decode(s, i)
                out.append(v)
                first_ok = True if first_ok is None else first_ok
                continue
            except ValueError:
                first_ok = False if first_ok is None else first_ok
        i += 1
    return out, bool(first_ok)


def repair(s: str) -> str:
    """Mend the quote slips seen in sampled replies: a key missing its opening quote (`, y": 5`) and a
    number followed by a stray quote (`"y": 5"`)."""
    s = re.sub(r'([{,]\s*)([A-Za-z_]\w*)"(\s*:)', r'\1"\2"\3', s)
    return re.sub(r'("[A-Za-z_]\w*"\s*:\s*-?\d+(?:\.\d+)?)"(?=\s*[,}\]])', r"\1", s)


def parse_json(reply: str) -> tuple[Any, bool]:
    """The JSON answer in a reply (code fences and text around it allowed), and whether it needed
    mending; (None, False) when there is none. Objects written one after another become a list. A
    reply that does not parse as is gets repair(); failing that, the values inside it that do parse
    count (mended)."""
    s = re.sub(r"```(?:json)?", "", reply)
    vals, clean = json_values(s)
    if not clean:
        fixed, fixed_clean = json_values(repair(s))
        if fixed_clean:
            vals = fixed
    if not vals:
        return None, False
    ans = vals if len(vals) > 1 and all(isinstance(v, dict) for v in vals) else vals[0]
    return ans, not clean or len(vals) > 1


def inside(pt: tuple[float, float], box: list[int]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def raw_point(ans: Any) -> tuple[float, float] | None:
    """The point of an answer in the requested form ({"x", "y"}) or in the grounding forms Qwen-VL
    models emit by habit: {"point_2d": [x, y]}, {"bbox_2d": [x0, y0, x1, y1]} (its centre),
    {"x": [x, y]}, {"points": [...]}, {"coordinates": {"x", "y"}}, or a list holding one of these (the
    first counts). Numbers written as strings count."""
    if isinstance(ans, list):
        return raw_point(ans[0]) if ans else None
    if not isinstance(ans, dict):
        return None
    try:
        if is_number(ans.get("x")) and is_number(ans.get("y")):
            return float(ans["x"]), float(ans["y"])
        for key in ("point_2d", "point", "x", "coordinates"):
            v = ans.get(key)
            if isinstance(v, list) and len(v) == 2:
                return float(v[0]), float(v[1])
            if isinstance(v, dict):
                return raw_point(v)
        v = ans.get("bbox_2d") or ans.get("bbox")
        if isinstance(v, list) and len(v) == 4:
            return (float(v[0]) + float(v[2])) / 2, (float(v[1]) + float(v[3])) / 2
        if isinstance(ans.get("points"), list):
            return raw_point(ans["points"])
    except (TypeError, ValueError):
        return None
    return None


def is_number(v: Any) -> bool:
    if isinstance(v, str):
        return re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", v) is not None
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def requested_form(ans: Any, cond: str) -> bool:
    """Whether a click answer (or one fill action) used exactly the requested form."""
    if cond == "ax":
        return isinstance(ans, dict) and type(ans.get("element")) is int
    return isinstance(ans, dict) and all(type(ans.get(k)) in (int, float) for k in ("x", "y"))


def point(ans: Any, cond: str, coords: str, el: dict) -> tuple[float, float] | None:
    """Where an answer clicks, in screenshot pixels (an ax answer clicks its element's centre)."""
    if cond == "ax":
        a = ans[0] if isinstance(ans, list) and ans else ans
        try:
            b = el["ax"][int(a["element"])]["box"]
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    xy = raw_point(ans)
    if xy is None:
        return None
    x, y = xy
    if coords == "norm1000":
        x, y = x * el["width"] / 1000, y * el["height"] / 1000
    return (x, y)


def regions(el: dict, qa_id: str) -> list[list[int]]:
    return [el["qa"][qa_id], *el.get("qa_labels", {}).get(qa_id, [])] if qa_id in el["qa"] else []


def role_of(el: dict, qa_id: str) -> str | None:
    return next((a["role"] for a in el["ax"] if a.get("qa") == qa_id), None)


def control_at(pt: tuple[float, float], el: dict) -> str | None:
    """The data-qa control a click lands on: the smallest box (or label) that holds the point."""
    hits = [(abs((b[2] - b[0]) * (b[3] - b[1])), q) for q in el["qa"] for b in regions(el, q) if inside(pt, b)]
    return min(hits)[1] if hits else None


def norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().strip(".。,，;；:：!！").casefold()


def first_number(s: Any) -> float | None:
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+", str(s).replace("$", ""))
    return float(m.group(0).replace(",", "")) if m else None


def same_value(got: Any, want: Any) -> bool:
    g, w = norm_text(got), norm_text(want)
    if g == w:
        return True
    dg, dw = re.sub(r"\D", "", g), re.sub(r"\D", "", w)
    phone_like = re.fullmatch(r"[\d\s()+\-.]+", w) is not None and len(dw) >= 7
    return phone_like and dg[-10:] == dw[-10:]


def score_click(item: dict, ans: Any, mended: bool, cond: str, coords: str, el: dict) -> dict:
    pt = point(ans, cond, coords, el)
    if pt is None:
        return {"ok": False, "parsed": False}
    return {"ok": any(inside(pt, b) for t in item["target"] for b in regions(el, t)), "parsed": True,
            "requested_form": requested_form(ans, cond) and not mended, "point": [round(pt[0]), round(pt[1])],
            "landed": control_at(pt, el)}


def score_fill(item: dict, ans: Any, mended: bool, cond: str, coords: str, el: dict) -> dict:
    wrapped = isinstance(ans, dict) and isinstance(ans.get("actions"), list)   # {"actions": [...]}
    if wrapped:
        ans = ans["actions"]
    if not isinstance(ans, list):
        return {"ok": False, "parsed": False, "fields": 0, "of": len(item["expect"])}
    form = not (mended or wrapped) and all(requested_form(a, cond) for a in ans)
    typed: dict[str, Any] = {}
    clicks: dict[str, int] = {}
    chosen: dict[str, Any] = {}
    stray = []
    for a in ans:
        pt = point(a, cond, coords, el) if isinstance(a, dict) else None   # action dicts carry their own point
        q = control_at(pt, el) if pt else None
        if q is None:
            stray.append(a)
            continue
        kind = a.get("action")
        if kind == "type":
            typed[q] = a.get("text", "")
        elif kind == "select":
            chosen[q] = a.get("option", "")
        else:
            clicks[q] = clicks.get(q, 0) + 1
    right = 0
    detail = {}
    for q, want in item["expect"].items():
        (kind, val), = want.items()
        if kind == "value":
            ok = q in typed and same_value(typed[q], val)
        elif kind == "option":
            ok = same_value(chosen.get(q, typed.get(q, "")), val)
        else:   # a radio stays selected however often it is clicked; a checkbox or switch flips each time
            n = clicks.get(q, 0)
            ok = n >= 1 if role_of(el, q) == "radio" else n % 2 == 1
        right += ok
        detail[q] = ok
    touched = set(typed) | set(clicks) | set(chosen)
    forbidden = sorted(set(item.get("forbid", [])) & touched)
    return {"ok": right == len(item["expect"]) and not forbidden, "parsed": True, "requested_form": form,
            "fields": right, "of": len(item["expect"]), "detail": detail, "forbidden_touched": forbidden,
            "stray": len(stray)}


def score_read(item: dict, reply: str) -> dict:
    want, match = item["answer"], item["match"]
    if match == "number":
        g, w = first_number(reply), first_number(want)
        ok = g is not None and w is not None and abs(g - w) <= 0.005
    elif match == "set":
        split = lambda s: {norm_text(x) for x in re.split(r"[,，、;；\n]|\band\b|和", str(s)) if norm_text(x)}
        ok = split(reply) == split(want)
    else:
        ok = norm_text(reply) == norm_text(want) or norm_text(reply).endswith(norm_text(want)) and len(norm_text(reply)) <= len(norm_text(want)) + 12
    return {"ok": ok, "parsed": True}


def score(item: dict, reply: str, cond: str, coords: str, el: dict) -> dict:
    if item["type"] == "read":
        return score_read(item, reply)
    ans, mended = parse_json(reply)
    return (score_click if item["type"] == "click" else score_fill)(item, ans, mended, cond, coords, el)


# ---- run ---------------------------------------------------------------------------------------------

def summarize(records: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for r in records:
        key = f'{r["model"]} | {r["condition"]} | {r["type"]}'
        s = out.setdefault(key, {"ok": 0, "n": 0, "unparsed": 0, "fields": 0, "fields_of": 0})
        s["n"] += 1
        s["ok"] += r["score"]["ok"]
        s["unparsed"] += not r["score"].get("parsed", True)
        s["fields"] += r["score"].get("fields", 0)
        s["fields_of"] += r["score"].get("of", 0)
        if "requested_form" in r["score"]:
            s["form"] = s.get("form", 0) + r["score"]["requested_form"]
    for s in out.values():
        if not s["fields_of"]:
            del s["fields"], s["fields_of"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models")
    ap.add_argument("--url", default="http://127.0.0.1:8510/v1/chat/completions")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--conditions", default="vision,vision+ocr,ax")
    ap.add_argument("--kinds", default="click,fill,read")
    ap.add_argument("--pages", default="")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--scale", type=int, default=1, choices=(1, 2))
    ap.add_argument("--coords", default="pixel", choices=("pixel", "norm1000"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rescore", type=Path, help="re-score <out>/records.jsonl with this scorer (no requests)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.rescore:
        return rescore(args.rescore)
    if not args.models or not args.out:
        ap.error("--models and --out are required")
    pages = args.pages.split(",") if args.pages else sorted(p.name.removesuffix(".tasks.json")
                                                           for p in PAGES.glob("*.tasks.json"))
    assets = render_all(args.out, pages, args.scale)
    conds, kinds, models = args.conditions.split(","), args.kinds.split(","), args.models.split(",")
    records = []
    with (args.out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for page in pages:
            a = assets[page]
            for item in (i for i in a["tasks"]["items"] if i["type"] in kinds):
                for cond in conds:
                    text, with_image = build(item, cond, args.coords, a)
                    for s in range(args.samples):
                        for model in (models if s % 2 == 0 else models[::-1]):
                            r = ask(args.url, model, text, a["png"] if with_image else None,
                                    MAX_TOKENS[item["type"]], args.timeout)
                            sc = score(item, r["reply"], cond, args.coords, a["elements"])
                            rec = {"page": page, "item": item["id"], "type": item["type"], "condition": cond,
                                   "coords": args.coords, "scale": args.scale, "sample": s, "model": model,
                                   **r, "score": sc}
                            records.append(rec)
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fh.flush()
                            print(f'{page:16} {item["id"]:10} {cond:10} {model[:24]:24} {"ok " if sc["ok"] else "BAD"} '
                                  f'{r["reply"][:60]!r}', flush=True)
    summary = {"pages": pages, "scale": args.scale, "coords": args.coords, "samples": args.samples,
               "sampling": SAMPLING, "thinking": "off", "results": summarize(records)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in summary["results"].items():
        print(f'{k:60} {v["ok"]}/{v["n"]}' + (f' fields {v["fields"]}/{v["fields_of"]}' if "fields" in v else "")
              + (f' unparsed {v["unparsed"]}' if v["unparsed"] else ""))
    return 0


def rescore(out: Path) -> int:
    records = [json.loads(line) for line in (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    cache: dict[str, tuple[dict, dict]] = {}
    for r in records:
        if r["page"] not in cache:
            el = json.loads((out / "render" / f'{r["page"]}@{r["scale"]}x.elements.json').read_text(encoding="utf-8"))
            items = {i["id"]: i for i in json.loads((PAGES / f'{r["page"]}.tasks.json').read_text(encoding="utf-8"))["items"]}
            cache[r["page"]] = (el, items)
        el, items = cache[r["page"]]
        r["score"] = score(items[r["item"]], r["reply"], r["condition"], r["coords"], el)
    (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary["results"] = summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in summary["results"].items():
        print(f'{k:60} {v["ok"]}/{v["n"]}' + (f' fields {v["fields"]}/{v["fields_of"]}' if "fields" in v else "")
              + (f' unparsed {v["unparsed"]}' if v["unparsed"] else "") + (f' requested-form {v["form"]}/{v["n"]}' if "form" in v else ""))
    return 0


def selftest() -> int:
    el = {"width": 1280, "height": 1024,
          "qa": {"email": [100, 100, 400, 130], "no": [100, 200, 113, 213], "yes": [100, 230, 113, 243],
                 "submit": [100, 300, 200, 330], "phone": [100, 150, 400, 180], "degree": [500, 100, 700, 130]},
          "qa_labels": {"no": [[118, 198, 160, 215]]},
          "ax": [{"role": "textbox", "name": "Email", "box": [100, 100, 400, 130], "qa": "email"},
                 {"role": "button", "name": "Submit", "box": [100, 300, 200, 330], "qa": "submit"},
                 {"role": "radio", "name": "No", "box": [100, 200, 113, 213], "qa": "no"}]}
    click = {"type": "click", "target": ["no"]}
    assert score(click, '{"x": 140, "y": 206}', "vision", "pixel", el)["ok"]          # on the label
    assert not score(click, '{"x": 105, "y": 236}', "vision", "pixel", el)["ok"]      # the other radio
    assert score(click, '```json\n{"x": 83, "y": 201}\n```', "vision", "norm1000", el)["ok"]
    assert score({"type": "click", "target": ["submit"]}, '{"element": 1}', "ax", "pixel", el)["ok"]
    assert not score(click, "no idea", "vision", "pixel", el)["parsed"]
    r = score(click, '```json\n[{"point_2d": [109, 201], "label": "No"}]\n```', "vision", "norm1000", el)
    assert r["ok"] and not r["requested_form"], r
    assert score(click, '{"bbox_2d": [92, 195, 126, 208]}', "vision", "norm1000", el)["ok"]
    assert score(click, '{"x": [109, 201]}', "vision", "norm1000", el)["ok"]
    for slip in ('{"x": 109, y": 201}', '{"x": 109, "y": 201"}', '{"x": 109, "y": "201"}',
                 '{"coordinates": {"x": 109, "y": 201}}'):
        r = score(click, slip, "vision", "norm1000", el)
        assert r["ok"] and not r["requested_form"], (slip, r)
    assert score(click, '{"x": 109, "y": 201}', "vision", "norm1000", el)["requested_form"]
    assert not score(click, 'I would click [No] at {"x": 109, "y": 201}', "vision", "norm1000", el)["requested_form"]
    fill = {"type": "fill", "expect": {"email": {"value": "a@example.com"}, "no": {"checked": True},
                                       "phone": {"value": "(415) 555-0172"}, "degree": {"option": "Master's"}},
            "forbid": ["submit"]}
    good = ('[{"action": "type", "x": 200, "y": 115, "text": "A@example.com "},'
            ' {"action": "type", "x": 200, "y": 165, "text": "415-555-0172"},'
            ' {"action": "select", "x": 600, "y": 115, "option": "master\'s"},'
            ' {"action": "click", "x": 130, "y": 205}]')
    r = score(fill, good, "vision", "pixel", el)
    assert r["ok"] and r["fields"] == 4 and r["requested_form"], r
    for variant in ('{"actions": ' + good + "}", good[1:-1].replace("},", "}\n"),
                    good.replace('"y": 165', 'y": 165')):
        r = score(fill, variant, "vision", "pixel", el)
        assert r["ok"] and not r["requested_form"], (variant, r)
    r = score({"type": "fill", "expect": {"email": {"value": "22:00"}}},
              '[{"action": "type", "x": 200, "y": 115, "text": "22:00"}, {"action": "click", "x": 9, y": 9}]',
              "vision", "pixel", el)
    assert r["ok"] and r["stray"] == 1, r
    r = score(fill, good[:-1] + ', {"action": "click", "x": 150, "y": 315}]', "vision", "pixel", el)
    assert not r["ok"] and r["forbidden_touched"] == ["submit"], r
    assert score({"type": "read", "answer": "$1,284.50", "match": "number"}, "The total is 1284.5", "vision", "pixel", el)["ok"]
    assert score({"type": "read", "answer": "Ana, Ben", "match": "set"}, "Ben and Ana", "vision", "pixel", el)["ok"]
    assert score({"type": "read", "answer": "Austin, TX", "match": "text"}, "Austin, TX.", "vision", "pixel", el)["ok"]
    assert not score({"type": "read", "answer": "Weekly", "match": "text"}, "Daily", "vision", "pixel", el)["ok"]
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
