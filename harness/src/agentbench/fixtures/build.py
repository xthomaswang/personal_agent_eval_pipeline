"""`agentbench fixtures build`: deterministic, seeded fixture generation.

Committed (small text, in <repo>/fixtures/): system_prompt.md, tools.openai.json,
tools.anthropic.json, m1_m5.json, m4_trajectory.json, m7_suite.json, gate.json and
manifest.json (sha256 of every generated file plus measured token sizes).
Generated (gitignored, <repo>/fixtures/generated/): screenshots, their text
equivalents, cold prompts, M3 prompts and helper prompts.

All sizes are measured with the local Qwen3.8 tokenizer and chat template. Nonce and
request-id placeholders are measured with zeros; real values are digits only, and every
digit is one token, so substitution never changes a prompt's token count.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any, Callable

import PIL

from ..tokenizer import QwenPrompt, image_tokens
from . import agent_steps as S
from . import agent_text as T
from .agent_tools import TOOLS, TOOL_INDEX, anthropic_tools
from .m7_data import CASES, QUICK_CASES
from .screens import KINDS, Screen, a11y_lines, render_files_highlight, render_save_dialog, render_screen, render_verify_code
from .textgen import TextGen

GENERATOR_VERSION = 1
DEFAULT_SEED = 20261017
REQUEST_ID = "{{REQUEST_ID}}"
NONCE = "{{NONCE}}"
ZERO12 = "000000000000"
IMAGE_TOKENS = 1282  # 1280 <|image_pad|> + <|vision_start|> + <|vision_end|> for 1280x1024
PREFIX_TARGET = (14_900, 15_100)
TOOL_RESULT_TOKENS = 250
COLD_SIZES = [2_000, 16_000, 32_000, 64_000]
M3_OFF_SIZES = [2_000, 16_000, 32_000, 64_000]
M3_ON_SIZES = [2_000, 16_000, 32_000]
HELPER_TOKENS = 2_000
N_M1_IMAGES = 12
N_M5_IMAGES = 12
COMMITTED = ["system_prompt.md", "tools.openai.json", "tools.anthropic.json", "m1_m5.json", "m4_trajectory.json",
             "m7_suite.json", "gate.json"]


def nonce_line(digits: str = ZERO12) -> str:
    """Nonce placed at the very start of the first message (requests without tools)."""
    return f"[run {digits}]\n"


def nonce_tool_prefix(digits: str = ZERO12) -> str:
    """Nonce placed at the start of the first tool's description (requests with tools)."""
    return f"[run {digits}] "


def _digits(r: random.Random, n: int) -> str:
    return "".join(str(r.randrange(10)) for _ in range(n))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ fitting
def fit_lines(lines: list[str], target: int, count: Callable[[str], int], r: random.Random,
              pad: Callable[[str], str]) -> str:
    """Join lines (each ending in a newline) and pad with digits to exactly `target` tokens.

    `pad(digits)` renders the final padding line. Lines are dropped from the end until
    the padded text fits; each extra digit adds exactly one token."""
    lines = list(lines)
    while True:
        base = "".join(ln + "\n" for ln in lines)
        digits = _digits(r, 1)
        n = count(base + pad(digits) + "\n")
        if n <= target or not lines:
            break
        lines.pop()
    while n < target:
        digits += str(r.randrange(10))
        n = count(base + pad(digits) + "\n")
    if n != target:
        raise RuntimeError(f"could not fit text to {target} tokens (got {n})")
    return base + pad(digits) + "\n"


def fit_document(doc: str, target: int, render_count: Callable[[str], int], tok, r: random.Random) -> str:
    """Cut `doc` (at a whitespace boundary) and append a digits reference so that
    render_count(result) == target exactly."""
    ids = tok.encode(doc, add_special_tokens=False).ids

    def cut(n: int) -> str:
        text = tok.decode(ids[:n])
        sp = max(text.rfind(" "), text.rfind("\n"))
        return text[:sp].rstrip() if sp > 0 else text

    def with_ref(text: str, digits: str) -> str:
        return f"{text}\n\n(End of excerpt. Reference {digits}.)"

    lo, hi = 0, len(ids)
    while lo < hi:  # largest n whose padded render fits
        mid = (lo + hi + 1) // 2
        if render_count(with_ref(cut(mid), "1")) <= target:
            lo = mid
        else:
            hi = mid - 1
    text = cut(lo)
    digits = "1"
    while render_count(with_ref(text, digits)) < target:
        digits += str(r.randrange(10))
    out = with_ref(text, digits)
    if render_count(out) != target:
        raise RuntimeError(f"document fit failed for target {target}")
    return out


# ------------------------------------------------------------- tool results
def _computer_result(g: TextGen, r: random.Random) -> list[str]:
    t = (r.randint(8, 18), r.randint(0, 59), r.randint(0, 59))
    lines = [f"request_id: {REQUEST_ID}", "status: ok", "action: left click at (287, 361), 1 click",
             "target: [checkbox] \"Format on save\" in window w-3f2a \"Settings\"",
             f"state change: unchecked -> checked ({r.randint(40, 180)} ms after the click)",
             "focused window: w-3f2a \"Settings\" (Code Editor), at (212, 96), size 856x716", "ui events:"]
    for i in range(r.randint(4, 6)):
        ms = r.randint(0, 999)
        ev = g.pick(["focus moved to [checkbox] \"Format on save\"", "setting editor.formatOnSave written to user scope",
                     "window \"Settings\" repainted", f"tooltip \"{g.sentence()[:40]}\" shown", "status bar text changed",
                     "settings file saved (412 bytes)", "extension host acknowledged the change"])
        lines.append(f"- {t[0]:02d}:{t[1]:02d}:{(t[2] + i) % 60:02d}.{ms:03d} {ev}")
    lines.append("open windows:")
    for _ in range(r.randint(4, 6)):
        lines.append(f"- w-{r.randrange(16**4):04x} \"{g.project()} - {g.ident()}.{g.pick(['py', 'md', 'yaml'])}\" "
                     f"({g.pick(['Code Editor', 'Terminal', 'Browser', 'Files'])}) at ({r.randint(0, 400)}, {r.randint(26, 200)}) "
                     f"size {r.randint(600, 1280)}x{r.randint(400, 980)}")
    lines.append("screen: 1280x1024; a new screenshot follows in the next message")
    for _ in range(6):
        lines.append(f"note: {g.sentence()}")
    return lines


def _filler(kind: str, g: TextGen, r: random.Random) -> list[str]:
    out: list[str] = []
    if kind == "windows":
        for _ in range(16):
            out.append(f"- w-{r.randrange(16**4):04x} \"{g.project()} - {g.ident()}\" ({g.pick(['Code Editor', 'Terminal', 'Browser', 'Files', 'Notes'])}) "
                       f"at ({r.randint(0, 400)}, {r.randint(26, 200)}) size {r.randint(600, 1280)}x{r.randint(400, 980)}")
    elif kind == "pytest":
        for _ in range(18):
            out.append(f"tests/test_{g.ident()}.py {'.' * r.randint(3, 14)} [{r.randint(1, 99):3d}%]")
    elif kind == "code":
        n = r.randint(1, 40)
        for _ in range(4):
            for ln in g.code().splitlines()[1:-1]:
                out.append(f"{n:4d}  {ln}")
                n += 1
    elif kind == "diff":
        out += [f"@@ -{r.randint(10, 90)},7 +{r.randint(10, 90)},7 @@"]
        for _ in range(3):
            for ln in g.code().splitlines()[1:-1]:
                out.append(f" {ln}")
    elif kind == "dirlist":
        for _ in range(16):
            out.append(f"{g.pick(['-rw-r--r--', 'drwxr-xr-x'])}  {r.randint(80, 90000):>6}  2026-{r.randint(1, 10):02d}-{r.randint(1, 28):02d}  "
                       f"{g.ident('kebab')}{g.pick(['.yaml', '.ini', '.json', '/', '.md'])}")
    elif kind == "page":
        for _ in range(8):
            out.append(g.sentence())
    elif kind == "notes":
        for _ in range(12):
            out.append(f"- n-{r.randrange(16**4):04x} \"{g.project()} {g.pick(['notes', 'checklist', 'summary'])}\" "
                       f"(updated 2026-{r.randint(1, 10):02d}-{r.randint(1, 28):02d})")
    else:  # log
        for _ in range(4):
            out += g.log().splitlines()[1:-1]
    return out


M4_FILLER = ["windows", "pytest", "code", "code", "diff", "pytest", "dirlist", "log", "diff", "log", "log", "log",
             "dirlist", "log", "page", "page", "log", "windows", "notes", "notes"]


def _tool_text(head: str, filler: list[str], count: Callable[[str], int], r: random.Random, target: int) -> str:
    """A tool result of exactly `target` tokens (placeholders measured as 12 zeros)."""
    lines = (head.rstrip("\n").split("\n") if head else []) + filler

    def measured(text: str) -> int:
        return count(text.rstrip("\n").replace(REQUEST_ID, ZERO12))

    text = fit_lines(lines, target, measured, r, lambda d: f"trace: {d}")
    return text.rstrip("\n")


# ------------------------------------------------------------------ prompts
_COLD_SYS = "You are a technical assistant. Answer questions about the provided engineering notes accurately and concisely."
_COLD_USER = ("{document}\n\nQuestion: summarize the three most important design decisions described in the notes "
              "above, in three sentences.")
_M3_SYS = "You are a data generation assistant. You follow output format instructions exactly."
_M3_USER = ("{document}\n\nTask: using the components and metrics mentioned in the notes above, write a JSON array "
            "of exactly 150 remediation actions. Each element must be an object with the keys \"step\" (integer, "
            "starting at 1), \"component\" (string), \"action\" (one of \"inspect\", \"restart\", \"scale\", "
            "\"patch\", \"rollback\"), \"target_host\" (string), \"priority\" (integer from 1 to 5) and "
            "\"rationale\" (one sentence of 15 to 25 words). Output only the JSON array, with no text before or "
            "after it, and do not stop before all 150 elements are written.")
_HELPER_SYS = "You are a helper model that condenses tool output for a coding agent. Be brief and factual."
_HELPER_USER = "Summarize the following build and test output in five short bullet points:\n\n{document}"
PROMPT_KINDS = {  # kind: (system, user template, enable_thinking, salt base)
    "cold": (_COLD_SYS, _COLD_USER, False, 7_000_000),
    "m3-off": (_M3_SYS, _M3_USER, False, 7_100_000),
    "m3-on": (_M3_SYS, _M3_USER, True, 7_200_000),
    "helper1": (_HELPER_SYS, _HELPER_USER, False, 7_300_001),
    "helper2": (_HELPER_SYS, _HELPER_USER, False, 7_300_002),
}


def make_prompt(p: QwenPrompt, kind: str, size: int, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """A prompt of exactly `size` rendered tokens (nonce measured as zeros). Deterministic
    in (seed, kind, size), so on-the-fly sizes match what a build would write."""
    system, user, thinking, salt = PROMPT_KINDS[kind]
    salt += 0 if kind.startswith("helper") else size
    r = random.Random(seed * 1000 + salt)
    doc = TextGen(seed + salt).document(int(size * 4.2) + 4000)

    def render_count(text: str) -> int:
        msgs = [{"role": "system", "content": nonce_line() + system},
                {"role": "user", "content": user.replace("{document}", text)}]
        return p.count(msgs, None, enable_thinking=thinking)

    body = fit_document(doc, size, render_count, p.tok, r)
    name = kind if kind.startswith("helper") else f"{kind}-{size}"
    msgs = [{"role": "system", "content": NONCE + system}, {"role": "user", "content": user.replace("{document}", body)}]
    return {"id": name, "target_tokens": size, "tokens": render_count(body), "enable_thinking": thinking,
            "nonce_placeholder": NONCE, "messages": msgs}


# --------------------------------------------------------------- the build
class _Builder:
    def __init__(self, root: Path, seed: int, log: Callable[[str], None]) -> None:
        self.root, self.gen, self.seed, self.log = root, root / "generated", seed, log
        self.p = QwenPrompt()
        self.count = self.p.count_text
        self.r = random.Random(seed)
        self.files: list[Path] = []
        self.sizes: dict[str, Any] = {}
        self.images: dict[str, dict[str, Any]] = {}

    def rng(self, salt: int) -> random.Random:
        return random.Random(self.seed * 1000 + salt)

    def write(self, rel: str, obj: Any) -> Path:
        path = self.root / rel
        if isinstance(obj, str):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(obj, encoding="utf-8")
        else:
            _dump(path, obj)
        self.files.append(path)
        return path

    # ---------------------------------------------------------------- prefix
    def system_prompt(self) -> str:
        sections = T.SYSTEM_SECTIONS + [T.ENVIRONMENT]

        def prefix_tokens(sp: str) -> int:
            full = self.p.render([{"role": "system", "content": sp}, {"role": "user", "content": T.M1_TASK}], TOOLS)
            return self.count(full[: full.index("<|im_end|>") + len("<|im_end|>\n")])

        base = "\n\n".join(sections)
        notes: list[str] = []
        sp = base
        for note in T.PAD_NOTES:
            if prefix_tokens(sp) >= PREFIX_TARGET[0]:
                break
            notes.append(note)
            sp = base + "\n\n# Additional notes\n\n" + "\n".join(f"- {n}" for n in notes)
        n = prefix_tokens(sp)
        if not PREFIX_TARGET[0] - 200 <= n <= PREFIX_TARGET[1] + 400:
            raise RuntimeError(f"agent prefix is {n} tokens, outside the target window")
        self.sizes["prefix_tokens"] = n
        self.sizes["system_prompt_tokens"] = self.count(sp)
        self.sizes["n_tools"] = len(TOOLS)
        return sp

    # ---------------------------------------------------------------- images
    def image(self, image_id: str, screen: Screen, salt: int) -> None:
        path = self.gen / "screens" / f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        screen.image.save(path, format="PNG")
        self.files.append(path)
        r = self.rng(salt)
        g = TextGen(self.seed + salt)
        lines = a11y_lines(screen)
        extra = []
        for _ in range(120):
            x0, y0 = r.randint(60, 600), r.randint(80, 900)
            extra.append(f"    [text] \"{g.sentence()[:90]}\" ({x0}, {y0}, {x0 + r.randint(200, 640)}, {y0 + 18})")
        text = fit_lines(lines + extra, IMAGE_TOKENS, self.count, r, lambda d: f"[status] \"ref {d}\"")
        self.write(f"generated/a11y/{image_id}.txt", text)
        w, h = screen.image.size
        self.images[image_id] = {"png": f"generated/screens/{image_id}.png", "text": f"generated/a11y/{image_id}.txt",
                                 "width": w, "height": h, **{k: v for k, v in screen.meta.items()}}

    def all_images(self) -> None:
        for i in range(N_M1_IMAGES):
            self.image(f"m1-{i:02d}", render_screen(self.seed + 100 + i, "settings", "Settings"), 100 + i)
        for k, step in enumerate(S.M4_STEPS, start=1):
            self.image(f"m4-{k:02d}", render_screen(self.seed + 200 + k, step[4], step[5]), 200 + k)
        r = self.rng(300)
        for i in range(N_M5_IMAGES):
            self.image(f"m5-{i:02d}", render_screen(self.seed + 300 + i, KINDS[r.randrange(len(KINDS))]), 300 + i)
        self.image("g2-verify", render_verify_code(self.seed + 400), 400)
        self.image("m7-save-dialog", render_save_dialog(self.seed + 401), 401)
        hl = TextGen(self.seed + 402)
        name = f"{hl.ident('kebab')}-{hl.r.randint(10, 99)}.csv"
        self.image("m7-files-highlight", render_files_highlight(self.seed + 402, name), 402)
        n = image_tokens(1280, 1024, self.p.pre) + 2
        if n != IMAGE_TOKENS:
            raise RuntimeError(f"unexpected image token count {n}")
        self.sizes["image_tokens"] = n
        self.sizes["n_images"] = len(self.images)

    # ------------------------------------------------------------------- M1
    def m1_m5(self, system_prompt: str) -> dict[str, Any]:
        r, g = self.rng(500), TextGen(self.seed + 500)
        results = [_tool_text("", _computer_result(g, r), self.count, r, TOOL_RESULT_TOKENS)
                   for _ in range(N_M1_IMAGES + N_M5_IMAGES)]
        sys_msg = {"role": "system", "content": system_prompt}
        warm = [sys_msg, {"role": "user", "content": T.M1_TASK}]
        z = [x.replace(REQUEST_ID, ZERO12) for x in results]
        counts = [self.count(x) for x in z]
        trial = warm + [S.M1_ASSISTANT, {"role": "tool", "tool_call_id": "call_m1_click", "content": z[0]},
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}},
                                                     {"type": "text", "text": S.M1_CAPTION}]}]
        warm_n = self.p.count(warm, TOOLS)
        trial_n = self.p.count(trial, TOOLS)
        warm_text = self.p.render(warm, TOOLS)
        if not self.p.render(trial, TOOLS).startswith(warm_text):
            raise RuntimeError("M1 warm-up prompt is not a prefix of the trial prompt")
        self.sizes["m1"] = {"task_tokens": self.count(T.M1_TASK), "warm_prompt_tokens": warm_n,
                            "trial_prompt_tokens": trial_n, "suffix_tokens": trial_n - warm_n,
                            "tool_result_tokens": {"min": min(counts), "median": statistics.median(counts), "max": max(counts)}}
        return {"m1_task": T.M1_TASK, "m1_assistant": S.M1_ASSISTANT, "m1_caption": S.M1_CAPTION,
                "m1_images": [f"m1-{i:02d}" for i in range(N_M1_IMAGES)],
                "m5_images": [f"m5-{i:02d}" for i in range(N_M5_IMAGES)],
                "tool_results": results, "request_id_placeholder": REQUEST_ID}

    # ------------------------------------------------------------------- M4
    def m4(self, system_prompt: str) -> dict[str, Any]:
        r, g = self.rng(600), TextGen(self.seed + 600)
        steps = []
        for k, (content, name, args, head, _kind, _title) in enumerate(S.M4_STEPS, start=1):
            cid = f"call_m4_{k:02d}"
            msg = S.call(cid, name, args, content)
            text = _tool_text(head, _filler(M4_FILLER[k - 1], g, r), self.count, r, TOOL_RESULT_TOKENS)
            steps.append({"step": k, "assistant": msg,
                          "tool": {"role": "tool", "tool_call_id": cid, "content": text},
                          "image": f"m4-{k:02d}", "caption": S.M4_CAPTION.format(k=k)})
        traj = {"task": T.M4_TASK, "steps": steps}
        # sizes, and the exact-prefix property between consecutive steps
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": T.M4_TASK}]
        prev_text, prev_n = self.p.render(msgs, TOOLS), self.p.count(msgs, TOOLS)
        per_step, asst = [], []
        for st in steps:
            gen_prompt = self.p.render(msgs, TOOLS)
            msgs = msgs + [st["assistant"], st["tool"], {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:,"}}, {"type": "text", "text": st["caption"]}]}]
            text, n = self.p.render(msgs, TOOLS), self.p.count(msgs, TOOLS)
            if not text.startswith(prev_text):
                raise RuntimeError(f"M4 step {st['step']} is not an extension of the previous prompt")
            turn = text[len(gen_prompt):]
            turn = turn[: turn.index("<|im_end|>") + len("<|im_end|>")]
            asst.append(self.count(turn))
            per_step.append({"step": st["step"], "prompt_tokens": n, "new_tokens": n - prev_n})
            prev_text, prev_n = text, n
        self.sizes["m4"] = {"start_prompt_tokens": self.p.count(msgs[:2], TOOLS), "steps": per_step,
                            "assistant_tokens": {"min": min(asst), "median": statistics.median(asst), "max": max(asst),
                                                 "per_step": asst},
                            "final_prompt_tokens": per_step[-1]["prompt_tokens"]}
        return traj

    # ------------------------------------------------------------------- M7
    def m7(self, system_prompt: str) -> dict[str, Any]:
        from ..m7check import check_case

        hl = self.images["m7-files-highlight"]["highlight"]
        box = self.images["m7-save-dialog"]["save_box"]
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

        def subst(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: subst(v) for k, v in o.items()}
            if isinstance(o, list):
                return [subst(v) for v in o]
            if o == "{save_box}":
                return box
            if o == "{save_cx}":
                return cx
            if o == "{save_cy}":
                return cy
            if isinstance(o, str):
                return o.replace("{highlight}", hl)
            return o

        cases = subst(CASES)
        schemas = {n: t["function"]["parameters"] for n, t in TOOL_INDEX.items()}
        sizes = {}
        for c in cases:
            calls = [{"name": x["name"], "arguments": json.dumps(x["arguments"], ensure_ascii=False),
                      "arguments_parsed": x["arguments"]} for x in c["reference"]]
            res = check_case(c, calls, "A 404 means the server could not find the requested resource." if not calls else "",
                             [], schemas)
            if not res["pass"]:
                raise RuntimeError(f"M7 reference answer fails its own checks: {c['id']}: {res['reasons']}")
            msgs = [{"role": "system", "content": system_prompt}] + [
                {**m, "content": [{"type": "image_url", "image_url": {"url": "data:,"}} if p.get("type") == "image_ref" else p
                                  for p in m["content"]]} if isinstance(m.get("content"), list) else m
                for m in c["messages"]]
            sizes[c["id"]] = self.p.count(msgs, TOOLS) - self.sizes["prefix_tokens"]
        self.sizes["m7_case_suffix_tokens"] = sizes
        return {"cases": cases, "quick": QUICK_CASES}

    # ------------------------------------------------------------- prompts
    def prompt_file(self, kind: str, size: int) -> int:
        obj = make_prompt(self.p, kind, size, self.seed)
        self.write(f"generated/prompts/{obj['id']}.json", obj)
        return obj["tokens"]

    def prompts(self) -> None:
        self.sizes["cold"] = {str(n): self.prompt_file("cold", n) for n in COLD_SIZES}
        self.sizes["m3_off"] = {str(n): self.prompt_file("m3-off", n) for n in M3_OFF_SIZES}
        self.sizes["m3_on"] = {str(n): self.prompt_file("m3-on", n) for n in M3_ON_SIZES}
        self.sizes["helper"] = {str(i): self.prompt_file(f"helper{i}", HELPER_TOKENS) for i in (1, 2)}

    # ------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        if self.gen.exists():
            shutil.rmtree(self.gen)
        sp = self.system_prompt()
        self.write("system_prompt.md", sp + "\n")
        self.write("tools.openai.json", TOOLS)
        self.write("tools.anthropic.json", anthropic_tools(TOOLS))
        self.log("rendering screenshots and their text equivalents ...")
        self.all_images()
        self.write("m1_m5.json", self.m1_m5(sp))
        self.write("m4_trajectory.json", self.m4(sp))
        self.write("m7_suite.json", self.m7(sp))
        self.write("gate.json", {
            "g1": {"prompt": T.G1_PROMPT, "expect": {"name": "fs_read_file", "path": "docs/setup.md"}},
            "g2": {"prompt": T.G2_PROMPT, "image": "g2-verify"}})
        self.log("fitting cold, M3 and helper prompts ...")
        self.prompts()
        manifest = {
            "generator_version": GENERATOR_VERSION, "seed": self.seed, "pillow": PIL.__version__,
            "tokenizer_snapshot": self.p.snap.name, "image_tokens_formula": "h*w/1024 + 2 vision markers",
            "placeholders": {"nonce": NONCE, "request_id": REQUEST_ID, "measured_with": ZERO12,
                             "nonce_at_start": nonce_line(), "nonce_in_first_tool": nonce_tool_prefix()},
            "sizes": self.sizes, "images": self.images,
            "g2_code": self.images["g2-verify"]["code"],
            "files": {str(p.relative_to(self.root)): {"sha256": _sha(p), "bytes": p.stat().st_size}
                      for p in sorted(self.files)},
        }
        _dump(self.root / "manifest.json", manifest)
        return manifest


def build(root: Path, seed: int = DEFAULT_SEED, log: Callable[[str], None] = print) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    return _Builder(root, seed, log).run()


def check(root: Path, log: Callable[[str], None] = print) -> list[str]:
    """Rebuild into a temporary directory and compare every file hash with the manifest."""
    manifest = json.loads((root / "manifest.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build(Path(tmp), manifest["seed"], log=lambda s: None)
    problems = []
    if fresh["pillow"] != manifest["pillow"]:
        problems.append(f"Pillow {fresh['pillow']} differs from the manifest's {manifest['pillow']}")
    for rel, meta in manifest["files"].items():
        got = fresh["files"].get(rel)
        if got is None:
            problems.append(f"{rel}: no longer generated")
        elif got["sha256"] != meta["sha256"]:
            problems.append(f"{rel}: sha256 differs from the manifest")
        p = root / rel
        if not p.exists():
            problems.append(f"{rel}: missing on disk (run `agentbench fixtures build`)")
        elif _sha(p) != meta["sha256"]:
            problems.append(f"{rel}: file on disk differs from the manifest")
    for rel in set(fresh["files"]) - set(manifest["files"]):
        problems.append(f"{rel}: new file not in the manifest")
    return problems


def format_sizes(m: dict[str, Any]) -> str:
    s = m["sizes"]
    m1, m4 = s["m1"], s["m4"]
    rows = [
        ("agent prefix (system turn: tools + system prompt)", s["prefix_tokens"]),
        ("  system prompt text", s["system_prompt_tokens"]),
        ("  tools", s["n_tools"]),
        ("image (1280x1024, incl. vision markers)", s["image_tokens"]),
        ("text equivalent per image", s["image_tokens"]),
        ("tool result (min/median/max)", f"{m1['tool_result_tokens']['min']}/{m1['tool_result_tokens']['median']}/{m1['tool_result_tokens']['max']}"),
        ("M1 warm-up prompt (prefix + task)", m1["warm_prompt_tokens"]),
        ("M1 trial prompt", m1["trial_prompt_tokens"]),
        ("M1 suffix (new tokens per trial)", m1["suffix_tokens"]),
        ("M4 start prompt (prefix + task)", m4["start_prompt_tokens"]),
        ("M4 assistant turn (min/median/max)", f"{m4['assistant_tokens']['min']}/{m4['assistant_tokens']['median']}/{m4['assistant_tokens']['max']}"),
        ("M4 new tokens per step (min/max)", f"{min(x['new_tokens'] for x in m4['steps'])}/{max(x['new_tokens'] for x in m4['steps'])}"),
        ("M4 step-20 prompt", m4["final_prompt_tokens"]),
    ]
    rows += [(f"cold prompt {k}", v) for k, v in s["cold"].items()]
    rows += [(f"M3 prompt, thinking off, {k}", v) for k, v in s["m3_off"].items()]
    rows += [(f"M3 prompt, thinking on, {k}", v) for k, v in s["m3_on"].items()]
    rows += [(f"M5 helper prompt {k}", v) for k, v in s["helper"].items()]
    suf = s["m7_case_suffix_tokens"]
    rows.append(("M7 case suffix beyond the prefix (min/max)", f"{min(suf.values())}/{max(suf.values())}"))
    w = max(len(a) for a, _ in rows)
    return "\n".join(f"{a:<{w}}  {b}" for a, b in rows)
