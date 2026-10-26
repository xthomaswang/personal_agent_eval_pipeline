import os
import socket

import pytest

from agentbench.memory import descendants, parse_footprint, parse_vm_stat
from agentbench.provenance import redact
from agentbench.tokenizer import image_tokens, smart_resize


def test_image_tokens_1280x1024():
    assert smart_resize(1024, 1280) == (1024, 1280)
    assert image_tokens(1280, 1024) == 1280
    assert image_tokens(100, 100) == 64  # upscaled to min_pixels 65536 -> 256x256 -> 64


def test_template_render_and_count():
    try:
        from agentbench.tokenizer import QwenPrompt
        p = QwenPrompt()
    except FileNotFoundError:
        pytest.skip("local Qwen3.8 snapshot not present")
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "hi"}]}]
    text = p.render(msgs, tools, enable_thinking=False)
    assert text.startswith("<|im_start|>system\n# Tools") and text.endswith("<think>\n\n</think>\n\n")
    assert p.count(msgs, tools) == p.count_text(text) + 1279


def test_memory_parsers():
    assert descendants(1, "1 0\n2 1\n3 2\n4 9\n") == [1, 2, 3]
    assert parse_footprint("zsh [39077]: 64-bit    Footprint: 2688 KB (16384 bytes per page)") == {39077: 2688 * 1024}
    vm = parse_vm_stat("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\n"
                       "Pages wired down: 5.\nPages occupied by compressor: 1.\nAnonymous pages: 20.\nPages purgeable: 2.\n")
    assert vm["used_bytes"] == (18 + 5 + 1) * 16384


def test_redact_paths():
    s = redact(os.path.expanduser("~/x") + " " + socket.gethostname())
    assert s.startswith("~/x")
