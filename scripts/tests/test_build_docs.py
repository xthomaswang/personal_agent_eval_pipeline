"""Focused website checks: public evidence, cohort isolation, and reproducible outputs."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_docs
from site_data import build_payload, capability_data, ui_data


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "img"} and attrs.get("src"):
            self.assets.append(attrs["src"])
        if tag == "link" and attrs.get("href"):
            self.assets.append(attrs["href"])


class WebsiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_docs.load()
        build_docs.tables(cls.rows)
        cls.payload = build_payload(build_docs.ROOT, cls.rows, build_docs.REPO_URL)

    def row(self, family, protocol):
        return next(r for r in self.rows if r["family"] == family and r["protocol"] == protocol
                    and r["run_id"] == "20261027-ui-real")

    def test_latest_ui_uses_audited_gold_and_keeps_all_input_arms(self):
        row = self.row("qwen3.8_27b", "ui-real-1")
        self.assertEqual(row["_ui"]["replica:vision"]["fill"], [46, 72])
        self.assertEqual(row["_ui"]["replica:vision+ocr"]["fill"], [52, 72])
        self.assertEqual(row["_ui"]["capture:vision"]["read_lenient"], [584, 624])
        self.assertNotIn("click", row["_ui"]["capture:vision"])
        self.assertEqual(row["_ui_meta"], {"pages": 75, "items": {"click": 198, "fill": 24, "read": 208},
                                           "totalItems": 430, "samples": 3})
        flash = self.row("qwen3.8_flash_next", "ui-real-1")
        self.assertEqual(flash["_ui"]["replica:ax"]["click"], [586, 594])
        self.assertEqual(flash["_ui_source"], row["_ui_source"])

    def test_ui_counts_must_agree_with_published_summary(self):
        row = copy.deepcopy(self.row("qwen3.8_27b", "ui-real-1"))
        row["metrics"][0]["value"] = "1/594"
        with self.assertRaisesRegex(ValueError, "disagrees with the summary"):
            ui_data(build_docs.ROOT, [row])

    def test_protocols_remain_separate_and_partial_harnesses_remain_missing(self):
        ui_group = next(g for g in self.payload["protocols"] if g["id"] == "ui-real-1")
        self.assertEqual(ui_group["kind"], "ui")
        self.assertTrue(all(r["suite"] == "agent_ability" for r in self.rows if r["protocol"] == ui_group["id"]))
        partial = next(r for r in self.rows if r["family"] == "nanbeige4.3_33b" and r["protocol"] == "capability-1")
        self.assertEqual(partial["_harnesses"]["hermes"]["gate"]["runs"], 18)
        self.assertEqual(partial["_harnesses"]["localmelo"]["ranking"]["runs"], 0)
        self.assertIsNone(partial["_harnesses"]["localmelo"]["ranking"]["modelTimeMedian"])
        self.assertTrue(set(self.payload["taskSelection"]["gate"]).isdisjoint(self.payload["taskSelection"]["ranking"]))

    def test_capability_versions_are_enriched_and_checked(self):
        original = next(r for r in self.rows if r["protocol"] == "capability-1" and r.get("reference"))
        other = copy.deepcopy(original)
        other["protocol"] = "capability-2"
        tasks = capability_data(build_docs.ROOT, [other])
        self.assertEqual(len(tasks["gate"]), 22)
        self.assertEqual(other["_harnesses"], original["_harnesses"])
        next(m for m in other["metrics"] if m["name"] == "Gate set")["value"] = "0/66"
        with self.assertRaisesRegex(ValueError, "disagrees with the summary"):
            capability_data(build_docs.ROOT, [other])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI interaction test")
    def test_capability_protocol_selection_updates_harness_comparison(self):
        script = Path(__file__).with_name("test_capability_ui.js")
        subprocess.run([shutil.which("node"), str(script), str(build_docs.ROOT / "docs/assets/app.js")], check=True)

    def test_build_is_deterministic_and_has_no_commit_self_reference(self):
        first = build_docs.output_files()
        self.assertEqual(first, build_docs.output_files())
        self.assertNotIn("sourceCommit", self.payload)
        self.assertEqual(len(self.payload["sourceDigest"]), 64)
        self.assertNotIn(b"{{", first["index.html"])

    def test_all_page_assets_exist_and_check_detects_stale_assets(self):
        files = build_docs.output_files()
        for page in ("index", "pipeline", "methodology", "reproduce"):
            parser = AssetParser()
            parser.feed(files[f"{page}.html"].decode())
            for url in parser.assets:
                self.assertIn(urlsplit(url).path, files, (page, url))
            self.assertIn(f'data-page="{build_docs.PAGES[page][0]}"'.encode(), files[f"{page}.html"])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            build_docs.write_outputs(files, output)
            self.assertEqual(build_docs.write_outputs(files, output, check=True), [])
            (output / "assets/site.css").write_text("stale")
            self.assertEqual(build_docs.write_outputs(files, output, check=True), ["assets/site.css"])
            self.assertEqual((output / "assets/site.css").read_text(), "stale")
            (output / "assets/old.js").write_text("left over")
            self.assertEqual(build_docs.write_outputs(files, output, check=True), ["assets/site.css", "assets/old.js"])
            build_docs.write_outputs(files, output)
            self.assertFalse((output / "assets/old.js").exists())
            self.assertEqual(build_docs.write_outputs(files, output, check=True), [])


if __name__ == "__main__":
    unittest.main()
